"""qa_items.py — LLM quality check of the collection items (dev-time only).

Runs each item's participant-facing `question` plus its actual response scale
past a local Ollama model and asks for a strict JSON verdict on three criteria:

  * makes_sense — coherent, grammatical, unambiguous as a self-report item
  * answerable  — the given response scale genuinely fits the question
  * standalone  — works on its own, without any preamble/stem/context

Output is `qa_report.csv` (one row per item) you can sort by `severity`.

The pass is **resumable**: already-judged item_ids in the existing report are
skipped, so you can stop/restart or extend a sample to the full set. Requests
can run a few at a time (`--workers`).

Examples
--------
    # 30-item calibration sample
    python collection_app/qa_items.py --sample 30 --out qa_sample.csv

    # full pass, 3 parallel, in the background
    python collection_app/qa_items.py --workers 3

Requires `ollama serve` with the chosen model pulled (default qwen3:32b).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS_CSV = os.path.join(HERE, 'collection_items.csv')

DEFAULT_MODEL = 'qwen3:32b'
DEFAULT_HOST = 'http://localhost:11434'

OUT_COLUMNS = [
    'item_id', 'dataset', 'extra', 'question', 'scale',
    'makes_sense', 'answerable', 'standalone', 'severity', 'issue', 'model',
]

SYSTEM = (
    'You are a meticulous survey-methodology reviewer. You judge a single '
    'self-report questionnaire item that will be shown to a participant with a '
    'fixed response scale. The participant is told only to indicate how much '
    'each statement describes them. Be strict but fair: a plain trait statement '
    'like "I act wild and crazy" rated on agree/disagree is fine and stands '
    'alone. Flag items that are garbled, double-barrelled, depend on missing '
    'context, or whose response scale does not fit the question.'
)

PROMPT = """Evaluate this questionnaire item.

QUESTION:
{question}

RESPONSE OPTIONS (in order):
{scale}

Judge three things:
- makes_sense: the item is coherent, grammatical, and unambiguous.
- answerable: the response options above genuinely fit the question.
- standalone: it works on its own, with no preamble, stem, or extra context.

Respond with ONLY a JSON object, no other text:
{{"makes_sense": <true|false>, "answerable": <true|false>, "standalone": <true|false>, "severity": "<none|minor|major>", "issue": "<short reason, <=15 words; empty if none>"}}"""


def _scale_str(options: list[dict]) -> str:
    return '\n'.join(f'  {o["value"]}. {o["label"]}' for o in options)


def judge(item: dict, model: str, host: str, timeout: int = 180) -> dict:
    scale = _scale_str(item['options'])
    body = {
        'model': model,
        'prompt': PROMPT.format(question=item['question'], scale=scale),
        'system': SYSTEM,
        'stream': False,
        'format': 'json',
        'think': False,                 # disable Qwen3 thinking for speed/clean JSON
        'options': {'temperature': 0.0, 'num_predict': 200},
    }
    r = requests.post(f'{host}/api/generate', json=body, timeout=timeout)
    r.raise_for_status()
    raw = r.json().get('response', '').strip()
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        return {'makes_sense': '', 'answerable': '', 'standalone': '',
                'severity': 'error', 'issue': f'unparseable: {raw[:60]}'}
    return {
        'makes_sense': bool(v.get('makes_sense', '')),
        'answerable': bool(v.get('answerable', '')),
        'standalone': bool(v.get('standalone', '')),
        'severity': str(v.get('severity', '')),
        'issue': str(v.get('issue', '')).strip(),
    }


def _row_for(item: dict, verdict: dict, model: str) -> dict:
    return {
        'item_id': item['item_id'], 'dataset': item['dataset'],
        'extra': item.get('extra', ''), 'question': item['question'],
        'scale': ' | '.join(o['label'] for o in item['options']),
        'model': model, **verdict,
    }


def load_done(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    try:
        s = pd.read_csv(out_path, dtype=str, on_bad_lines='skip')['item_id']
        return set(s.dropna())
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--host', default=DEFAULT_HOST)
    ap.add_argument('--out', default=os.path.join(HERE, 'qa_report.csv'))
    ap.add_argument('--sample', type=int, default=0,
                    help='judge a random N-item sample (0 = all)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(ITEMS_CSV, dtype=str).fillna('')
    df['options'] = df['options_json'].apply(json.loads)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    done = load_done(out_path)
    todo = [r for _, r in df.iterrows() if r['item_id'] not in done]
    print(f'{len(todo)} items to judge (skipping {len(done)} done) '
          f'with {args.model} x{args.workers} -> {out_path}')

    new_file = not os.path.exists(out_path)
    fh = open(out_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
    if new_file:
        writer.writeheader()
        fh.flush()

    def work(item) -> dict:
        try:
            verdict = judge(item, args.model, args.host)
        except Exception as e:  # noqa: BLE001
            verdict = {'makes_sense': '', 'answerable': '', 'standalone': '',
                       'severity': 'error', 'issue': f'request-failed: {e}'}
        return _row_for(item, verdict, args.model)

    # Workers only do the (slow) LLM call; the main thread is the sole writer,
    # so concurrent rows can never interleave in the CSV.
    done = flagged = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, it) for it in todo]):
            row = fut.result()
            writer.writerow(row)
            fh.flush()
            done += 1
            flagged += row['severity'] in ('minor', 'major')
            errors += row['severity'] == 'error'
            if done % 25 == 0 or done == len(todo):
                print(f'  {done}/{len(todo)} (flagged {flagged}, errors {errors})')
    fh.close()
    print(f'done. {done} judged, {flagged} flagged, {errors} errors -> {out_path}')


if __name__ == '__main__':
    main()
