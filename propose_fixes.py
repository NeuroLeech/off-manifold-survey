"""propose_fixes.py — propose a minimal fix for each QA-flagged item (dev-time).

Reads `qa_report.csv`, takes the items flagged minor/major, and asks the local
model for the SMALLEST edit that resolves the flagged issue while preserving
meaning. Writes `qa_proposed_fixes.csv` with the original alongside a proposed
question and (when the scale is the problem) proposed options — for you to
review and approve. Nothing is written back to the codebook.

Fix types the model may return:
  prepend_I   — stem-less self-report fragment -> grammatical first-person item
  reword      — minor grammar/typo/encoding fix
  fix_options — corrected response options (e.g. duplicate/garbled label)
  add_context — needs a short framing line (returned in `note`)
  accept      — fine as a standard questionnaire convention; no change

Resumable and single-writer (same design as qa_items.py). Run:
    python collection_app/propose_fixes.py --workers 4
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
QA_CSV = os.path.join(HERE, 'qa_report.csv')

DEFAULT_MODEL = 'qwen3:32b'
DEFAULT_HOST = 'http://localhost:11434'

OUT_COLUMNS = [
    'item_id', 'dataset', 'extra', 'severity', 'issue',
    'question', 'proposed_question', 'scale', 'proposed_options',
    'fix_type', 'needs_fix', 'note', 'model',
]

SYSTEM = (
    'You are a careful survey-methodology editor. A questionnaire item shown to '
    'participants (with a fixed response scale and the instruction to indicate '
    'how much it describes them) was flagged in review. Propose the SMALLEST '
    'edit that fixes the flagged issue while preserving the original meaning and '
    'difficulty. Do not rewrite good items. For a first-person self-report '
    'fragment missing its subject (common IPIP style, e.g. "Tire out quickly"), '
    'prepend "I" and fix capitalisation to make a grammatical statement. If the '
    'response options are the problem, give corrected options. If the item is '
    'genuinely acceptable as a standard convention, accept it unchanged.'
)

PROMPT = """A reviewer flagged this item.

FLAGGED ISSUE: {issue}

QUESTION:
{question}

RESPONSE OPTIONS (in order):
{scale}

Return ONLY a JSON object:
{{"needs_fix": <true|false>, "fix_type": "<prepend_I|reword|fix_options|add_context|accept>", "proposed_question": "<the full corrected question; repeat the original unchanged if no change>", "proposed_options": "<corrected options separated by ' | ', or empty string if the scale is unchanged>", "note": "<=12 words explaining the change or why none>"}}"""


def propose(row: dict, model: str, host: str, timeout: int = 180) -> dict:
    body = {
        'model': model,
        'prompt': PROMPT.format(issue=row['issue'] or '(general)',
                                question=row['question'], scale=row['scale']),
        'system': SYSTEM,
        'stream': False,
        'format': 'json',
        'think': False,
        'options': {'temperature': 0.0, 'num_predict': 320},
    }
    r = requests.post(f'{host}/api/generate', json=body, timeout=timeout)
    r.raise_for_status()
    raw = r.json().get('response', '').strip()
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        return {'needs_fix': '', 'fix_type': 'error', 'proposed_question': '',
                'proposed_options': '', 'note': f'unparseable: {raw[:50]}'}
    return {
        'needs_fix': bool(v.get('needs_fix', '')),
        'fix_type': str(v.get('fix_type', '')),
        'proposed_question': str(v.get('proposed_question', '')).strip(),
        'proposed_options': str(v.get('proposed_options', '')).strip(),
        'note': str(v.get('note', '')).strip(),
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
    ap.add_argument('--qa', default=QA_CSV)
    ap.add_argument('--out', default=os.path.join(HERE, 'qa_proposed_fixes.csv'))
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    qa = pd.read_csv(args.qa, dtype=str).fillna('')
    flagged = qa[qa['severity'].isin(['major', 'minor'])]
    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    done = load_done(out_path)
    todo = [r for _, r in flagged.iterrows() if r['item_id'] not in done]
    print(f'{len(todo)} flagged items to fix (skipping {len(done)} done) '
          f'with {args.model} x{args.workers} -> {out_path}')

    new_file = not os.path.exists(out_path)
    fh = open(out_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
    if new_file:
        writer.writeheader()
        fh.flush()

    def work(row) -> dict:
        try:
            fix = propose(row, args.model, args.host)
        except Exception as e:  # noqa: BLE001
            fix = {'needs_fix': '', 'fix_type': 'error', 'proposed_question': '',
                   'proposed_options': '', 'note': f'request-failed: {e}'}
        return {
            'item_id': row['item_id'], 'dataset': row['dataset'],
            'extra': row.get('extra', ''), 'severity': row['severity'],
            'issue': row['issue'], 'question': row['question'],
            'scale': row['scale'], 'model': args.model, **fix,
        }

    done_n = changed = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, r) for r in todo]):
            row = fut.result()
            writer.writerow(row)
            fh.flush()
            done_n += 1
            changed += row['needs_fix'] is True
            errors += row['fix_type'] == 'error'
            if done_n % 25 == 0 or done_n == len(todo):
                print(f'  {done_n}/{len(todo)} (needs_fix {changed}, errors {errors})')
    fh.close()
    print(f'done. {done_n} proposed, {changed} need a fix, {errors} errors '
          f'-> {out_path}')


if __name__ == '__main__':
    main()
