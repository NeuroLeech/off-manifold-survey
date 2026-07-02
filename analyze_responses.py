"""analyze_responses.py — tidy collected responses and merge onto CNP space.

Pulls the survey responses (from the Google Sheet, or a CSV export / the local
fallback file), splits them into items / attention-checks / demographics, and
produces analysis-ready tables:

  * tidy_responses.csv   — one row per (session, item) for kind=='item', with a
    numeric `response` and the item's 32-d CNP coordinate merged in (emb_0..31),
    joined on item_text_clean. New/off-manifold items have no CNP coord (blank).
  * session_summary.csv  — per participant: items answered, attention-checks
    passed, demographics, prolific_pid, and a `usable` flag (all checks passed).

Read from the Sheet with a service-account JSON, or from a CSV you exported
(Google Sheets → File → Download → CSV) / the dev `local_responses.csv`:

    python collection_app/analyze_responses.py --sa key.json --url <sheet-url>
    python collection_app/analyze_responses.py --csv responses_export.csv

CNP coordinates are read from ../data/cnp_embeddings.csv.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CNP = os.path.join(HERE, '..', 'data', 'cnp_embeddings.csv')


def load_from_sheet(sa_json: str, url: str, worksheet: str) -> pd.DataFrame:
    import gspread
    gc = gspread.service_account(filename=sa_json)
    ws = gc.open_by_url(url).worksheet(worksheet)
    return pd.DataFrame(ws.get_all_records())


def build(df: pd.DataFrame, outdir: str) -> None:
    df = df.fillna('')
    df['kind'] = df.get('kind', 'item').replace('', 'item')
    items = df[df['kind'] == 'item'].copy()
    checks = df[df['kind'] == 'attention'].copy()
    demos = df[df['kind'] == 'demographic'].copy()

    # ---- per-session summary ----
    ans = items.assign(answered=items['response_label'].astype(str).str.strip() != '')
    summ = ans.groupby('session_id').agg(
        items_shown=('item_id', 'size'),
        items_answered=('answered', 'sum')).reset_index()
    if len(checks):
        checks['passed'] = checks['check_passed'].astype(str).str.lower() == 'true'
        cs = checks.groupby('session_id').agg(
            checks_total=('passed', 'size'),
            checks_passed=('passed', 'sum')).reset_index()
        summ = summ.merge(cs, on='session_id', how='left')
        summ['usable'] = summ['checks_passed'] == summ['checks_total']
    if len(demos):
        wide = demos.pivot_table(index='session_id', columns='item_id',
                                 values='response_label', aggfunc='first')
        summ = summ.merge(wide.reset_index(), on='session_id', how='left')
    for c in ('prolific_pid', 'timestamp_utc'):
        if c in df.columns:
            first = df.groupby('session_id')[c].first().reset_index()
            summ = summ.merge(first, on='session_id', how='left')
    summ.to_csv(os.path.join(outdir, 'session_summary.csv'), index=False)

    # ---- tidy item responses + CNP coords ----
    items['response'] = pd.to_numeric(items['response_value'], errors='coerce')
    tidy = items[items['response_label'].astype(str).str.strip() != ''].copy()
    keep = ['session_id', 'prolific_pid', 'item_id', 'item_text_clean',
            'dataset', 'extra', 'question', 'response', 'response_label',
            'n_options']
    tidy = tidy[[c for c in keep if c in tidy.columns]]

    cnp = pd.read_csv(CNP)
    emb_cols = [c for c in cnp.columns if c.startswith('emb_')]
    tidy = tidy.merge(cnp[['item_prompt'] + emb_cols],
                      left_on='item_text_clean', right_on='item_prompt',
                      how='left').drop(columns=['item_prompt'])
    tidy.to_csv(os.path.join(outdir, 'tidy_responses.csv'), index=False)

    n_sess = tidy['session_id'].nunique()
    on_manifold = tidy[emb_cols[0]].notna().sum() if emb_cols else 0
    print(f'sessions: {n_sess} | item responses: {len(tidy)} '
          f'({on_manifold} with a CNP coord, {len(tidy) - on_manifold} off-manifold)')
    if 'usable' in summ.columns:
        print(f'usable sessions (all attention checks passed): '
              f'{int(summ["usable"].sum())} / {len(summ)}')
    print(f'wrote tidy_responses.csv + session_summary.csv -> {outdir}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', help='responses CSV export / local_responses.csv')
    ap.add_argument('--sa', help='service-account JSON (to read the Sheet)')
    ap.add_argument('--url', help='Google Sheet URL (with --sa)')
    ap.add_argument('--worksheet', default='responses_v2')
    ap.add_argument('--outdir', default=HERE)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, dtype=str)
    elif args.sa and args.url:
        df = load_from_sheet(args.sa, args.url, args.worksheet)
    else:
        ap.error('provide --csv PATH, or --sa key.json --url <sheet-url>')
    build(df, args.outdir)


if __name__ == '__main__':
    main()
