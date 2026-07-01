"""apply_fixes.py — apply approved QA fixes back into master_codebook.csv.

Reads the reviewed `qa_proposed_fixes.csv` (rows with needs_fix=TRUE) and applies:
  * question changes -> written to the `manual_full_question` column (build_items
    already prefers it, and item_text_clean / the CNP anchor is left untouched).
  * scale changes    -> rewrites value_k / label_k / n_options, preserving each
    item's numbering base (0- or 1-based).

Matching: qa `item_id` == codebook `item_text_clean` (or `item_text` for New
items). Multi-match items are updated on every matching row and reported.

The codebook's exact structure (title row 0, header row 1) is preserved by
operating on the raw rows. Always writes a `fixes_applied_log.csv` audit trail.

    python collection_app/apply_fixes.py --dry-run   # preview, no writes
    python collection_app/apply_fixes.py             # apply
"""
from __future__ import annotations

import argparse
import csv
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODEBOOK = os.path.join(HERE, '..', 'data', 'master_codebook.csv')
FIXES = os.path.join(HERE, 'qa_proposed_fixes.csv')
LOG = os.path.join(HERE, 'fixes_applied_log.csv')
MAXK = 10


def _labs(s: str) -> list[str]:
    return [x.strip() for x in str(s).split('|') if x.strip()]


def _norm(s: str) -> str:
    return ' | '.join(_labs(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--codebook', default=CODEBOOK)
    ap.add_argument('--fixes', default=FIXES)
    args = ap.parse_args()

    # raw rows: row 0 = title, row 1 = header, 2+ = data
    with open(args.codebook, newline='', encoding='utf-8') as f:
        raw = list(csv.reader(f))
    header = [h.strip() for h in raw[1]]
    col = {name: i for i, name in enumerate(header)}
    data_start = 2

    def cell(r: int, name: str) -> str:
        return raw[r][col[name]].strip()

    def setcell(r: int, name: str, val: str) -> None:
        raw[r][col[name]] = val

    # index codebook rows by their match key
    clean_rows: dict[str, list[int]] = {}
    text_rows: dict[str, list[int]] = {}
    for r in range(data_start, len(raw)):
        c = cell(r, 'item_text_clean')
        t = cell(r, 'item_text')
        if c:
            clean_rows.setdefault(c, []).append(r)
        if t:
            text_rows.setdefault(t, []).append(r)

    fixes = pd.read_csv(args.fixes).fillna('')
    approved = fixes[fixes['needs_fix'].astype(str).str.strip().str.lower() == 'true']

    log = []
    q_applied = s_applied = rows_touched = multi = missing = 0
    for _, fx in approved.iterrows():
        iid = str(fx['item_id']).strip()
        is_new = str(fx['dataset']).strip() == 'New'
        idxs = (text_rows if is_new else clean_rows).get(iid, [])
        if not idxs:
            missing += 1
            log.append({'item_id': iid, 'action': 'NO_MATCH', 'detail': ''})
            continue
        if len(idxs) > 1:
            multi += 1

        new_q = str(fx['proposed_question']).strip()
        q_change = bool(new_q) and new_q != str(fx['question']).strip()
        new_opts = _labs(fx['proposed_options'])
        s_change = bool(new_opts) and _norm(fx['proposed_options']) != _norm(fx['scale'])

        for r in idxs:
            touched = False
            if q_change:
                old = cell(r, 'manual_full_question')
                setcell(r, 'manual_full_question', new_q)
                log.append({'item_id': iid, 'action': 'question',
                            'detail': f'{old!r} -> {new_q!r}'})
                q_applied += 1
                touched = True
            if s_change:
                base = 0 if cell(r, 'value_1') == '0' else 1
                for k in range(1, MAXK + 1):     # clear existing scale
                    if f'value_{k}' in col:
                        setcell(r, f'value_{k}', '')
                    if f'label_{k}' in col:
                        setcell(r, f'label_{k}', '')
                for k, lab in enumerate(new_opts, start=1):
                    setcell(r, f'value_{k}', str(base + k - 1))
                    setcell(r, f'label_{k}', lab)
                setcell(r, 'n_options', str(len(new_opts)))
                log.append({'item_id': iid, 'action': 'scale',
                            'detail': f'{_norm(fx["scale"])} -> {_norm(fx["proposed_options"])}'})
                s_applied += 1
                touched = True
            rows_touched += touched

    pd.DataFrame(log).to_csv(LOG, index=False)
    mode = 'DRY-RUN (no writes)' if args.dry_run else 'APPLIED'
    print(f'=== {mode} ===')
    print(f'approved items          : {len(approved)}')
    print(f'question changes applied: {q_applied}')
    print(f'scale changes applied   : {s_applied}')
    print(f'codebook rows touched   : {rows_touched}')
    print(f'multi-match items        : {multi}')
    print(f'unmatched (skipped)      : {missing}')
    print(f'audit log -> {LOG}')

    if not args.dry_run:
        with open(args.codebook, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(raw)
        print(f'wrote {args.codebook}')


if __name__ == '__main__':
    main()
