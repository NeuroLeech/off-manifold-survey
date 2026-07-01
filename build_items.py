"""build_items.py — build the data-collection item pool (dev-time only).

Produces `collection_items.csv`, the items presented to participants. Two
sources, unioned:

1. **Codebook items in CNP space** — rows with `USE == TRUE` whose
   `item_text_clean` is present in `cnp_embeddings.csv`. Responses to these map
   straight back onto the 32-d surrogate for retraining.
2. **New items** — every row with `dataset == 'New'`, regardless of `USE` or CNP
   membership. These are freshly added measures (tagged in the `Extra` column,
   e.g. MDES / BDI / ESS) that have no data yet; we include them so this study
   collects their first responses. They are not in CNP space and have no
   `item_text_clean`, so a stable `item_id` is synthesised from `item_text`.

Each row carries the human-facing question wording plus its own native response
scale (the ordered value/label pairs), so the survey renders each item with its
correct anchors. Run once to regenerate the committed CSV:

    python collection_app/build_items.py

The deployed app reads only the resulting CSV — it never needs the codebook.
"""
from __future__ import annotations

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CODEBOOK = os.path.join(DATA, 'master_codebook.csv')
CNP = os.path.join(DATA, 'cnp_embeddings.csv')
OUT = os.path.join(HERE, 'collection_items.csv')

MIN_OPTIONS, MAX_OPTIONS = 2, 10


def _cell(r, col) -> str:
    v = r.get(col)
    return '' if pd.isna(v) else str(v).strip()


def _options(r) -> list[dict] | None:
    """Ordered value/label pairs for the item, or None if the scale is invalid."""
    try:
        n = int(float(r['n_options']))
    except (TypeError, ValueError):
        return None
    if not (MIN_OPTIONS <= n <= MAX_OPTIONS):
        return None
    opts = []
    for k in range(1, n + 1):
        val, lab = _cell(r, f'value_{k}'), _cell(r, f'label_{k}')
        if val == '' or lab == '':
            return None
        opts.append({'value': val, 'label': lab})
    return opts


def main() -> None:
    cb = pd.read_csv(CODEBOOK, header=1, dtype=str)
    cb.columns = [c.strip() for c in cb.columns]
    cnp_set = set(pd.read_csv(CNP)['item_prompt'].astype(str).str.strip())

    in_cnp = cb['item_text_clean'].astype(str).str.strip().isin(cnp_set)
    is_new = cb['dataset'].astype(str).str.strip() == 'New'
    pool = cb[((cb['USE'] == 'TRUE') & in_cnp) | is_new].copy()

    rows, seen = [], set()
    for _, r in pool.iterrows():
        clean = _cell(r, 'item_text_clean')          # CNP / join key ('' for New)
        item_id = clean or _cell(r, 'item_text')     # stable id (New has no clean)
        if not item_id or item_id in seen:
            continue
        opts = _options(r)
        if opts is None:
            continue
        # Prefer curated wording, then the dataset's full_question, then the id.
        question = (_cell(r, 'manual_full_question') or _cell(r, 'full_question')
                    or item_id)
        rows.append({
            'item_id': item_id,
            'item_text_clean': clean,
            'dataset': _cell(r, 'dataset'),
            'extra': _cell(r, 'Extra'),
            'question': question,
            'n_options': len(opts),
            'options_json': json.dumps(opts, ensure_ascii=False),
        })
        seen.add(item_id)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'wrote {len(out)} items -> {OUT}')
    print('by source:')
    print(f'  CNP/codebook : {(out["dataset"] != "New").sum()}')
    print(f'  New          : {(out["dataset"] == "New").sum()}')
    if (out['dataset'] == 'New').any():
        print('  New by Extra :',
              out[out['dataset'] == 'New']['extra'].value_counts().to_dict())
    print('options-count distribution:')
    print(out['n_options'].value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()
