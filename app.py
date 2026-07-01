"""app.py — participant data-collection survey (Streamlit Community Cloud).

Self-contained: reads only `collection_items.csv` (built by `build_items.py`
from the master codebook ∩ CNP space) and writes responses via `storage.py`.
No ML dependencies, so it deploys as a slim standalone app.

Flow: welcome + consent → N randomly sampled items, each shown with its own
native response scale, paged a few at a time with a progress bar → submit, which
appends one row per response (keyed by `item_text_clean` so the data maps back
onto the CNP surrogate).

Run locally:
    streamlit run collection_app/app.py
"""
from __future__ import annotations

import json
import os
import random
import uuid

import pandas as pd
import streamlit as st

from storage import save_responses, utc_now

APP_VERSION = 'collect-v1'
N_ITEMS = 50            # items sampled from the main pool per participant
MDES_PER_SESSION = 4    # guaranteed random MDES items, added on top of N_ITEMS
PAGE_SIZE = 10          # items shown per page

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS_CSV = os.path.join(HERE, 'collection_items.csv')

# --- Consent / welcome copy. Replace the bracketed parts with your study's
# --- ethics-approved wording and contact details before collecting real data.
WELCOME_MD = """
### Thank you for taking part

In this short study you'll see a series of everyday statements. For each one,
please indicate **how much it describes you**, using the scale shown beneath it.

- There are no right or wrong answers — we're interested in your honest response.
- It takes about **8–12 minutes** ({n} statements).
- Your responses are **anonymous**: we do not collect your name, email, or any
  identifying information.

---

| | |
|---|---|
| **Study** | Off-Manifold: psychometric survey building |
| **Institution** | King's College London |
| **Researcher** | Professor Robert Leech ([robert.leech@kcl.ac.uk](mailto:robert.leech@kcl.ac.uk)) |
| **Ethics reference** | MRA-25/26-57941 |

If you have any questions about this study, please contact the researcher at the
email above.
"""

CONSENT_LABEL = ('I have read the information above and agree to take part. '
                 'I understand my responses are anonymous and will be used for '
                 'research.')


@st.cache_data(show_spinner=False)
def load_pool() -> pd.DataFrame:
    df = pd.read_csv(ITEMS_CSV, dtype=str).fillna('')   # blank, not 'nan', for New
    df['n_options'] = df['n_options'].astype(int)
    df['options'] = df['options_json'].apply(json.loads)
    return df


def _init_state():
    ss = st.session_state
    ss.setdefault('stage', 'welcome')
    ss.setdefault('session_id', uuid.uuid4().hex)
    ss.setdefault('page', 0)
    ss.setdefault('items', None)


def _split_pool(pool: pd.DataFrame):
    is_mdes = pool['extra'].astype(str) == 'MDES'
    return pool[~is_mdes], pool[is_mdes]


def _session_size(pool: pd.DataFrame) -> int:
    main, mdes = _split_pool(pool)
    return min(N_ITEMS, len(main)) + min(MDES_PER_SESSION, len(mdes))


def _sample_items(pool: pd.DataFrame, session_id: str) -> list[dict]:
    """N_ITEMS random items from the main pool plus MDES_PER_SESSION guaranteed
    random MDES items, shuffled together. Seeded by session_id for reproducibility."""
    rng = random.Random(session_id)
    main, mdes = _split_pool(pool)
    chosen = main.iloc[rng.sample(range(len(main)), min(N_ITEMS, len(main)))] \
        .to_dict('records')
    n_mdes = min(MDES_PER_SESSION, len(mdes))
    if n_mdes:
        chosen += mdes.iloc[rng.sample(range(len(mdes)), n_mdes)].to_dict('records')
    rng.shuffle(chosen)
    return chosen


def _answered_count(items: list[dict]) -> int:
    return sum(1 for i in range(len(items))
              if st.session_state.get(f'resp_{i}') is not None)


def render_welcome(pool: pd.DataFrame):
    st.title('Off-Manifold: psychometric survey building')
    st.markdown(WELCOME_MD.format(n=_session_size(pool)))
    agree = st.checkbox(CONSENT_LABEL, key='consent')
    if st.button('Begin', type='primary', disabled=not agree):
        st.session_state['items'] = _sample_items(pool, st.session_state['session_id'])
        st.session_state['stage'] = 'survey'
        st.session_state['page'] = 0
        st.rerun()


def _render_item(i: int, item: dict):
    labels = [o['label'] for o in item['options']]
    st.markdown(f"**{item['question']}**")
    st.radio('response', labels, index=None, key=f'resp_{i}',
             horizontal=len(labels) <= 7, label_visibility='collapsed')
    st.divider()


def render_survey():
    items = st.session_state['items']
    n = len(items)
    page = st.session_state['page']
    n_pages = (n + PAGE_SIZE - 1) // PAGE_SIZE
    start, end = page * PAGE_SIZE, min((page + 1) * PAGE_SIZE, n)

    answered = _answered_count(items)
    st.progress(answered / n, text=f'{answered} / {n} answered')
    st.caption(f'Page {page + 1} of {n_pages}')

    for i in range(start, end):
        _render_item(i, items[i])

    page_unanswered = [i for i in range(start, end)
                       if st.session_state.get(f'resp_{i}') is None]

    cols = st.columns(2)
    if page > 0 and cols[0].button('← Back'):
        st.session_state['page'] -= 1
        st.rerun()

    last = page == n_pages - 1
    label = 'Submit' if last else 'Next →'
    if cols[1].button(label, type='primary'):
        if page_unanswered:
            st.warning('Please answer every statement on this page before '
                       'continuing.')
        elif last:
            _submit(items)
        else:
            st.session_state['page'] += 1
            st.rerun()


def _submit(items: list[dict]):
    ss = st.session_state
    ts = utc_now()
    rows = []
    for i, item in enumerate(items):
        chosen_label = ss.get(f'resp_{i}')
        value = next((o['value'] for o in item['options']
                      if o['label'] == chosen_label), '')
        rows.append({
            'session_id': ss['session_id'], 'timestamp_utc': ts,
            'consent': bool(ss.get('consent')), 'position': i,
            'item_id': item['item_id'],
            'item_text_clean': item['item_text_clean'],
            'dataset': item['dataset'], 'extra': item.get('extra', ''),
            'question': item['question'],
            'response_value': value, 'response_label': chosen_label,
            'n_options': item['n_options'], 'app_version': APP_VERSION,
        })
    with st.spinner('Saving your responses…'):
        backend, detail = save_responses(rows)
    ss['save_backend'] = backend
    ss['save_detail'] = detail
    ss['stage'] = 'done'
    st.rerun()


def render_done():
    st.title('All done — thank you! 🎉')
    st.markdown('Your responses have been recorded. You may now close this tab.')
    backend = st.session_state.get('save_backend')
    if backend == 'local':
        st.info('Note: responses were saved to a local file (the cloud Sheet '
                'is not configured). Detail: '
                f'`{st.session_state.get("save_detail")}`')
    st.caption(f'Session: {st.session_state["session_id"][:8]}')


def main():
    st.set_page_config(page_title='Off-Manifold survey', page_icon='📝')
    _init_state()
    # Preserve answered radios across page navigation: Streamlit drops the
    # session_state entry for any widget not rendered in the current run, so
    # without this "touch" only the current page's answers would survive.
    for k in list(st.session_state.keys()):
        if k.startswith('resp_'):
            st.session_state[k] = st.session_state[k]
    pool = load_pool()
    stage = st.session_state['stage']
    if stage == 'welcome':
        render_welcome(pool)
    elif stage == 'survey':
        render_survey()
    else:
        render_done()


if __name__ == '__main__':
    main()
