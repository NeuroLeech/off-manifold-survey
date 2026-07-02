"""app.py — participant data-collection survey (Streamlit Community Cloud).

Self-contained: reads only `collection_items.csv` (built by `build_items.py`
from the master codebook ∩ CNP space) and writes responses via `storage.py`.
No ML dependencies, so it deploys as a slim standalone app.

Flow: welcome + consent → short demographics → items (main sample + guaranteed
MDES + interspersed attention checks), each shown with its own native response
scale, paged a few at a time → done page with a completion code. Prolific URL
parameters (PROLIFIC_PID / STUDY_ID / SESSION_ID) are captured and stored.

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
import streamlit.components.v1 as components

from storage import save_responses, utc_now

APP_VERSION = 'collect-v2'
N_ITEMS = 50            # items sampled from the main pool per participant
MDES_PER_SESSION = 4    # guaranteed random MDES items, added on top
N_ATTENTION = 3         # interspersed instructed-response attention checks
PAGE_SIZE = 10          # items shown per page

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS_CSV = os.path.join(HERE, 'collection_items.csv')

INSTRUCTION = ("In this short study you'll see a series of statements and "
               "questions. For each one, please indicate how much it describes "
               "you, using the scale shown beneath it.")

# Shown at the top of every question page as a reminder.
PAGE_REMINDER = ("For each statement or question below, indicate how much it "
                 "describes you using the scale shown beneath it.")

WELCOME_MD = f"""
### Thank you for taking part

{INSTRUCTION}

- There are no right or wrong answers — we're interested in your honest response.
- It takes about **15–20 minutes**.
- Please **read each item carefully**: the study includes a few attention
  checks, and submissions that miss them may not be approved.
- Your responses are **anonymous**: we do not collect your name, email, or any
  identifying information.

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

# Minimal demographics (all optional; first option is a non-answer).
DEMOGRAPHICS = [
    ('age', 'Age', ['Prefer not to say', '18–24', '25–34', '35–44',
                    '45–54', '55–64', '65 or older']),
    ('gender', 'Gender', ['Prefer not to say', 'Woman', 'Man', 'Non-binary',
                          'Prefer to self-describe']),
    ('english_first_language', 'Is English your first language?',
     ['Prefer not to say', 'Yes', 'No']),
]

# Instructed-response attention checks use a neutral 5-point agree scale.
ATTN_SCALE = [
    {'value': '1', 'label': 'Strongly disagree'},
    {'value': '2', 'label': 'Disagree'},
    {'value': '3', 'label': 'Neither agree nor disagree'},
    {'value': '4', 'label': 'Agree'},
    {'value': '5', 'label': 'Strongly agree'},
]


def _secret(section: str, key: str, default: str) -> str:
    try:
        return str(st.secrets[section][key])
    except Exception:
        return default


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
    # Capture Prolific URL params once (empty for the pilot / direct links).
    if 'prolific' not in ss:
        qp = st.query_params
        ss['prolific'] = {
            'prolific_pid': qp.get('PROLIFIC_PID', ''),
            'study_id': qp.get('STUDY_ID', ''),
            'prolific_session_id': qp.get('SESSION_ID', ''),
        }


def _split_pool(pool: pd.DataFrame):
    is_mdes = pool['extra'].astype(str) == 'MDES'
    return pool[~is_mdes], pool[is_mdes]


def _attention_items(rng: random.Random, n: int) -> list[dict]:
    checks = []
    for i in range(n):
        target = rng.choice(ATTN_SCALE)
        checks.append({
            'kind': 'attention', 'item_id': f'attention_check_{i + 1}',
            'item_text_clean': '', 'dataset': 'attention_check',
            'extra': 'attention', 'n_options': len(ATTN_SCALE),
            'options': ATTN_SCALE, 'correct_label': target['label'],
            'question': ('This is an attention check — please select '
                         f'“{target["label"]}” for this item.'),
        })
    return checks


def _sample_items(pool: pd.DataFrame, session_id: str) -> list[dict]:
    """N_ITEMS from the main pool + MDES_PER_SESSION guaranteed MDES +
    N_ATTENTION instructed-response checks, all shuffled together."""
    rng = random.Random(session_id)
    main, mdes = _split_pool(pool)
    chosen = main.iloc[rng.sample(range(len(main)), min(N_ITEMS, len(main)))] \
        .to_dict('records')
    n_mdes = min(MDES_PER_SESSION, len(mdes))
    if n_mdes:
        chosen += mdes.iloc[rng.sample(range(len(mdes)), n_mdes)].to_dict('records')
    for it in chosen:
        it['kind'] = 'item'
        it['correct_label'] = ''
    chosen += _attention_items(rng, N_ATTENTION)
    rng.shuffle(chosen)
    return chosen


def _answered_count(items: list[dict]) -> int:
    return sum(1 for i in range(len(items))
              if st.session_state.get(f'resp_{i}') is not None)


def _scroll_top_if_new_page():
    """Reset scroll to the top when the page changes (but not on every rerun,
    so selecting an answer doesn't jump the view)."""
    page = st.session_state.get('page', 0)
    if st.session_state.get('_scrolled_page') != page:
        components.html(
            f"<script>/* page {page} */\n"
            "const d = window.parent.document;\n"
            "const c = d.querySelector('section.main') || "
            "d.querySelector('[data-testid=\"stAppViewContainer\"]');\n"
            "if (c) c.scrollTo({top: 0, behavior: 'instant'});\n"
            "window.parent.scrollTo(0, 0);</script>", height=0)
        st.session_state['_scrolled_page'] = page


# --- render steps -----------------------------------------------------------

def render_welcome(pool: pd.DataFrame):
    st.title('Off-Manifold: psychometric survey building')
    st.markdown(WELCOME_MD)
    agree = st.checkbox(CONSENT_LABEL, key='consent')
    if st.button('Begin', type='primary', disabled=not agree):
        st.session_state['stage'] = 'demographics'
        st.rerun()


def render_demographics(pool: pd.DataFrame):
    st.title('About you')
    st.caption('A few optional questions before we start. '
               'Choose "Prefer not to say" for anything you\'d rather skip.')
    for key, label, options in DEMOGRAPHICS:
        st.selectbox(label, options, index=0, key=f'demo_{key}')
    if st.button('Continue', type='primary'):
        # Snapshot now, while the widgets exist — Streamlit drops their
        # session_state once we leave this page.
        st.session_state['demographics'] = {
            key: st.session_state.get(f'demo_{key}', '')
            for key, _label, _opts in DEMOGRAPHICS}
        st.session_state['items'] = _sample_items(
            pool, st.session_state['session_id'])
        st.session_state['stage'] = 'survey'
        st.session_state['page'] = 0
        st.session_state.pop('_scrolled_page', None)
        st.rerun()


def _render_item(i: int, item: dict):
    labels = [o['label'] for o in item['options']]
    st.markdown(f"**{item['question']}**")
    st.radio('response', labels, index=None, key=f'resp_{i}',
             horizontal=False, label_visibility='collapsed')
    st.divider()


def render_survey():
    _scroll_top_if_new_page()
    items = st.session_state['items']
    n = len(items)
    page = st.session_state['page']
    n_pages = (n + PAGE_SIZE - 1) // PAGE_SIZE
    start, end = page * PAGE_SIZE, min((page + 1) * PAGE_SIZE, n)

    st.info(PAGE_REMINDER)
    answered = _answered_count(items)
    st.progress(answered / n, text=f'{answered} / {n} answered')
    st.caption(f'Page {page + 1} of {n_pages}')

    for i in range(start, end):
        _render_item(i, items[i])

    cols = st.columns(2)
    if page > 0 and cols[0].button('← Back'):
        st.session_state['page'] -= 1
        st.rerun()

    last = page == n_pages - 1
    if cols[1].button('Submit' if last else 'Next →', type='primary'):
        if last:
            _submit(items)
        else:
            st.session_state['page'] += 1
            st.rerun()


def _submit(items: list[dict]):
    ss = st.session_state
    ts = utc_now()
    prolific = ss.get('prolific', {})
    consent = bool(ss.get('consent'))

    def base_row(**kw) -> dict:
        row = {'session_id': ss['session_id'], 'timestamp_utc': ts,
               'consent': consent, **prolific, 'app_version': APP_VERSION}
        row.update(kw)
        return row

    rows = []
    # demographics (snapshotted at the Continue step)
    demo = ss.get('demographics', {})
    for key, label, _opts in DEMOGRAPHICS:
        val = demo.get(key, '')
        rows.append(base_row(kind='demographic', item_id=key, question=label,
                             response_label=val, response_value=val))
    # items + attention checks
    for i, item in enumerate(items):
        chosen = ss.get(f'resp_{i}')
        value = next((o['value'] for o in item['options']
                      if o['label'] == chosen), '')
        check_passed = ''
        if item.get('kind') == 'attention':
            check_passed = str(chosen == item.get('correct_label'))
        rows.append(base_row(
            kind=item.get('kind', 'item'), position=i,
            item_id=item['item_id'], item_text_clean=item.get('item_text_clean', ''),
            dataset=item['dataset'], extra=item.get('extra', ''),
            question=item['question'], response_value=value,
            response_label=chosen, n_options=item['n_options'],
            check_passed=check_passed))

    with st.spinner('Saving your responses…'):
        backend, detail = save_responses(rows)
    ss['save_backend'] = backend
    ss['save_detail'] = detail
    ss['stage'] = 'done'
    st.rerun()


def render_done():
    st.title('All done — thank you! 🎉')
    code = _secret('prolific', 'completion_code', 'PILOT-COMPLETE')
    st.markdown('Your responses have been recorded.')
    st.success(f'Your completion code is:  **{code}**')
    st.markdown('Please copy this code back into Prolific to confirm your '
                'participation. You may then close this tab.')
    if st.session_state.get('save_backend') == 'local':
        st.info('Note: responses were saved to a local file (the cloud Sheet '
                'is not configured). Detail: '
                f'`{st.session_state.get("save_detail")}`')
    st.caption(f'Session: {st.session_state["session_id"][:8]}')


def main():
    st.set_page_config(page_title='Off-Manifold survey', page_icon='📝')
    _init_state()
    # Preserve answered radios across page navigation: Streamlit drops the
    # session_state entry for any widget not rendered in the current run.
    for k in list(st.session_state.keys()):
        if k.startswith('resp_'):
            st.session_state[k] = st.session_state[k]
    pool = load_pool()
    stage = st.session_state['stage']
    if stage == 'welcome':
        render_welcome(pool)
    elif stage == 'demographics':
        render_demographics(pool)
    elif stage == 'survey':
        render_survey()
    else:
        render_done()


if __name__ == '__main__':
    main()
