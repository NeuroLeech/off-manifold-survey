"""storage.py — persist participant responses.

Primary backend is a Google Sheet (one row per item response) written via a
service account; this works on Streamlit Community Cloud where the local disk is
ephemeral. If no credentials are configured (e.g. local development), it falls
back to appending to a local CSV so the app still runs end-to-end.

Configure the Google Sheet backend through Streamlit secrets — see
`.streamlit/secrets.toml.example`:

    [gcp_service_account]      # the service-account JSON, field by field
    ...
    [sheet]
    url = "https://docs.google.com/spreadsheets/d/<id>/edit"
    worksheet = "responses"

Share the target Sheet with the service account's `client_email` (Editor).
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import streamlit as st

# Column order written to the sheet / CSV. One row per (participant, item /
# attention-check / demographic).
COLUMNS = [
    'session_id', 'timestamp_utc', 'consent',
    'prolific_pid', 'study_id', 'prolific_session_id',
    'kind', 'position', 'item_id', 'item_text_clean', 'dataset', 'extra',
    'question', 'response_value', 'response_label', 'n_options',
    'check_passed', 'app_version',
]

LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'local_responses.csv')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@st.cache_resource(show_spinner=False)
def _worksheet():
    """Return (worksheet, None) on success or (None, reason) if unavailable.

    Cached so we authenticate once per server process. Header is created if the
    worksheet is empty.
    """
    # Accessing st.secrets raises if no secrets.toml exists anywhere; treat that
    # (and a missing section) as "not configured" and fall back to local CSV.
    try:
        configured = ('gcp_service_account' in st.secrets
                      and 'sheet' in st.secrets)
    except Exception:
        configured = False
    if not configured:
        return None, 'no-secrets'
    try:
        import gspread
        sa = dict(st.secrets['gcp_service_account'])
        gc = gspread.service_account_from_dict(sa)
        sheet_cfg = st.secrets['sheet']
        sh = (gc.open_by_url(sheet_cfg['url']) if 'url' in sheet_cfg
              else gc.open_by_key(sheet_cfg['id']))
        ws_name = sheet_cfg.get('worksheet', 'responses')

        def get_or_create(name):
            try:
                return sh.worksheet(name)
            except Exception:
                return sh.add_worksheet(title=name, rows=2000, cols=len(COLUMNS))

        ws = get_or_create(ws_name)
        head = ws.row_values(1)
        # If the existing tab has a different (older) schema, don't misalign into
        # it — route to a sibling "<name>_v2" tab so prior data stays intact.
        if head and head != COLUMNS:
            ws = get_or_create(f'{ws_name}_v2')
            head = ws.row_values(1)
        if not head:
            ws.append_row(COLUMNS, value_input_option='RAW')
        return ws, None
    except Exception as e:  # noqa: BLE001
        return None, f'sheets-error: {e}'


def _append_local(rows: list[dict]) -> None:
    # If an existing file has a different header (e.g. the schema changed during
    # development), don't silently mix schemas — append to a suffixed file.
    path = LOCAL_CSV
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            existing_header = next(csv.reader(f), [])
        if existing_header and existing_header != COLUMNS:
            i = 1
            while os.path.exists(f'{LOCAL_CSV[:-4]}_{i}.csv'):
                i += 1
            path = f'{LOCAL_CSV[:-4]}_{i}.csv'
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in COLUMNS})


def save_responses(rows: list[dict]) -> tuple[str, str]:
    """Persist response rows. Returns (backend, detail).

    backend is 'sheets' or 'local'. Falls back to a local CSV if the Sheet is
    not configured or the write fails, so data is never silently dropped.
    """
    ws, reason = _worksheet()
    if ws is not None:
        try:
            values = [[r.get(c, '') for c in COLUMNS] for r in rows]
            ws.append_rows(values, value_input_option='RAW')
            return 'sheets', 'ok'
        except Exception as e:  # noqa: BLE001
            reason = f'append-failed: {e}'
    _append_local(rows)
    return 'local', reason or 'local-fallback'
