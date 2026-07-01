# Deploying the data-collection survey

This `collection_app/` folder is self-contained. It needs only its own files
(`app.py`, `storage.py`, `collection_items.csv`, `requirements.txt`,
`.streamlit/`) — no ML libraries, no codebook. Deploy it as its own repo on
**Streamlit Community Cloud**, with responses written to a **Google Sheet**.

## 0. Run it locally first
```bash
streamlit run collection_app/app.py
```
With no secrets configured, responses fall back to `collection_app/local_responses.csv`
so you can test the full flow before wiring up the cloud.

## 1. Google Sheet + service account (storage)
1. Create an empty **Google Sheet** (any name). Copy its URL.
2. In **Google Cloud Console**: create/select a project → enable the
   **Google Sheets API** → create a **Service Account** → add a **JSON key** and
   download it.
3. **Share the Sheet** with the service account's `client_email` (from the JSON),
   giving **Editor** access.
4. Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and fill in
   the JSON fields under `[gcp_service_account]` and your Sheet URL under
   `[sheet]`. Re-run locally; the "All done" page should no longer show the
   local-file note, and rows should appear in the Sheet (header auto-created).

## 2. Put `collection_app/` in its own GitHub repo
Streamlit Cloud deploys from a repo. Keep this slim — push only this folder's
contents to a new repo (don't push the 5 GB parent project):
```bash
cd collection_app
git init && git add . && git commit -m "survey app"
# create an empty GitHub repo, then:
git remote add origin git@github.com:<you>/<survey-repo>.git
git push -u origin main
```
`.gitignore` already excludes `secrets.toml` and `local_responses.csv`.

## 3. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io → **New app** → pick the repo/branch.
2. Set **Main file path** to `app.py`.
3. Open **Advanced settings → Secrets** and paste the *contents* of your
   `secrets.toml` (the `[gcp_service_account]` and `[sheet]` blocks).
4. **Deploy.** You'll get a public `https://<name>.streamlit.app` URL to share.

## Updating the item pool
If the codebook or CNP set changes, regenerate the pool from the parent project
and re-commit:
```bash
python collection_app/build_items.py      # rewrites collection_items.csv
```

## Before collecting real data
Edit `WELCOME_MD` and `CONSENT_LABEL` in `app.py` to add your study title,
institution, ethics-approval reference, and a contact email. Tune `N_ITEMS`
(items per participant) and `PAGE_SIZE` at the top of `app.py` if needed.
