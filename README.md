# Soccer Analytics

Soccer Analytics fetches football match data, builds rolling team and Elo features, trains a three-outcome XGBoost model for each competition, and exports the Touchline static dashboard. Model bundles contain the estimator, team history, feature schema and validation results.

## Project layout

```text
app/
  config.py             # Environment settings and competition selection
  pipeline.py           # train, predict, export-site and all commands
  data_service/fetch/   # football-data.org client and request rate limiting
  ml/                   # Features, training, evaluation and model bundles
  web/                  # Dashboard export and release checks
docs/                   # Static dashboard and exported JSON
models/                 # Local model bundles and previous versions
tests/                  # Python tests and JavaScript runtime checks
```

## Run with Docker

Use Docker Compose v2.24 or newer for the optional `.env` file syntax in `docker-compose.yml`. Set `FOOTBALL_DATA_API_KEY` in a local `.env` file, then run:

```bash
bash run.sh --competitions PL --days 3
```

The script builds the image, trains the selected models and exports the dashboard. `rebuild.sh` runs the same flow. Models persist in `models/`, dashboard data in `docs/data/`, and logs go to the terminal. Each command runs in a disposable container:

```bash
docker compose run --rm app train --competitions PL
docker compose run --rm app predict --competitions PL --days 3
docker compose run --rm app export-site --competitions PL --days 3
docker compose run --rm app all --competitions PL --days 3
```

## Run locally

Use Python 3.12 or newer and install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export FOOTBALL_DATA_API_KEY='your-api-key'
python -m app.pipeline all --competitions PL --days 3
```

Local Python reads exported environment variables; Docker Compose loads `.env`. The pipeline uses in-process rate limiting and local model files. `predict` writes a JSON result to standard output, including any competition errors; logs go to standard error.

Choose competitions available to your API subscription. The default selection is all codes in `app/config.py`. Optional settings:

| Variable | Default |
| --- | --- |
| `SOCCER_ANALYTICS_COMPETITIONS` | All configured competitions |
| `SOCCER_ANALYTICS_TRAINING_SEASONS` | Four European season start years, including the current season |
| `SOCCER_ANALYTICS_PREDICTION_DAYS` | `3` |
| `SOCCER_ANALYTICS_SITE_EXPORT_DAYS` | `1` |
| `SOCCER_ANALYTICS_REQUEST_TIMEOUT` | `15` seconds |
| `SOCCER_ANALYTICS_REQUEST_RETRIES` | `3` |
| `SOCCER_ANALYTICS_MODEL_DIR` | `models` |
| `SOCCER_ANALYTICS_SITE_DATA_DIR` | `docs/data` |

Use comma-separated values for competitions and season years. `--competitions` selects from enabled competitions; `--seasons` overrides training years. For calendar-year leagues and international tournaments, select the applicable seasons explicitly. Training uses only finished matches. `--days 1` covers the remainder of today in UTC; `--days 3` also includes the next two UTC dates. Predictions include future `SCHEDULED` and `TIMED` fixtures in that range.

## Dashboard and release checks

`export-site` writes six files under `docs/data/`:

| File | Contents |
| --- | --- |
| `predictions.json` | Probabilities, explanations, model identities and partial errors |
| `preset_questions.json` | Recommended picks permitted by the release gate |
| `scores.json` | Daily scores and source errors |
| `operations.json` | Model coverage, validation, backtests, calibration and freshness |
| `release.json` | Release decision, blockers, thresholds, expiry and risk disclosure |
| `manifest.json` | Snapshot timestamp, run ID, file sizes and SHA-256 hashes |

Each file is replaced atomically, with the manifest written last. The dashboard verifies timestamps and hashes before showing the snapshot. Serve the site over localhost or HTTPS so browser SHA-256 verification is available:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://localhost:8000`. Search teams, filter competitions and confidence, sort fixtures, or expand a prediction to inspect its model evidence.

`release.json` uses `SoccerAnalytics.ReleaseGovernance.v3`. Recommendations require an approved decision, `can_publish_recommendations: true`, no blockers, and an unexpired `expires_at`. Permission expires at the first fixture's kickoff or 24 hours after export, whichever comes first. Refresh the export to reassess remaining fixtures. The shortlist uses the operations high-confidence threshold, normally 60%.

Failed training leaves existing model files intact. Prediction can fall back to a previous bundle if the active file is missing or invalid; release checks block recommendations using a fallback, stale model, incomplete coverage or inadequate quality evidence. Model probabilities do not guarantee match outcomes.

## GitHub Actions

Set the repository Actions secret `FOOTBALL_DATA_API_KEY` and configure GitHub Pages to deploy through Actions. `daily_update.yml` runs at 07:00 UTC, trains models and commits exported site data. `pages.yml` deploys `docs/` after a successful daily refresh, a push to `main`, or a manual run. The completion trigger handles daily commits made with `GITHUB_TOKEN`, which do not trigger a second push workflow.

## Verification

Node.js is required for the dashboard runtime tests.

```bash
python3 -m unittest discover -s tests -v
node --check docs/app.js
bash -n run.sh rebuild.sh
```

Tests use controlled data and temporary model files. Live training and export additionally require API access. Training evaluates expanding time windows and a final chronological holdout against majority-class and Elo baselines before saving a model.
