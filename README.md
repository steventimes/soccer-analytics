# Soccer Analytics

Soccer Analytics is a data + ML pipeline that collects football match data, stores it in Postgres/Redis, engineers rolling team features, trains prediction models per competition, exports a static daily dashboard, and publishes model operations plus release-governance artifacts for safer recommendation workflows.

## Project layout

```text
app/
  data_service/          # DB + cache access
  ml/                    # Feature engineering + model training/prediction
  run/                   # CLI script wrappers
  web/                   # Static site export, operations, release governance

docs/                    # Static dashboard and exported JSON payloads
models/                  # Trained model artifacts
```

## Requirements

- Docker + Docker Compose for the normal app flow.
- Python dependencies from `requirements.txt` for local runs.
- A `.env` file with the following values:

```text
FOOTBALL_DATA_API_KEY
DATABASE_URL
REDIS_URL

# Redis setup
REDIS_HOST
REDIS_PORT
REDIS_PASSWORD
REDIS_DB
```

## Quick start

Run the full Docker-based flow:

```bash
bash run.sh
```

Artifacts:
- Logs are written to `./log`.
- Models are saved to `./models`.
- Static dashboard data is written to `docs/data` by the export step.

## Pipeline commands

Once containers are up, run individual steps via the CLI:

```bash
docker exec -it football_app python3 -m app.pipeline train
docker exec -it football_app python3 -m app.pipeline predict --days 3
docker exec -it football_app python3 -m app.pipeline simulate
docker exec -it football_app python3 -m app.pipeline all --days 3
docker exec -it football_app python3 -m app.pipeline export-site --days 3
```

Local Python uses the same module commands when dependencies, Postgres, Redis, and `.env` are available.

## Daily dashboard

`python3 -m app.pipeline export-site --days 3` writes dashboard payloads under `docs/data`:

- `predictions.json`: upcoming model predictions.
- `preset_questions.json`: cached daily recommended-picks answers.
- `scores.json`: latest score payload.
- `operations.json`: model run health and data quality summary.
- `release.json`: recommendation publishing gate and risk disclosure.
- `manifest.json`: generated-at timestamp, prediction count, and artifact paths.

The static dashboard in `docs/index.html` reads these files directly, so it can be served through GitHub Pages or any static host.

## Operations summary

`docs/data/operations.json` includes run status, prediction count, competition coverage, average confidence, high-confidence pick count, and operational alerts for empty exports or weak confidence coverage. Operators should inspect this before using the latest picks.

## Release governance

`docs/data/release.json` uses the `SoccerAnalytics.ReleaseGovernance.v1` schema. It decides whether recommendations may be published, lists blockers, records thresholds, includes responsible-use risk disclosure, and publishes verification commands.

Public or paid picks should stay blocked until `can_publish_recommendations` is `true` and the release blockers list is empty. Model picks are probabilistic decision support, not guaranteed outcomes or financial advice.

## GitHub Actions config secrets

For CI/CD and scheduled jobs, add repository secrets in **GitHub -> Settings -> Secrets and variables -> Actions**:

- `FOOTBALL_DATA_API_KEY`
- `DATABASE_URL`
- `REDIS_URL`

Or set them with `gh`:

```bash
gh secret set FOOTBALL_DATA_API_KEY
gh secret set DATABASE_URL
gh secret set REDIS_URL
```

## Deployment options

### Docker Compose

```bash
docker compose up --build -d
```

Then run seeding and training from the app container:

```bash
docker exec -it football_app python3 -m app.seeds.seed_matches
docker exec -it football_app python3 -m app.seeds.seed_players
docker exec -it football_app python3 -m app.seeds.seed_competitions
docker exec -it football_app python3 -m app.pipeline train
```

### Static dashboard

After the export step, serve `docs/` from GitHub Pages or another static host. Commit `docs/data/*.json` only when you intentionally want cached demo data in the repository.

## Verification

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile app/web/release_governance.py app/web/export_site.py
node --check docs/app.js
```

Full pipeline export requires the Python ML dependencies, including pandas:

```bash
python3 -m app.pipeline export-site --days 3
```

## Improving model accuracy

The current model uses rolling team stats and ELO features. To improve accuracy, consider adding:

- Recent form at different windows, such as 3/5/10 game rolling windows.
- Home/away splits for xG, xGA, and points.
- Opponent-adjusted strength of schedule.
- Market features such as odds, implied probabilities, and line movement.
- Player availability, minutes, lineup strength, injuries, and suspensions.
- Match congestion, travel distance, and competition overlap.
- Calibration and class weights to handle draw imbalance.

Also consider time-series-aware cross-validation, separating domestic leagues from cups when distributions differ, and post-training probability calibration.
