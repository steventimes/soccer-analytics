# Soccer Analytics

Soccer Analytics is a data + ML pipeline that collects football match data, stores it in Postgres/Redis, engineers rolling team features, and trains prediction models per competition. It also includes a weekend prediction workflow and a betting simulation runner.

## Project layout

```
app/
  data_service/          # DB + cache access
  ml/                    # Feature engineering + model training/prediction
  run/                   # CLI scripts
  web/                   # Scheduled jobs (daily update)
docs/                    # Static UI demo assets
```

## Requirements

- Docker + Docker Compose
- A `.env` file with the following values:

```
FOOTBALL_DATA_API_KEY
DATABASE_URL
REDIS_URL

# Redis setup
REDIS_HOST
REDIS_PORT
REDIS_PASSWORD
REDIS_DB
```

## Quick start (local)

Run the full Docker-based flow (build containers, seed data, train models):

```bash
bash run.sh
```

Artifacts:
- Logs are written to `./log`.
- Models are saved to `./models`.

## Running the pipeline manually

Once containers are up, you can run individual steps via the CLI:

```bash
docker exec -it football_app python3 -m app.pipeline train
docker exec -it football_app python3 -m app.pipeline predict --days 3
docker exec -it football_app python3 -m app.pipeline simulate
docker exec -it football_app python3 -m app.pipeline all --days 3
```

## Deployment options

### Docker Compose (recommended)

```bash
docker compose up --build -d
```

Then run seeding + training from the app container:

```bash
docker exec -it football_app python3 -m app.seeds.seed_matches
docker exec -it football_app python3 -m app.seeds.seed_players
docker exec -it football_app python3 -m app.seeds.seed_competitions
docker exec -it football_app python3 -m app.pipeline train
```

### Local Python (advanced)

If you want to run locally without Docker, install dependencies from `requirements.txt`,
ensure Postgres/Redis are running, set `.env`, and run the same `python3 -m app.pipeline ...` commands.

## Improving model accuracy (ideas)

The current model uses rolling team stats + ELO features. If you want to boost accuracy,
consider adding:

- **Recent form at different windows** (e.g., 3/5/10 game rolling windows).
- **Home/away splits** for xG, xGA, and points.
- **Opponent-adjusted strength of schedule** (opponent ELO or points in the rolling window).
- **Market features** (odds, implied probabilities, line movement).
- **Player availability** (injuries/suspensions, minutes, lineup strength).
- **Match congestion** (days since last match, travel distance, competition overlaps).
- **Calibration + class weights** to handle draw imbalance.

Also consider:
- Hyperparameter tuning with time-series-aware CV.
- Separating domestic leagues vs cups if distributions differ.
- Post-training calibration (Platt/Isotonic) for probability quality.
