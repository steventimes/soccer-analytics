#!/bin/bash

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Please run this script with bash: bash run.sh" >&2
  exit 1
fi

set -euo pipefail

mkdir -p ./log
mkdir -p ./models

echo "Rebuilding Docker..."
if command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE="docker-compose"
else
  DOCKER_COMPOSE="docker compose"
fi

${DOCKER_COMPOSE} down -v
${DOCKER_COMPOSE} up --build -d

echo "Waiting for Database to initialize..."
sleep 10

echo "Running Seed Matches..."
docker exec -it football_app python3 -m app.seeds.seed_matches > ./log/match_gather.log

echo "Running Seed Players..."
docker exec -it football_app python3 -m app.seeds.seed_players > ./log/player_gather.log

echo "Running Seed Competitions..."
docker exec -it football_app python3 -m app.seeds.seed_competitions > ./log/context_gather.log

echo "Training Models..."

docker exec -it football_app python3 -m app.pipeline train > ./log/ml.log

echo "Pipeline Complete."