#!/bin/bash

mkdir -p ./log
mkdir -p ./models

echo "Rebuilding Docker..."
docker-compose down -v
docker-compose up --build -d

echo "Waiting for Database to initialize..."
sleep 10

echo "Running Seed Matches..."
docker exec -it football_app python3 -m app.seeds.seed_matches > ./log/match_gather.log

echo "Running Seed Players..."
docker exec -it football_app python3 -m app.seeds.seed_players > ./log/player_gather.log

echo "Running Seed Competitions..."
docker exec -it football_app python3 -m app.seeds.seed_competitions > ./log/context_gather.log

echo "Training Models..."
docker exec -it football_app python3 -m app.run_training > ./log/ml.log

echo "Pipeline Complete."