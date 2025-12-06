#!/bin/bash

# mkdir -p ./log
# docker-compose down -v
# docker-compose up --build -d
sleep 20
echo "Running Seed Data..."
docker exec -it football_app python3 -m app.seed_data > ./log/match_gather.log

echo "Running Seed Players..."
docker exec -it football_app python3 -m app.seed_player > ./log/player_gather.log

echo "Running Seed Context..."
docker exec -it football_app python3 -m app.seed_context > ./log/context_gather.log

echo "Training Model..."
docker exec -it football_app python3 -m app.run_ml > ./log/ml.log

echo "Update Complete."