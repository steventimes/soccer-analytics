"""Public ratings downloads; no credentials or provider code execution."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests


def collect(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = {
        "fifa23_all_updates.zip": "https://www.kaggle.com/api/v1/datasets/download/stefanoleone992/fifa-23-complete-player-dataset",
        "fm19.zip": "https://www.kaggle.com/api/v1/datasets/download/mathurinache/football-manager-2019-players-list",
        "fm20.zip": "https://www.kaggle.com/api/v1/datasets/download/ktyptorio/football-manager-2020",
        "fm22.zip": "https://www.kaggle.com/api/v1/datasets/download/furkanuluta/football-manager-22-complete-player-dataset",
        "fc24.zip": "https://www.kaggle.com/api/v1/datasets/download/stefanoleone992/ea-sports-fc-24-complete-player-dataset",
        "fc25.zip": "https://www.kaggle.com/api/v1/datasets/download/nyagami/ea-sports-fc-25-database-ratings-and-stats",
        "fc26.zip": "https://www.kaggle.com/api/v1/datasets/download/justdhia/ea-sports-fc-26-player-ratings",
        "fm17.zip": "https://www.kaggle.com/api/v1/datasets/download/ajinkyablaze/football-manager-data",
        "fc25_players.csv": "https://raw.githubusercontent.com/Radzim/football_data/main/players.csv",
        "fc25_sofifa.csv": "https://raw.githubusercontent.com/Radzim/football_data/main/sofifa.csv",
        "fifa12_fc25_top_players.csv": "https://raw.githubusercontent.com/07fahim/Top-FIFA-Players-Analysis-2012-2025/main/data/scraped/fifa_players_2012_2025.csv",
        "fm24_derived_role_profiles.csv": "https://raw.githubusercontent.com/adityasharma624/Player-Role-Dashboard/main/data/players_with_role_clusters_k5_v1.csv",
        "fm24_role_centroids.csv": "https://raw.githubusercontent.com/adityasharma624/Player-Role-Dashboard/main/data/cluster_centroids_k5.csv",
        "fc25_teams.csv": "https://raw.githubusercontent.com/Radzim/football_data/main/teams.csv",
        "fc25_models.csv": "https://raw.githubusercontent.com/Radzim/football_data/main/models.csv",
    }
    manifest = []
    with requests.Session() as session:
        for name, url in sources.items():
            path = root / name
            if not path.exists():
                with session.get(url, stream=True, timeout=(20, 180)) as response:
                    response.raise_for_status()
                    temporary = path.with_suffix(path.suffix + ".part")
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                    temporary.replace(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            manifest.append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "file": name,
                    "url": url,
                    "sha256": digest.hexdigest(),
                    "bytes": path.stat().st_size,
                    "quality_note": (
                        "Limited top-player sample, not complete rosters"
                        if name == "fifa12_fc25_top_players.csv"
                        else (
                            "Derived standardized attributes and model clusters, not native 1-20 attributes"
                            if name.startswith("fm24_")
                            else None
                        )
                    ),
                }
            )
        page_dir = root / "ea-endpoint-snapshot"
        page_dir.mkdir(exist_ok=True)
        offset = 0
        total = None
        while total is None or offset < total:
            path = page_dir / f"{offset:06}.json"
            url = f"https://drop-api.ea.com/rating/ea-sports-fc?locale=en&limit=100&offset={offset}"
            if path.exists():
                payload = json.loads(path.read_text())
            else:
                response = session.get(url, timeout=(20, 120))
                response.raise_for_status()
                payload = response.json()
                if not payload.get("items"):
                    raise ValueError("Unexpected empty EA page")
                path.write_text(json.dumps(payload, ensure_ascii=False))
                time.sleep(0.2)
            total = int(payload["totalItems"])
            offset += len(payload["items"])
            print("EA ratings", offset, "/", total, flush=True)
        manifest.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "directory": page_dir.name,
                "url": "https://drop-api.ea.com/rating/ea-sports-fc",
                "players": offset,
                "edition": None,
                "effective_date": None,
                "quality_note": "Endpoint differs from current EA website; edition and freshness unverified. Do not infer edition from avatar URLs.",
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".tmp/open-data/game-ratings")
    collect(parser.parse_args().output)
