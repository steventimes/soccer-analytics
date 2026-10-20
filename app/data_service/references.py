"""Catalog audited public reference artifacts without executing their code."""

import argparse
import csv
import json
from pathlib import Path
from .warehouse import Warehouse
from .build_warehouse import document, table_name


def ingest(database="data/football.sqlite3", root=".tmp/open-data/research-references"):
    root = Path(root)
    reviews = []
    with Warehouse(database) as db:
        dataset = db.dataset("research-references", "reference", "public_github")
        for source in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = str(source.relative_to(root))
            aid = db.asset(dataset, source, relative)
            if source.suffix == ".csv":
                table = "reference_" + table_name(relative.removesuffix(".csv"))
                db.csv_table(aid, source, table)
                mapping = db.connection.execute(
                    "SELECT COUNT(*) FROM csv_columns WHERE asset_id=?", (aid,)
                ).fetchone()[0]
                irregular = db.connection.execute(
                    "SELECT COUNT(*) FROM csv_irregular_rows WHERE asset_id=?", (aid,)
                ).fetchone()[0]
                total = db.connection.execute(
                    "SELECT rows FROM imports WHERE asset_id=? AND kind=?", (aid, table)
                ).fetchone()[0]
                reviews.append(
                    {
                        "asset_id": aid,
                        "path": relative,
                        "source_records": total,
                        "header_columns": mapping,
                        "irregular_records": irregular,
                        "accepted_records": total - irregular,
                        "status": "schema_mismatch" if irregular else "reference_only",
                    }
                )
            elif source.suffix == ".json":
                document(db, aid, source, "reference_metadata")
        reviews.extend(
            [
                {
                    "source": "https://fminside.net/players/7-fm-26/28108494-phil-foden",
                    "status": "direct_access_403",
                    "scale": "Publisher states normalized 0-99, distinct from native FM attributes 1-20",
                    "note": "Search-readable page is a scope/scale reference, not a collected complete database.",
                },
                {
                    "source": "https://github.com/E2J1/STAT-499----Senior-project",
                    "method": "Repeated simulation runs and hidden-attribute ablation",
                    "scope": "FM24 simulated career outcomes",
                    "adaptation": "Use grouped holdouts and ablations; simulation results do not validate real-world career prediction.",
                },
                {
                    "source": "https://github.com/ElectroHugin/FM-Player-Analyzer",
                    "method": "Transparent role-weighted attribute scoring",
                    "scope": "FM24 game role fit",
                    "adaptation": "Treat weights as hypotheses; test against dated independent real outcomes before calling them scouting validity.",
                },
                {
                    "source": "https://github.com/LiamHardman/py-fm-dash",
                    "method": "User-configurable physical/mental/technical attribute weights",
                    "scope": "Game export analysis",
                    "adaptation": "Maintain attribute scales and report weights; do not reinterpret weighted game scores as measured athletic capacity.",
                },
            ]
        )
        payload = json.dumps(reviews, indent=2)
        out = root / "review.json"
        out.write_text(payload)
        aid = db.asset(dataset, out, "review.json")
        document(db, aid, out, "reference_review")
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return reviews


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--sources", default=".tmp/open-data/research-references")
    args = parser.parse_args()
    print(json.dumps(ingest(args.database, args.sources), indent=2))
