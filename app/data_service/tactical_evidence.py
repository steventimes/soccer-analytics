"""Import reviewed tactical reading/caption cards, retaining their actual evidence level."""

import argparse
import hashlib
import json
from pathlib import Path

from .warehouse import Warehouse
from .build_warehouse import document


def validate_card(source, card, text):
    if source["content_level"] in {"metadata_only", "blocked", "show_notes"} and card[
        "claim_type"
    ] not in {"topic_only", "availability"}:
        raise ValueError("Insufficient source content for a substantive tactical claim")
    anchor = card.get("anchor")
    if (
        anchor
        and " ".join(anchor.split()).lower() not in " ".join(text.split()).lower()
    ):
        raise ValueError("Claim anchor does not appear in inspected source content")
    if card["claim_type"] == "topic_only" and card.get("empirical_finding"):
        raise ValueError("Show notes are not empirical findings")


def ingest(database, root):
    root = Path(root).resolve()
    catalog = json.loads((root / "catalogue.json").read_text())
    with Warehouse(database) as db:
        ds = db.dataset(
            "tactical-media-evidence", "reference", "public_tactical_sources"
        )
        db.connection.executescript(
            """CREATE TABLE IF NOT EXISTS tactical_evidence_sources(source_id TEXT PRIMARY KEY,url TEXT,medium TEXT,publisher TEXT,published_at TEXT,content_level TEXT,asset_id INTEGER REFERENCES assets(id),payload TEXT);
            CREATE TABLE IF NOT EXISTS tactical_evidence_claims(claim_id TEXT PRIMARY KEY,source_id TEXT REFERENCES tactical_evidence_sources(source_id),claim_type TEXT,payload TEXT);"""
        )
        for source in catalog["sources"]:
            path = (root / source["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Evidence source file outside its collection root")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError("Evidence payload digest mismatch")
            aid = db.asset(ds, path, source["file"])
            text = (
                raw.decode("utf-8", errors="strict")
                if path.suffix in {".txt", ".vtt", ".json", ".html"}
                else ""
            )
            db.connection.execute(
                "INSERT OR REPLACE INTO tactical_evidence_sources VALUES(?,?,?,?,?,?,?,?)",
                (
                    source["id"],
                    source["url"],
                    source["medium"],
                    source["publisher"],
                    source.get("published_at"),
                    source["content_level"],
                    aid,
                    json.dumps(source),
                ),
            )
            for card in source["cards"]:
                validate_card(source, card, text)
                db.connection.execute(
                    "INSERT OR REPLACE INTO tactical_evidence_claims VALUES(?,?,?,?)",
                    (card["id"], source["id"], card["claim_type"], json.dumps(card)),
                )
        for item in catalog.get("supporting_assets", []):
            path = (root / item["file"]).resolve()
            if (
                not path.is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
            ):
                raise ValueError("Supporting asset path/hash mismatch")
            db.asset(ds, path, item["file"])
        aid = db.asset(ds, root / "catalogue.json", "catalogue.json")
        document(db, aid, root / "catalogue.json", "tactical_evidence_catalogue")
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {
        "sources": len(catalog["sources"]),
        "cards": sum(len(s["cards"]) for s in catalog["sources"]),
        "content_levels": {
            level: sum(s["content_level"] == level for s in catalog["sources"])
            for level in sorted({s["content_level"] for s in catalog["sources"]})
        },
        "scope": "Author statements, coach self-reports and testable project hypotheses remain separate from measured match findings. No external page or caption is executed.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/football.sqlite3")
    p.add_argument("--root", default=".tmp/tactical-evidence")
    a = p.parse_args()
    print(json.dumps(ingest(a.database, a.root), indent=2))
