"""Metrica CSV metadata and normalized tracking imports."""

import json

from .archive import open_asset_file


def ingest_metrica_csv(db, aid, path):
    import csv
    import tempfile
    from .warehouse import quote

    kind = "metrica_tracking_samples"
    if db.done(aid, kind):
        return
    db.connection.execute(
        "CREATE TABLE IF NOT EXISTS csv_preamble(asset_id INTEGER REFERENCES assets(id),row_number INTEGER,payload TEXT,PRIMARY KEY(asset_id,row_number))"
    )
    # Replace only this importer's former generic interpretation of the preamble.
    previous = db.connection.execute(
        "SELECT kind FROM imports WHERE asset_id=? AND kind LIKE 'metrica_csv_%rawtrackingdata_%'",
        (aid,),
    ).fetchall()
    for row in previous:
        db.connection.execute(f"DELETE FROM {quote(row[0])} WHERE asset_id=?", (aid,))
        db.connection.execute(
            "DELETE FROM imports WHERE asset_id=? AND kind=?", (aid, row[0])
        )
    db.connection.execute("DELETE FROM csv_irregular_rows WHERE asset_id=?", (aid,))
    with open_asset_file(path, "r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        for index in [1, 2]:
            db.connection.execute(
                "INSERT OR REPLACE INTO csv_preamble VALUES(?,?,?)",
                (aid, index, json.dumps(next(reader))),
            )
        original = next(reader)
        headers = original[:3]
        for index in range(3, len(original), 2):
            label = original[index]
            if not label or index + 1 >= len(original):
                raise ValueError("Unexpected Metrica coordinate header")
            headers.extend([label + "_x", label + "_y"])
        with tempfile.SpooledTemporaryFile(
            max_size=1024 * 1024, mode="w+", encoding="utf-8", newline=""
        ) as normalized:
            writer = csv.writer(normalized)
            writer.writerow(headers)
            writer.writerows(reader)
            normalized.seek(0)
            db.csv_stream(aid, normalized, kind)
        for index, name in enumerate(original):
            db.connection.execute(
                "UPDATE csv_columns SET source_name=? WHERE asset_id=? AND ordinal=?",
                (name, aid, index),
            )
    db.connection.commit()


def import_metrica(db, root):
    from app.data_service.tracking_frames import metrica_csv_frames, metrica_epts_frames
    import math

    dataset = db.dataset("metrica", "football", "metrica")
    for match in [1, 2, 3]:
        if match < 3:
            home = next(
                root.rglob(f"Sample_Game_{match}_RawTrackingData_Home_Team.csv")
            )
            away = next(
                root.rglob(f"Sample_Game_{match}_RawTrackingData_Away_Team.csv")
            )
            frames = metrica_csv_frames(home, away)
            source = home
        else:
            metadata = next(root.rglob("*.xml"))
            source = next(root.rglob("*.txt"))
            frames = metrica_epts_frames(metadata, source)
        aid = db.asset(dataset, source, str(source.relative_to(root)))
        if match < 3:
            for csv_source in [home, away]:
                csv_aid = db.asset(
                    dataset, csv_source, str(csv_source.relative_to(root))
                )
                ingest_metrica_csv(db, csv_aid, csv_source)
        if db.done(aid, "normalized_tracking"):
            continue
        batch = []
        count = 0
        for count, frame in enumerate(frames, 1):
            for player in frame["players"]:
                for key in ["x", "y"]:
                    if isinstance(player.get(key), float) and not math.isfinite(
                        player[key]
                    ):
                        player[key] = None
            batch.append(
                (aid, count, json.dumps(frame, separators=(",", ":"), allow_nan=False))
            )
            if len(batch) >= 1000:
                db.connection.executemany(
                    "INSERT OR REPLACE INTO normalized_tracking VALUES(?,?,?)", batch
                )
                db.connection.commit()
                batch = []
        db.connection.executemany(
            "INSERT OR REPLACE INTO normalized_tracking VALUES(?,?,?)", batch
        )
        db.finish(aid, "normalized_tracking", count)
