"""Import collected sources into one resumable, self-contained SQLite database."""

import argparse
import io
from datetime import datetime, date, timedelta
import json
import re
import zipfile
from pathlib import Path
from .warehouse import Warehouse


def ea_birthdate(value):
    try:
        return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p").date().isoformat()
    except (ValueError, TypeError):
        return None


def table_name(value):
    return re.sub(r"[^a-zA-Z0-9_]", "_", value.lower())


def document(db, aid, path, kind):
    if db.done(aid, "document"):
        return
    try:
        payload = path.read_text(encoding="utf-8-sig")
        json.loads(payload)
        db.connection.execute(
            "INSERT OR REPLACE INTO documents VALUES(?,?,?)", (aid, kind, payload)
        )
        db.finish(aid, "document", 1)
    except (ValueError, UnicodeError) as exc:
        db.connection.execute(
            "INSERT OR REPLACE INTO quarantine VALUES(?,?)", (aid, str(exc))
        )
        db.finish(aid, "document", 0)


def lines(db, aid, stream, *, pose=False, match_id=None):
    kind = "pose" if pose else "json_lines"
    if db.done(aid, kind):
        return
    batch = []
    count = 0
    for line in stream:
        if not line.strip():
            continue
        payload = json.loads(line)
        count += 1
        if pose:
            batch.append(
                (
                    aid,
                    str(match_id),
                    payload.get("frame", count),
                    payload.get("period"),
                    payload.get("timestamp"),
                    json.dumps(payload, separators=(",", ":")),
                )
            )
            sql = "INSERT OR REPLACE INTO pose_frames VALUES(?,?,?,?,?,?)"
        else:
            batch.append((aid, count, json.dumps(payload, separators=(",", ":"))))
            sql = "INSERT OR REPLACE INTO json_lines VALUES(?,?,?)"
        if len(batch) >= 1000:
            db.connection.executemany(sql, batch)
            db.connection.commit()
            batch = []
    if batch:
        db.connection.executemany(sql, batch)
    db.finish(aid, kind, count)


def profiles_from_table(db, table, provider, id_column, name, dob, score, scale):
    from .warehouse import quote

    cols = {r[1] for r in db.connection.execute(f"PRAGMA table_info({quote(table)})")}
    aliases = {c.strip(): c for c in cols}
    if id_column not in aliases:
        return

    def field(c):
        return quote(aliases[c]) if c in aliases else "NULL"

    db.connection.execute(
        f"""INSERT INTO game_profiles(asset_id,row_number,provider,player_id,name,birth_date,overall,scale,edition,effective_date)
        SELECT asset_id,row_number,?,TRIM({field(id_column)}),TRIM({field(name)}),{field(dob)},{field(score)},?,{field('fifa_version')},{field('fifa_update_date') if 'fifa_update_date' in aliases else field('update_as_of')} FROM {quote(table)} WHERE 1
        ON CONFLICT(asset_id,row_number) DO UPDATE SET
        edition=COALESCE(excluded.edition,game_profiles.edition),
        effective_date=COALESCE(excluded.effective_date,game_profiles.effective_date)
        WHERE (excluded.edition IS NOT NULL AND game_profiles.edition IS NOT excluded.edition)
        OR (excluded.effective_date IS NOT NULL AND game_profiles.effective_date IS NOT excluded.effective_date)""",
        (provider, scale),
    )
    db.connection.commit()


def normalize_profile_birthdates(
    db, table, column, date_format, *, pattern=None, edition=None
):
    """Always normalize from immutable source rows, including on resumed builds."""
    from .warehouse import quote

    columns = {
        row[1].strip(): row[1]
        for row in db.connection.execute(f"PRAGMA table_info({quote(table)})")
    }
    if not columns:
        return
    birth_column = quote(columns[column]) if column in columns else "NULL"
    for row in db.connection.execute(
        f"SELECT asset_id,row_number,{birth_column} AS birth_date FROM {quote(table)}"
    ):
        value = row["birth_date"]
        if pattern:
            match = re.search(pattern, value or "")
            value = match.group(1) if match else None
        try:
            birth = datetime.strptime(value, date_format).date().isoformat()
        except (ValueError, TypeError):
            birth = None
        db.connection.execute(
            "UPDATE game_profiles SET birth_date=?,edition=COALESCE(?,edition) "
            "WHERE asset_id=? AND row_number=?",
            (birth, edition, row["asset_id"], row["row_number"]),
        )


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
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
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


def build(path, root, fm_root=None):
    root = Path(root).resolve()
    with Warehouse(path) as db:
        # Preserve source data, not SDK browser profiles or runtime state.
        for family_root in sorted(p for p in root.iterdir() if p.is_dir()):
            name = family_root.name
            if family_root.is_symlink():
                raise ValueError("Source family cannot be a directory alias")
            if name in {"isolated-browser-profile", "runtime"}:
                continue
            if name == "soccerdata":
                from .soccerdata_ingest import ingest_into

                ingest_into(db, family_root)
                continue
            dataset = db.dataset(
                name, "game" if name == "game-ratings" else "football", name
            )
            for source in sorted(
                p
                for p in family_root.rglob("*")
                if p.is_file() and not p.name.endswith(".part")
            ):
                relative = str(source.relative_to(family_root))
                resolved = source.resolve()
                if not resolved.is_relative_to(family_root.resolve()):
                    raise ValueError(
                        "Source file resolves outside its family directory"
                    )
                private = {"isolated-browser-profile", "runtime"}
                if private.intersection(
                    source.relative_to(family_root).parts
                ) or private.intersection(
                    resolved.relative_to(family_root.resolve()).parts
                ):
                    continue
                if source.stat().st_nlink != 1:
                    raise ValueError("Source assets cannot be hard-link aliases")
                aid = db.asset(dataset, source, relative)
                if source.suffix == ".csv":
                    if name == "metrica" and "RawTrackingData" in source.name:
                        ingest_metrica_csv(db, aid, source)
                        print("Imported", name, relative, flush=True)
                        continue
                    table = (
                        ("ratings_" + table_name(source.stem))
                        if name == "game-ratings"
                        else name.replace("-", "_") + "_csv_" + table_name(source.stem)
                    )
                    # Football seasonal CSVs share provider schema; SkillCorner dynamic events share theirs.
                    if name == "football-csv":
                        table = "football_results_raw"
                    if name == "skillcorner" and source.name.endswith(
                        "_dynamic_events.csv"
                    ):
                        table = "skillcorner_dynamic_events"
                    if name == "skillcorner" and source.name.endswith(
                        "_phases_of_play.csv"
                    ):
                        table = "skillcorner_phases"
                    db.csv_table(aid, source, table)
                elif source.suffix == ".json":
                    kind = name + "_document"
                    if name == "statsbomb":
                        if relative.startswith("data/events/"):
                            kind = "statsbomb_events"
                        elif relative.startswith("data/matches/"):
                            kind = "statsbomb_matches"
                        elif relative.startswith("data/three-sixty/"):
                            kind = "statsbomb_360"
                    document(db, aid, source, kind)
                elif source.suffix == ".jsonl":
                    with source.open("rb") as stream:
                        lines(db, aid, stream)
                elif source.suffix == ".zip" and name == "game-ratings":
                    with zipfile.ZipFile(source) as archive:
                        for member in archive.namelist():
                            if not member.endswith(".csv"):
                                continue
                            print("CSV", source.name, member, flush=True)
                            with archive.open(member) as stream:
                                member_id = db.asset_stream(
                                    dataset, stream, source.name + "/" + member
                                )
                            with archive.open(member) as stream:
                                table = (
                                    "fm17_players"
                                    if source.name == "fm17.zip"
                                    else (
                                        "fifa_"
                                        if source.name == "fifa23_all_updates.zip"
                                        else source.stem + "_"
                                    )
                                    + table_name(Path(member).stem)
                                )
                                try:
                                    db.csv_stream(
                                        member_id,
                                        io.TextIOWrapper(
                                            stream, encoding="utf-8-sig", newline=""
                                        ),
                                        table,
                                    )
                                except UnicodeDecodeError:
                                    with archive.open(member) as fallback:
                                        db.csv_stream(
                                            member_id,
                                            io.TextIOWrapper(
                                                fallback, encoding="latin-1", newline=""
                                            ),
                                            table,
                                        )
                elif source.suffix == ".zip" and "external-bodypose" in relative:
                    with zipfile.ZipFile(source) as archive:
                        for member in archive.namelist():
                            if (
                                not member.endswith(".jsonl")
                                or "__MACOSX" in member
                                or Path(member).name.startswith("._")
                            ):
                                continue
                            with archive.open(member) as stream:
                                lines(
                                    db,
                                    aid,
                                    stream,
                                    pose=True,
                                    match_id=source.name.split(".")[0],
                                )
                print("Imported", name, relative, flush=True)
        if fm_root:
            fm_root = Path(fm_root)
            for directory in sorted(fm_root.glob("csv_*")):
                report = json.loads((directory / "conversion_report.json").read_text())
                dataset = db.dataset(
                    directory.name, "game", "fm", edition=report.get("version")
                )
                db.connection.execute(
                    "UPDATE datasets SET edition=? WHERE id=?",
                    (report.get("version"), dataset),
                )
                for source in sorted(p for p in directory.iterdir() if p.is_file()):
                    aid = db.asset(dataset, source, source.name)
                    print("FM", directory.name, source.name, flush=True)
                    if source.suffix == ".csv":
                        db.csv_table(aid, source, "fm_" + table_name(source.stem))
                    elif source.suffix == ".json":
                        document(db, aid, source, "fm_provenance")
            audits = fm_root.parent / "audits" / "fm_database_check_2026-10-06"
            if audits.exists():
                dataset = db.dataset("fm_source_audit", "quality", "fm")
                for source in sorted(
                    p for p in audits.iterdir() if p.suffix in {".json", ".csv"}
                ):
                    aid = db.asset(dataset, source, source.name)
                    if source.suffix == ".json":
                        document(db, aid, source, "fm_quality")
                    else:
                        db.csv_table(aid, source, "fm_audit_" + table_name(source.stem))
            archives = fm_root / "raw_archives"
            if archives.exists():
                dataset = db.dataset("fm_original_archives", "game", "fm")
                for source in sorted(p for p in archives.iterdir() if p.is_file()):
                    db.asset(dataset, source, source.name)
        for extra in [root.parent / "history", root.parent / "scouting"]:
            if not extra.exists():
                continue
            dataset = db.dataset(extra.name, "derived", "local")
            for source in sorted(
                p
                for p in extra.rglob("*")
                if p.is_file() and p.suffix in {".json", ".csv", ".png", ".svg"}
            ):
                aid = db.asset(dataset, source, str(source.relative_to(extra)))
                if source.suffix == ".json":
                    document(db, aid, source, "local_" + extra.name)
                elif source.suffix == ".csv":
                    db.csv_table(aid, source, "derived_" + table_name(source.stem))
        dataset = db.dataset("analysis_reports", "derived", "local")
        for name in [
            "bodypose-collection.json",
            "history-analysis.json",
            "history-collection.json",
            "history-comparison.json",
            "history-comparison.partial.json",
            "open-data-inventory.json",
            "pl-analysis.json",
            "pl-experiments.json",
            "source-audit.json",
        ]:
            source = root.parent / name
            if source.exists():
                aid = db.asset(dataset, source, name)
                document(
                    db,
                    aid,
                    source,
                    (
                        "local_analysis_partial"
                        if ".partial." in name
                        else "local_analysis"
                    ),
                )
        if (root / "metrica").exists():
            import_metrica(db, root / "metrica")
        # Preserve EA response provenance; the snapshot has no verified edition date.
        ea_birthdates = {}
        for row in db.connection.execute(
            "SELECT d.asset_id,d.payload FROM documents d JOIN assets a ON a.id=d.asset_id WHERE a.path LIKE 'ea-endpoint-snapshot/%' ORDER BY d.asset_id"
        ).fetchall():
            for index, player in enumerate(json.loads(row["payload"])["items"], 1):
                ea_birthdates[str(player["id"])] = ea_birthdate(player.get("birthdate"))
                db.profile(
                    row["asset_id"],
                    index,
                    "ea_endpoint",
                    str(player["id"]),
                    name=player.get("commonName")
                    or " ".join(
                        filter(None, [player.get("firstName"), player.get("lastName")])
                    ),
                    birth_date=ea_birthdate(player.get("birthdate")),
                    overall=player.get("overallRating"),
                    scale="0-99; edition unverified",
                    gender={0: "male", 1: "female"}.get(
                        (player.get("gender") or {}).get("id")
                    ),
                )
        profiles_from_table(
            db,
            "fm_players",
            "fm",
            "person_id",
            "full_name",
            "birth_date",
            "current_ability",
            "0-200; negative potential is a range code",
        )
        profiles_from_table(
            db,
            "fm17_players",
            "fm17",
            "UID",
            "Name",
            "Born",
            None,
            "attributes 1-20; CA/PA unavailable",
        )
        normalize_profile_birthdates(db, "fm17_players", "Born", "%d-%m-%Y")
        for table in ["fifa_male_players", "fifa_female_players"]:
            profiles_from_table(
                db, table, "fifa", "player_id", "long_name", "dob", "overall", "0-99"
            )
        if db.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='ratings_fifa12_fc25_top_players'"
        ).fetchone():
            for row in db.connection.execute(
                "SELECT * FROM ratings_fifa12_fc25_top_players"
            ).fetchall():
                identity = re.search(r"/player/(\d+)/", row["Profile URL"] or "")
                if identity:
                    db.profile(
                        row["asset_id"],
                        row["row_number"],
                        "sofifa_top_sample",
                        identity.group(1),
                        name=row["Name"],
                        overall=(
                            float(row["Overall rating"])
                            if row["Overall rating"]
                            else None
                        ),
                        scale="0-99; limited top-player sample",
                        edition=row["FIFA Version Name"],
                    )
        for table in ["fc24_male_players", "fc24_female_players"]:
            profiles_from_table(
                db, table, "fifa", "player_id", "long_name", "dob", "overall", "0-99"
            )
        # FM UI exports have padded headers. UID exports keep game IDs; exports
        # without a UID receive explicitly local row identifiers.
        for table, edition in [
            ("fm22_fm20data", "2020"),
            ("fm22_fm21data", "2021"),
            ("fm22_fm22data", "2022"),
            ("fm22_fm2023", "2023"),
        ]:
            profiles_from_table(
                db,
                table,
                "fm_export",
                "UID",
                "Name",
                "DoB",
                None,
                "native attributes 1-20; CA/PA unavailable",
            )
            normalize_profile_birthdates(
                db,
                table,
                "DoB",
                "%d/%m/%Y",
                pattern=r"(\d{1,2}/\d{1,2}/\d{4})",
                edition=edition,
            )
        for table, provider, edition, name, score, scale in [
            ("fm20_datafm20", "fm20_export", "2020", "Name", "CA", "0-200"),
            (
                "fm19_players_list",
                "fm19_export",
                "2019",
                "Name",
                None,
                "CA/PA unavailable; role percentage retained in source",
            ),
        ]:
            if not db.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name=?", (table,)
            ).fetchone():
                continue
            for row in db.connection.execute("SELECT * FROM " + table).fetchall():
                db.profile(
                    row["asset_id"],
                    row["row_number"],
                    provider,
                    "local-row:" + str(row["asset_id"]) + ":" + str(row["row_number"]),
                    name=row[name],
                    overall=float(row[score]) if score and row[score] else None,
                    scale=scale,
                    edition=edition,
                )
        for table, provider, edition in [
            ("fc25_all_players", "fc25_ea_export", "FC25"),
            ("fc26_ea_fc26_players", "fc26_ea_export", "FC26"),
        ]:
            if not db.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name=?", (table,)
            ).fetchone():
                continue
            for row in db.connection.execute("SELECT * FROM " + table).fetchall():
                if edition == "FC25":
                    match = re.search(r"/(\d+)/?$", row["url"] or "")
                    if not match:
                        continue
                    identity = match.group(1)
                    name = row["Name"]
                    rating = row["OVR"]
                    birth = None
                else:
                    identity = row["id"]
                    name = row["commonName"] or " ".join(
                        filter(None, [row["firstName"], row["lastName"]])
                    )
                    rating = row["overallRating"]
                    birth = ea_birthdate(row["birthdate"])
                db.profile(
                    row["asset_id"],
                    row["row_number"],
                    provider,
                    identity,
                    name=name,
                    birth_date=birth,
                    overall=float(rating) if rating else None,
                    scale="0-99; publisher-labelled edition",
                    edition=edition,
                )
        profiles_from_table(
            db,
            "ratings_fc25_sofifa",
            "fc25",
            "playerid",
            "full_name",
            None,
            "overall_rating",
            "0-99; raw dob retained as source serial",
        )
        if db.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='ratings_fc25_sofifa'"
        ).fetchone():
            for row in db.connection.execute(
                "SELECT asset_id,row_number,playerid,full_name,dob FROM ratings_fc25_sofifa"
            ).fetchall():
                birth = None
                try:
                    decoded = (
                        date(1899, 12, 30) + timedelta(days=int(row["dob"]))
                    ).isoformat()
                    if decoded == ea_birthdates.get(row["playerid"]):
                        birth = decoded
                except (ValueError, TypeError, OverflowError):
                    pass
                db.connection.execute(
                    "UPDATE game_profiles SET name=?,birth_date=? WHERE asset_id=? AND row_number=?",
                    (
                        (row["full_name"] or "").strip() or None,
                        birth,
                        row["asset_id"],
                        row["row_number"],
                    ),
                )
        db.connection.execute(
            "UPDATE game_profiles SET edition='2017' WHERE provider='fm17'"
        )
        db.connection.execute(
            "UPDATE game_profiles SET edition='FC25' WHERE provider='fc25'"
        )
        db.connection.execute(
            "UPDATE game_profiles SET edition=(SELECT d.edition FROM datasets d JOIN assets a ON a.dataset_id=d.id WHERE a.id=game_profiles.asset_id) WHERE provider='fm'"
        )
        for row in db.connection.execute(
            "SELECT a.dataset_id,json_extract(d.payload,'$.database_date') effective_date FROM documents d JOIN assets a ON a.id=d.asset_id WHERE d.kind='fm_provenance' AND a.path='conversion_report.json'"
        ).fetchall():
            db.connection.execute(
                "UPDATE game_profiles SET effective_date=? WHERE asset_id IN (SELECT id FROM assets WHERE dataset_id=?) AND provider='fm'",
                (row["effective_date"], row["dataset_id"]),
            )
        db.connection.execute(
            "UPDATE game_profiles SET gender=CASE WHEN asset_id IN (SELECT id FROM assets WHERE path LIKE '%/female_players.csv') THEN 'female' WHEN asset_id IN (SELECT id FROM assets WHERE path LIKE '%/male_players.csv') THEN 'male' ELSE gender END WHERE provider='fifa'"
        )
        db.connection.executescript("""
        CREATE VIEW IF NOT EXISTS fm_player_ratings AS SELECT p.asset_id,d.edition,p.person_id,p.full_name,p.birth_date,p.club_id,p.natural_positions,CAST(p.current_ability AS INTEGER) current_ability,CAST(p.potential_ability AS INTEGER) potential_ability,CASE WHEN p.potential_ability IS NULL THEN NULL WHEN CAST(p.potential_ability AS INTEGER)<0 THEN 'range_code' ELSE 'fixed' END potential_kind FROM fm_players p JOIN assets a ON a.id=p.asset_id JOIN datasets d ON d.id=a.dataset_id;
        CREATE VIEW IF NOT EXISTS game_edition_coverage AS SELECT provider,edition,effective_date,COUNT(*) records,COUNT(DISTINCT player_id) players FROM game_profiles GROUP BY provider,edition,effective_date;
        CREATE VIEW IF NOT EXISTS game_profile_versions AS SELECT p.*,d.name dataset,d.edition dataset_edition,a.path source_file FROM game_profiles p JOIN assets a ON a.id=p.asset_id JOIN datasets d ON d.id=a.dataset_id;
        CREATE VIEW IF NOT EXISTS conservative_identity_candidates AS
        WITH identities AS (SELECT provider,player_id,MIN(name) name,MIN(birth_date) birth_date FROM game_profiles WHERE name IS NOT NULL AND birth_date GLOB '????-??-??' GROUP BY provider,player_id HAVING COUNT(DISTINCT name)=1 AND COUNT(DISTINCT birth_date)=1),
        unique_names AS (SELECT provider,name,birth_date,MIN(player_id) player_id FROM identities GROUP BY provider,name,birth_date HAVING COUNT(*)=1)
        SELECT a.provider left_provider,a.player_id left_id,b.provider right_provider,b.player_id right_id,a.name,a.birth_date FROM unique_names a JOIN unique_names b ON a.name=b.name AND a.birth_date=b.birth_date AND a.provider<b.provider;
        CREATE VIEW IF NOT EXISTS fifa_edition_coverage AS SELECT fifa_version,fifa_update_date,COUNT(*) profiles,COUNT(DISTINCT player_id) players FROM fifa_male_players GROUP BY fifa_version,fifa_update_date;
        CREATE VIEW IF NOT EXISTS football_matches AS SELECT d.asset_id,json_extract(m.value,'$.id') match_id,m.value payload FROM documents d JOIN assets a ON a.id=d.asset_id,json_each(d.payload,'$.matches') m WHERE a.path LIKE 'history/football-csv/%';
        CREATE INDEX IF NOT EXISTS profiles_identity ON game_profiles(provider,player_id);
        CREATE INDEX IF NOT EXISTS profiles_name_birth ON game_profiles(name,birth_date);
        CREATE INDEX IF NOT EXISTS pose_match_frame ON pose_frames(match_id,frame);
        CREATE INDEX IF NOT EXISTS documents_kind ON documents(kind);
        """)
        db.connection.commit()
        status = {
            "datasets": [
                dict(r) for r in db.connection.execute("SELECT * FROM datasets")
            ],
            "assets": db.connection.execute("SELECT COUNT(*) FROM assets").fetchone()[
                0
            ],
            "source_bytes": db.connection.execute(
                "SELECT SUM(bytes) FROM assets"
            ).fetchone()[0],
            "imports": [
                dict(r)
                for r in db.connection.execute(
                    "SELECT kind,SUM(rows) rows,COUNT(*) assets FROM imports GROUP BY kind"
                )
            ],
            "quarantine": [
                dict(r)
                for r in db.connection.execute(
                    "SELECT a.path,q.reason FROM quarantine q JOIN assets a ON a.id=q.asset_id"
                )
            ],
            "irregular_csv_rows": db.connection.execute(
                "SELECT COUNT(*) FROM csv_irregular_rows"
            ).fetchone()[0],
            "foreign_key_errors": [
                tuple(r) for r in db.connection.execute("PRAGMA foreign_key_check")
            ],
        }
        Path(path).with_suffix(".inventory.json").write_text(
            json.dumps(status, indent=2)
        )
        print(json.dumps(status, indent=2), flush=True)
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--sources", default=".tmp/open-data")
    parser.add_argument("--fm-root")
    args = parser.parse_args()
    build(args.database, args.sources, args.fm_root)
