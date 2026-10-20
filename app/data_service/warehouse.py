"""Self-contained SQLite archive with lossless source files and native SQL views."""

from __future__ import annotations
import csv
import hashlib
import io
import json
import sqlite3
import zlib
from pathlib import Path


def quote(value):
    return '"' + value.replace('"', '""') + '"'


class Warehouse:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
        PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
        PRAGMA cache_size=-131072;
        CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY,name TEXT UNIQUE,family TEXT,provider TEXT,edition TEXT,observed_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS assets(id INTEGER PRIMARY KEY,dataset_id INTEGER REFERENCES datasets(id),path TEXT,sha256 TEXT,bytes INTEGER,complete INTEGER DEFAULT 0,UNIQUE(dataset_id,path));
        CREATE TABLE IF NOT EXISTS raw_chunks(asset_id INTEGER REFERENCES assets(id),ordinal INTEGER,content BLOB,PRIMARY KEY(asset_id,ordinal));
        CREATE TABLE IF NOT EXISTS imports(asset_id INTEGER REFERENCES assets(id),kind TEXT,rows INTEGER,PRIMARY KEY(asset_id,kind));
        CREATE TABLE IF NOT EXISTS asset_encodings(asset_id INTEGER PRIMARY KEY REFERENCES assets(id),encoding TEXT);
        CREATE TABLE IF NOT EXISTS csv_columns(asset_id INTEGER REFERENCES assets(id),ordinal INTEGER,source_name TEXT,column_name TEXT,PRIMARY KEY(asset_id,ordinal));
        CREATE TABLE IF NOT EXISTS csv_irregular_rows(asset_id INTEGER REFERENCES assets(id),row_number INTEGER,payload TEXT,PRIMARY KEY(asset_id,row_number));
        CREATE TABLE IF NOT EXISTS documents(asset_id INTEGER PRIMARY KEY REFERENCES assets(id),kind TEXT,payload TEXT);
        CREATE TABLE IF NOT EXISTS normalized_tracking(asset_id INTEGER REFERENCES assets(id),row_number INTEGER,payload TEXT,PRIMARY KEY(asset_id,row_number));
        CREATE VIEW IF NOT EXISTS tracking_positions AS SELECT f.asset_id,f.row_number,json_extract(f.payload,'$.period') period,json_extract(f.payload,'$.time') seconds,json_extract(p.value,'$.id') player_id,json_extract(p.value,'$.team') team,json_extract(p.value,'$.x') x_m,json_extract(p.value,'$.y') y_m,json_extract(p.value,'$.detected') detected FROM normalized_tracking f,json_each(f.payload,'$.players') p;
        CREATE TABLE IF NOT EXISTS json_lines(asset_id INTEGER REFERENCES assets(id),row_number INTEGER,payload TEXT,PRIMARY KEY(asset_id,row_number));
        CREATE TABLE IF NOT EXISTS quarantine(asset_id INTEGER PRIMARY KEY REFERENCES assets(id),reason TEXT);
        CREATE TABLE IF NOT EXISTS game_profiles(asset_id INTEGER REFERENCES assets(id),row_number INTEGER,provider TEXT,player_id TEXT,name TEXT,birth_date TEXT,overall REAL,scale TEXT,PRIMARY KEY(asset_id,row_number));
        CREATE TABLE IF NOT EXISTS pose_frames(asset_id INTEGER REFERENCES assets(id),match_id TEXT,frame INTEGER,period INTEGER,timestamp TEXT,payload TEXT,PRIMARY KEY(asset_id,frame));
        CREATE VIEW IF NOT EXISTS pose_joints AS SELECT f.asset_id,f.match_id,f.frame,f.period,f.timestamp,json_extract(p.value,'$.player_id') player_id,j.key joint_name,json_extract(j.value,'$.xyz[0]') x,json_extract(j.value,'$.xyz[1]') y,json_extract(j.value,'$.xyz[2]') z,json_extract(j.value,'$.p90_mae_cm') p90_mae_cm FROM pose_frames f,json_each(f.payload,'$.player_data') p,json_each(p.value,'$.joints') j;
        CREATE VIEW IF NOT EXISTS statsbomb_events AS SELECT d.asset_id,substr(a.path,13,length(a.path)-17) match_id,json_extract(e.value,'$.id') event_id,json_extract(e.value,'$.type.name') event_type,json_extract(e.value,'$.player.id') player_id,json_extract(e.value,'$.team.id') team_id,json_extract(e.value,'$.period') period,json_extract(e.value,'$.timestamp') timestamp,e.value payload FROM documents d JOIN assets a ON a.id=d.asset_id,json_each(d.payload) e WHERE d.kind='statsbomb_events';
        CREATE VIEW IF NOT EXISTS statsbomb_matches AS SELECT d.asset_id,json_extract(m.value,'$.match_id') match_id,json_extract(m.value,'$.match_date') match_date,m.value payload FROM documents d,json_each(d.payload) m WHERE d.kind='statsbomb_matches';
        CREATE VIEW IF NOT EXISTS statsbomb_visible_players AS SELECT d.asset_id,json_extract(e.value,'$.event_uuid') event_id,p.value payload FROM documents d,json_each(d.payload) e,json_each(e.value,'$.freeze_frame') p WHERE d.kind='statsbomb_360';
        DROP VIEW IF EXISTS skillcorner_tracking;
        CREATE VIEW skillcorner_tracking AS SELECT l.asset_id,l.row_number,json_extract(l.payload,'$.frame') frame,json_extract(l.payload,'$.period') period,json_extract(l.payload,'$.timestamp') timestamp,json_extract(p.value,'$.player_id') player_id,json_extract(p.value,'$.x') x_m,json_extract(p.value,'$.y') y_m,json_extract(p.value,'$.is_detected') is_detected,p.value payload FROM json_lines l JOIN assets a ON a.id=l.asset_id,json_each(l.payload,'$.player_data') p WHERE a.path LIKE '%tracking_extrapolated.jsonl';
        """)
        columns = {
            r[1] for r in self.connection.execute("PRAGMA table_info(game_profiles)")
        }
        for column in ["edition", "effective_date", "gender"]:
            if column not in columns:
                self.connection.execute(
                    f"ALTER TABLE game_profiles ADD COLUMN {column} TEXT"
                )
        self.connection.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.commit()
        self.connection.close()

    def dataset(self, name, family, provider, edition=None):
        self.connection.execute(
            "INSERT OR IGNORE INTO datasets(name,family,provider,edition) VALUES(?,?,?,?)",
            (name, family, provider, edition),
        )
        return self.connection.execute(
            "SELECT id FROM datasets WHERE name=?", (name,)
        ).fetchone()[0]

    def asset(self, dataset_id, path, relative_path):
        with open(path, "rb") as stream:
            return self.asset_stream(dataset_id, stream, relative_path)

    def record_analysis_inputs(self, run_kind, run_id, asset_ids):
        """Record only explicit, complete sources for an already archived run."""
        if not all(isinstance(value, str) and value.strip() for value in (run_kind, run_id)):
            raise ValueError("Run kind and ID must be nonempty strings")
        if isinstance(asset_ids, (str, bytes)):
            raise ValueError("Input assets must be a collection of asset IDs")
        inputs = tuple(asset_ids)
        if not inputs or any(isinstance(aid, bool) or not isinstance(aid, int) or aid <= 0 for aid in inputs):
            raise ValueError("Input asset IDs must be positive integers")
        inputs = set(inputs)
        self.connection.execute("SAVEPOINT record_analysis_inputs")
        try:
            for aid in inputs:
                row = self.connection.execute("SELECT complete FROM assets WHERE id=?", (aid,)).fetchone()
                if row is None or row[0] != 1:
                    raise ValueError(f"Unknown or incomplete input asset: {aid}")
            tables = {r[0] for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "analysis_run_assets" not in tables or self.connection.execute(
                "SELECT 1 FROM analysis_run_assets WHERE run_kind=? AND run_id=? LIMIT 1", (run_kind, run_id)
            ).fetchone() is None:
                raise ValueError("Unknown archived analysis run")
            if "analysis_run_inputs" in tables:
                previous = {r[0] for r in self.connection.execute(
                    "SELECT asset_id FROM analysis_run_inputs WHERE run_kind=? AND run_id=?", (run_kind, run_id)
                )}
                if previous and previous != inputs:
                    raise ValueError("Analysis run already has different recorded inputs")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS analysis_run_inputs(run_kind TEXT,run_id TEXT,asset_id INTEGER REFERENCES assets(id),PRIMARY KEY(run_kind,run_id,asset_id))"
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO analysis_run_inputs VALUES(?,?,?)",
                [(run_kind, run_id, aid) for aid in sorted(inputs)],
            )
        except BaseException:
            self.connection.execute("ROLLBACK TO record_analysis_inputs")
            raise
        finally:
            self.connection.execute("RELEASE record_analysis_inputs")

    def asset_stream(self, dataset_id, stream, relative_path):
        row = self.connection.execute(
            "SELECT id,complete FROM assets WHERE dataset_id=? AND path=?",
            (dataset_id, relative_path),
        ).fetchone()
        if row and row["complete"]:
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            previous = self.connection.execute(
                "SELECT sha256,bytes FROM assets WHERE id=?", (row["id"],)
            ).fetchone()
            if previous["sha256"] == digest.hexdigest() and previous["bytes"] == size:
                return row["id"]
            stream.seek(0)
            return self.asset_stream(
                dataset_id, stream, relative_path + "@" + digest.hexdigest()
            )
        self.connection.execute(
            "INSERT OR IGNORE INTO assets(dataset_id,path) VALUES(?,?)",
            (dataset_id, relative_path),
        )
        aid = self.connection.execute(
            "SELECT id FROM assets WHERE dataset_id=? AND path=?",
            (dataset_id, relative_path),
        ).fetchone()[0]
        self.connection.execute("DELETE FROM raw_chunks WHERE asset_id=?", (aid,))
        digest = hashlib.sha256()
        size = 0
        ordinal = 0
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            self.connection.execute(
                "INSERT INTO raw_chunks VALUES(?,?,?)",
                (aid, ordinal, zlib.compress(chunk, 1)),
            )
            ordinal += 1
        self.connection.execute(
            "UPDATE assets SET sha256=?,bytes=?,complete=1 WHERE id=?",
            (digest.hexdigest(), size, aid),
        )
        self.connection.commit()
        return aid

    def read_raw(self, asset_id):
        return b"".join(
            zlib.decompress(r[0])
            for r in self.connection.execute(
                "SELECT content FROM raw_chunks WHERE asset_id=? ORDER BY ordinal",
                (asset_id,),
            )
        )

    def done(self, aid, kind):
        return (
            self.connection.execute(
                "SELECT rows FROM imports WHERE asset_id=? AND kind=?", (aid, kind)
            ).fetchone()
            is not None
        )

    def finish(self, aid, kind, count):
        self.connection.execute(
            "INSERT OR REPLACE INTO imports VALUES(?,?,?)", (aid, kind, count)
        )
        self.connection.commit()

    def csv_table(self, asset_id, path, table_name):
        if self.done(asset_id, table_name):
            return
        try:
            with open(path, encoding="utf-8-sig", newline="") as stream:
                self.connection.execute(
                    "INSERT OR REPLACE INTO asset_encodings VALUES(?,?)",
                    (asset_id, "utf-8-sig"),
                )
                return self.csv_stream(asset_id, stream, table_name)
        except UnicodeDecodeError:
            with open(path, encoding="latin-1", newline="") as stream:
                self.connection.execute(
                    "INSERT OR REPLACE INTO asset_encodings VALUES(?,?)",
                    (asset_id, "latin-1"),
                )
                return self.csv_stream(asset_id, stream, table_name)

    def csv_stream(self, aid, stream, table):
        if self.done(aid, table):
            return
        if getattr(stream, "encoding", None):
            self.connection.execute(
                "INSERT OR REPLACE INTO asset_encodings VALUES(?,?)",
                (aid, stream.encoding),
            )
        sample = stream.read(8192)
        stream.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.reader(stream, delimiter=delimiter)
        original = next(reader)
        headers = []
        used = {"asset_id", "row_number"}
        for index, header in enumerate(original):
            candidate = header or f"unnamed_{index+1}"
            while candidate.lower() in used:
                candidate += "_" + str(index + 1)
            used.add(candidate.lower())
            headers.append(candidate)
            self.connection.execute(
                "INSERT OR REPLACE INTO csv_columns VALUES(?,?,?,?)",
                (aid, index, header, candidate),
            )
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS {quote(table)} (asset_id INTEGER REFERENCES assets(id),row_number INTEGER,PRIMARY KEY(asset_id,row_number))"
        )
        existing = {
            r[1] for r in self.connection.execute(f"PRAGMA table_info({quote(table)})")
        }
        for header in headers:
            if header.lower() not in {c.lower() for c in existing}:
                self.connection.execute(
                    f"ALTER TABLE {quote(table)} ADD COLUMN {quote(header)} TEXT"
                )
        sql = f'INSERT OR REPLACE INTO {quote(table)} ({",".join(map(quote,["asset_id","row_number"]+headers))}) VALUES ({",".join("?" for _ in range(len(headers)+2))})'
        batch = []
        count = 0
        for row in reader:
            if not row:
                continue
            count += 1
            if len(row) != len(headers):
                self.connection.execute(
                    "INSERT OR REPLACE INTO csv_irregular_rows VALUES(?,?,?)",
                    (aid, count, json.dumps(row)),
                )
                continue
            batch.append((aid, count, *[v if v != "" else None for v in row]))
            if len(batch) >= 20000:
                self.connection.executemany(sql, batch)
                self.connection.commit()
                batch = []
        self.connection.executemany(sql, batch)
        self.finish(aid, table, count)

    def pose_frame(self, aid, match_id, frame, period, timestamp, payload):
        self.connection.execute(
            "INSERT OR REPLACE INTO pose_frames VALUES(?,?,?,?,?,?)",
            (
                aid,
                match_id,
                frame,
                period,
                timestamp,
                json.dumps(payload, separators=(",", ":")),
            ),
        )

    def profile(
        self,
        aid,
        row_number,
        provider,
        player_id,
        *,
        name=None,
        birth_date=None,
        overall=None,
        scale=None,
        edition=None,
        effective_date=None,
        gender=None,
    ):
        self.connection.execute(
            "INSERT OR REPLACE INTO game_profiles(asset_id,row_number,provider,player_id,name,birth_date,overall,scale,edition,effective_date,gender) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                aid,
                row_number,
                provider,
                player_id,
                name,
                birth_date,
                overall,
                scale,
                edition,
                effective_date,
                gender,
            ),
        )
