"""Local, checksummed public-season snapshots and conservative provider comparisons."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from app.data_service.fetch.public import LEAGUES, create_data_client
from app.data_service.match_records import normalize_match
from app.file_io import write_json_atomic as _atomic_write_json
from app.data_service.freshness import SnapshotCachePolicy, utc_now

logger = logging.getLogger(__name__)
PUBLIC_SOURCES = ("football-csv", "openfootball")


def digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


@lru_cache(maxsize=4096)
def canonical_team(name: str) -> str:
    tokens = re.findall(
        r"[a-z0-9]+",
        unicodedata.normalize("NFKD", name.casefold())
        .encode("ascii", "ignore")
        .decode(),
    )
    key = "".join(
        token for token in tokens if token not in {"fc", "afc", "cf", "sc", "ac"}
    )
    aliases = {
        "manunited": "manchesterunited",
        "mancity": "manchestercity",
        "tottenham": "tottenhamhotspur",
        "wolves": "wolverhamptonwanderers",
        "westbrom": "westbromwichalbion",
        "westham": "westhamunited",
        "newcastle": "newcastleunited",
        "leicester": "leicestercity",
        "stoke": "stokecity",
        "hull": "hullcity",
        "cardiff": "cardiffcity",
        "swansea": "swanseacity",
        "leeds": "leedsunited",
        "sheffieldutd": "sheffieldunited",
        "nottmforest": "nottinghamforest",
        "qpr": "queensparkrangers",
        "norwich": "norwichcity",
        "ipswich": "ipswichtown",
        "luton": "lutontown",
        "huddersfield": "huddersfieldtown",
        "bayernmunich": "bayernmunchen",
        "bayernmuenchen": "bayernmunchen",
        "dortmund": "borussiadortmund",
        "mgladbach": "borussiamonchengladbach",
        "borussiamonchengladbach": "borussiamonchengladbach",
        "1koln": "koln",
        "cologne": "koln",
        "hertha": "herthabsc",
        "herthabscberlin": "herthabsc",
        "einfrankfurt": "eintrachtfrankfurt",
        "mainz05": "mainz",
        "1mainz05": "mainz",
        "leipzig": "rbleipzig",
        "athmadrid": "atleticomadrid",
        "atleticodemadrid": "atleticomadrid",
        "athbilbao": "athleticbilbao",
        "athleticclub": "athleticbilbao",
        "sociedad": "realsociedad",
        "betis": "realbetis",
        "parissg": "parissaintgermain",
        "olympiquelyonnais": "lyon",
        "olympiquedemarseille": "marseille",
        "stetienne": "saintetienne",
        "assaintetienne": "saintetienne",
        "asmonaco": "monaco",
        "inter": "internazionale",
        "intermilan": "internazionale",
        "asroma": "roma",
        "sporting": "sportingcp",
        "sportinglisbon": "sportingcp",
        "sportingclubeportugal": "sportingcp",
        "psveindhoven": "psv",
        "ajaxamsterdam": "ajax",
        "azalkmaar": "az",
        "brightonhovealbion": "brighton",
        "coventrycity": "coventry",
        "blackburnrovers": "blackburn",
        "prestonnorthend": "preston",
        "birminghamcity": "birmingham",
        "sheffieldweds": "sheffieldwednesday",
        "derbycounty": "derby",
        "rotherhamunited": "rotherham",
        "plymouthargyle": "plymouth",
        "oxfordunited": "oxford",
        "wiganathletic": "wigan",
        "wycombewanderers": "wycombe",
        "peterboro": "peterboroughunited",
        "boltonwanderers": "bolton",
        "charltonathletic": "charlton",
        "vflwolfsburg": "wolfsburg",
        "1fsvmainz05": "mainz",
        "vfbstuttgart": "stuttgart",
        "1unionberlin": "unionberlin",
        "bayer04leverkusen": "leverkusen",
        "bayerleverkusen": "leverkusen",
        "tsg1899hoffenheim": "hoffenheim",
        "1899hoffenheim": "hoffenheim",
        "svwerderbremen": "werderbremen",
        "vflbochum1848": "bochum",
        "bormonchengladbach": "borussiamonchengladbach",
        "hamburgersv": "hamburg",
        "1heidenheim1846": "heidenheim",
        "stpauli1910": "stpauli",
        "svdarmstadt98": "darmstadt",
        "arminiabielefeld": "bielefeld",
        "hannover96": "hannover",
        "spvgggreutherfurth1903": "greutherfurth",
        "paderborn07": "paderborn",
        "1nurnberg": "nurnberg",
        "ingolstadt04": "ingolstadt",
        "forsittard": "fortunasittard",
        "peczwolle": "zwolle",
        "heraclesalmelo": "heracles",
        "rkcwaalwijk": "waalwijk",
        "sbvexcelsior": "excelsior",
        "nec": "nijmegen",
        "necnijmegen": "nijmegen",
        "adodenhaag": "denhaag",
        "feyenoordrotterdam": "feyenoord",
        "twente65": "twente",
        "sbvvitesse": "vitesse",
        "willemiitilburg": "willemii",
        "degraafschap": "graafschap",
        "telstar1963": "telstar",
        "ogcnice": "nice",
        "lilleosc": "lille",
        "montpellierhsc": "montpellier",
        "angerssco": "angers",
        "olympiquemarseille": "marseille",
        "stadebrestois29": "brest",
        "staderennais": "rennes",
        "staderennais1901": "rennes",
        "stadedereims": "reims",
        "rcstrasbourg": "strasbourg",
        "rcstrasbourgalsace": "strasbourg",
        "girondinsbordeaux": "bordeaux",
        "dijonfco": "dijon",
        "eaguingamp": "guingamp",
        "smcaen": "caen",
        "rclens": "lens",
        "racingclubdelens": "lens",
        "ajauxerre": "auxerre",
        "clermontfoot63": "clermont",
        "nimesolympique": "nimes",
        "estactroyes": "troyes",
        "estroyes": "troyes",
        "sv07elversberg": "elversberg",
        "frosinonecalcio": "frosinone",
        "realracingclubdesantander": "santander",
        "rcdeportivolacoruna": "deportivolacoruna",
        "asnancylorraine": "nancy",
        "caosasuna": "osasuna",
        "levanteud": "levante",
        "realbetisbalompie": "realbetis",
        "rcceltadevigo": "celta",
        "rccelta": "celta",
        "realsociedaddefutbol": "realsociedad",
        "rcdmallorca": "mallorca",
        "clubatleticodemadrid": "atleticomadrid",
        "deportivoalaves": "alaves",
        "cdalaves": "alaves",
        "sdeibar": "eibar",
        "cdleganes": "leganes",
        "rayovallecanodemadrid": "vallecano",
        "rayovallecano": "vallecano",
        "espanol": "espanyol",
        "espanyolbarcelona": "espanyol",
        "rcdespanyoldebarcelona": "espanyol",
        "udlaspalmas": "laspalmas",
        "realvalladolid": "valladolid",
        "lacoruna": "deportivolacoruna",
        "udalmeria": "almeria",
        "sdhuesca": "huesca",
        "spgijon": "sportinggijon",
        "realoviedo": "oviedo",
        "vitoriaguimaraes": "guimaraes",
        "cdsantaclara": "santaclara",
        "splisbon": "sportingcp",
        "sportingclubedeportugal": "sportingcp",
        "cdtondela": "tondela",
        "slbenfica": "benfica",
        "sportlisboaebenfica": "benfica",
        "csmaritimo": "maritimo",
        "spbraga": "sportingbraga",
        "sportingclubedebraga": "sportingbraga",
        "pacosdeferreira": "pacosferreira",
        "cdnacional": "nacional",
        "gdchaves": "chaves",
        "estreladaamadora": "estrela",
        "gdestorilpraia": "estoril",
        "gdestoril": "estoril",
        "vitoriasetubal": "setubal",
        "cdaves": "aves",
        "cdfeirense": "feirense",
        "acffiorentina": "fiorentina",
        "sscnapoli": "napoli",
        "udinesecalcio": "udinese",
        "genoacfc": "genoa",
        "cagliaricalcio": "cagliari",
        "bologna1909": "bologna",
        "sslazio": "lazio",
        "lazioroma": "lazio",
        "atalantabc": "atalanta",
        "internazionalemilano": "internazionale",
        "hellasverona": "verona",
        "parmacalcio1913": "parma",
        "uslecce": "lecce",
        "ussassuolocalcio": "sassuolo",
        "sassuolocalcio": "sassuolo",
        "speziacalcio": "spezia",
        "ucsampdoria": "sampdoria",
        "ussalernitana1919": "salernitana",
        "chievoverona": "chievo",
        "spal2013ferrara": "spal",
        "uscremonese": "cremonese",
        "como1907": "como",
        "beneventocalcio": "benevento",
    }
    # These explicit aliases cover names observed in the archived providers.
    # Ambiguous club changes (e.g. Belenenses / Belenenses SAD) remain separate.
    return aliases.get(key, key)


def match_key(match) -> tuple[str, str, str]:
    return (
        match.get("source_date", match["utcDate"][:10]),
        canonical_team(match["homeTeam"]["name"]),
        canonical_team(match["awayTeam"]["name"]),
    )


def compare_sources(left: list[dict], right: list[dict]) -> dict:
    def index(matches):
        result = {}
        for match in matches:
            key = match_key(match)
            score = match["score"]["fullTime"]
            if key in result and result[key] != score:
                raise ValueError(f"同源比赛存在冲突: {key}")
            result[key] = score
        return result

    first, second = index(left), index(right)
    common = first.keys() & second.keys()
    conflicts = [
        {
            "date": key[0],
            "home": key[1],
            "away": key[2],
            "left": first[key],
            "right": second[key],
        }
        for key in sorted(common)
        if first[key] != second[key]
    ]
    pairs = []
    for matches in (left, right):
        grouped = {}
        for match in matches:
            key = match_key(match)
            grouped.setdefault(key[1:], []).append(match)
        pairs.append(grouped)
    date_conflicts = []
    for pair in sorted(pairs[0].keys() & pairs[1].keys()):
        if len(pairs[0][pair]) != 1 or len(pairs[1][pair]) != 1:
            continue
        a, b = pairs[0][pair][0], pairs[1][pair][0]
        if match_key(a)[0] != match_key(b)[0]:
            date_conflicts.append(
                {
                    "home": pair[0],
                    "away": pair[1],
                    "left_date": match_key(a)[0],
                    "right_date": match_key(b)[0],
                    "left_match_id": a["id"],
                    "right_match_id": b["id"],
                }
            )
    return {
        "left_rows": len(first),
        "right_rows": len(second),
        "matched": len(common),
        "only_left": len(first.keys() - second.keys()),
        "only_right": len(second.keys() - first.keys()),
        "score_conflicts": len(conflicts),
        "score_agreement": (
            (len(common) - len(conflicts)) / len(common) if common else None
        ),
        "conflict_examples": conflicts[:20],
        "score_conflict_records": conflicts,
        "date_conflicts": len(date_conflicts),
        "date_conflict_examples": date_conflicts[:20],
        "date_conflict_records": date_conflicts,
        "unmatched_left_examples": [
            list(key) for key in sorted(first.keys() - second.keys())[:10]
        ],
        "unmatched_right_examples": [
            list(key) for key in sorted(second.keys() - first.keys())[:10]
        ],
        "matching": "Date and explicit normalized team names; unmatched records are not automatically merged.",
    }


def audit_history(collections: dict) -> dict:
    comparisons, excluded = [], []
    for code in sorted({key[1] for key in collections}):
        for season in sorted({key[2] for key in collections}):
            keys = [("football-csv", code, season), ("openfootball", code, season)]
            if any(key not in collections for key in keys):
                continue
            left, right = (collections[key] for key in keys)
            report = compare_sources(left, right)
            comparisons.append({"competition": code, "season": season, **report})
            bad_ids = [set(), set()]
            for conflict in report["date_conflict_records"]:
                bad_ids[0].add(conflict["left_match_id"])
                bad_ids[1].add(conflict["right_match_id"])
            for side, matches in enumerate((left, right)):
                for match in matches:
                    key = match_key(match)
                    if any(
                        (item["date"], item["home"], item["away"]) == key
                        for item in report["score_conflict_records"]
                    ):
                        bad_ids[side].add(match["id"])
                    if match["id"] in bad_ids[side]:
                        excluded.append(
                            {
                                "source": keys[side][0],
                                "competition": code,
                                "season": season,
                                "match_id": match["id"],
                                "date": key[0],
                                "home": key[1],
                                "away": key[2],
                                "reason": "cross-provider date or score conflict",
                            }
                        )
    return {
        "comparisons": comparisons,
        "excluded_records": excluded,
        "policy": "Keep raw snapshots intact; isolate conflicting records from both providers in experiments.",
    }


def read_snapshot(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("schema") != 1 or not expected or digest(body) != expected:
        raise ValueError(f"数据快照校验失败: {path}")
    if not isinstance(payload.get("matches"), list) or not payload["matches"]:
        raise ValueError(f"快照缺少比赛: {path}")
    for match in payload["matches"]:
        if normalize_match(match).status != "FINISHED":
            raise ValueError(f"历史快照包含未完赛比赛: {path}")
    return payload


def collect_history(
    cache_dir: Path,
    sources,
    competitions,
    seasons,
    *,
    refresh: bool = False,
    client_factory=create_data_client,
    max_age_hours=None,
    active_max_age_hours=None,
    clock=None,
) -> dict:
    policy = SnapshotCachePolicy(max_age_hours, active_max_age_hours)
    cache_dir = Path(cache_dir)
    sources, competitions, seasons = (
        list(dict.fromkeys(sources)),
        list(dict.fromkeys(competitions)),
        sorted(set(seasons)),
    )
    if not sources or any(source not in PUBLIC_SOURCES for source in sources):
        raise ValueError("只支持 football-csv 和 openfootball 历史快照")
    if not competitions or any(code not in LEAGUES for code in competitions):
        raise ValueError("联赛必须是公开源支持的代码")
    if not seasons or any(re.fullmatch(r"\d{4}", season) is None for season in seasons):
        raise ValueError("赛季必须是四位起始年")

    def collect_source(source):
        client = client_factory(source)
        collections, errors = [], []
        for code in competitions:
            for season in seasons:
                path = cache_dir / source / code / f"{season}.json"
                decision = None
                try:
                    snapshot = None
                    if path.exists() and not refresh:
                        snapshot = read_snapshot(path)
                        if (
                            snapshot["source"],
                            snapshot["competition"],
                            snapshot["season"],
                        ) != (source, code, season):
                            raise ValueError("快照范围与路径不一致")
                    decision = policy.decide(snapshot.get('fetched_at') if snapshot else None, season,
                                             now=utc_now(clock), exists=path.exists(), refresh=refresh)
                    if decision['action'] == 'fetch':
                        matches = client.fetch_matches(
                            code, season=season, status="FINISHED"
                        )
                        if not matches:
                            raise ValueError("数据源没有返回已完赛比赛")
                        unique = {}
                        for match in matches:
                            record = normalize_match(match)
                            if (
                                record.status != "FINISHED"
                                or match.get("source") != client.source_name
                            ):
                                raise ValueError("数据源或完赛状态不一致")
                            if (
                                record.match_id in unique
                                and normalize_match(unique[record.match_id]) != record
                            ):
                                raise ValueError("冲突比赛 ID")
                            unique[record.match_id] = match
                        snapshot = {
                            "schema": 1,
                            "source": source,
                            "provider": client.source_name,
                            "competition": code,
                            "season": season,
                            "fetched_at": utc_now(clock).isoformat(),
                            "matches": sorted(
                                unique.values(), key=lambda m: (m["utcDate"], m["id"])
                            ),
                        }
                        snapshot["sha256"] = digest(snapshot)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        _atomic_write_json(path, snapshot)
                    collections.append(
                        {
                            "source": source,
                            "competition": code,
                            "season": season,
                            "path": str(path.relative_to(cache_dir)),
                            "sha256": snapshot["sha256"],
                            "rows": len(snapshot["matches"]),
                            "fetched_at": snapshot["fetched_at"],
                            "cache": decision,
                            "source_urls": sorted(
                                {m.get("source_url", "") for m in snapshot["matches"]}
                            ),
                        }
                    )
                    logger.info(
                        "历史快照 %s %s %s: %d 场",
                        source,
                        code,
                        season,
                        len(snapshot["matches"]),
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "source": source,
                            "competition": code,
                            "season": season,
                            "code": type(exc).__name__,
                            "message": str(exc),
                            "cache": decision,
                        }
                    )
                    logger.warning(
                        "历史快照失败 %s %s %s: %s", source, code, season, exc
                    )
        return collections, errors

    with ThreadPoolExecutor(max_workers=min(2, len(sources))) as pool:
        results = list(pool.map(collect_source, sources))
    report = {
        "schema": 1,
        "generated_at": utc_now(clock).isoformat(),
        "cache_policy": policy.summary(),
        "collections": [item for collections, _ in results for item in collections],
        "errors": [item for _, errors in results for item in errors],
    }
    report["rows"] = sum(item["rows"] for item in report["collections"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(cache_dir / "manifest.json", report)
    return report


def load_history(cache_dir: Path) -> tuple[dict, dict]:
    root = Path(cache_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    collections = {}
    for entry in manifest["collections"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("快照路径越界")
        snapshot = read_snapshot(path)
        if snapshot["sha256"] != entry["sha256"]:
            raise ValueError("快照与 manifest 摘要不一致")
        key = (entry["source"], entry["competition"], entry["season"])
        if (
            key != (snapshot["source"], snapshot["competition"], snapshot["season"])
            or key in collections
        ):
            raise ValueError("快照身份与 manifest 不一致或重复")
        collections[key] = snapshot["matches"]
    return collections, manifest
