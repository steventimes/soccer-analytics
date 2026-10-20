"""Public match-event and official-report projections."""

import json
import math
import re
from urllib.parse import urlparse


def public_fifa_offers(report):
    """Publish validated team totals, never local player rows or PDF text."""
    from app.analytics.fifa_offers import FIELDS, SCHEMA
    from app.data_service.fifa_reports import SERIES

    def integer(value, maximum=None):
        if type(value) is not int or value < 0 or maximum is not None and value > maximum:
            raise ValueError("Invalid FIFA offer count/page")
        return value

    if report.get("schema") != SCHEMA or not re.fullmatch(r"[a-f0-9]{20}", report.get("run_id", "")):
        raise ValueError("Unrecognized FIFA offers artifact")
    matches, seen = [], set()
    for match in report["matches"]:
        series, number = match["series"], integer(match["match_number"])
        if series not in SERIES or not 1 <= number <= SERIES[series]["matches"]:
            raise ValueError("Invalid FIFA report edition/number")
        expected_id = f"fifa-{series}-{number}"
        url = urlparse(match["source_url"])
        if (match["id"] != expected_id or expected_id in seen or url.scheme != "https"
                or url.netloc != "www.fifatrainingcentre.com" or not url.path.startswith("/media/native/")
                or not url.path.lower().endswith(".pdf") or url.query or url.fragment
                or not re.fullmatch(r"[a-f0-9]{64}", match["source_sha256"])):
            raise ValueError("Invalid FIFA offers identity/source")
        seen.add(expected_id)
        teams = []
        if len(match["teams"]) != 2:
            raise ValueError("Offers require two teams")
        for side, team in enumerate(match["teams"]):
            if team["team_slot"] != side or not isinstance(team["team"], str) or not team["team"].strip():
                raise ValueError("Invalid FIFA team identity")
            row = {"team_slot": side, "team": team["team"], **{key: integer(team[key]) for key in FIELDS}}
            for key in ("players", "player_table_page", "overview_page"):
                row[key] = integer(team[key], 100)
                if row[key] == 0:
                    raise ValueError("Missing FIFA roster/page")
            if row["total_offers"] != sum(row[key] for key in FIELDS[1:7]) or row["offers_received"] > row["total_offers"]:
                raise ValueError("Unreconciled FIFA offers counts")
            teams.append(row)
        if teams[0]["team"] == teams[1]["team"] or len(match["goals"]) != 2:
            raise ValueError("Duplicate FIFA teams or malformed score")
        matches.append({key: match[key] for key in ("id", "series", "match_number", "date", "stage", "source_url", "source_sha256")}
                       | {"goals": [integer(n) for n in match["goals"]], "teams": teams})
    if not matches:
        raise ValueError("No public FIFA offers")
    coverage = [{"series": series, "reports": sum(m["series"] == series for m in matches),
                 "expected_reports": SERIES[series]["matches"]} for series in sorted({m["series"] for m in matches})]
    return {"schema": SCHEMA, "run_id": report["run_id"], "matches": matches, "coverage": coverage,
            "evidence": {"reports": len(matches), "team_pages": 2 * len(matches),
                         "player_rows": sum(t["players"] for m in matches for t in m["teams"]),
                         "overview_totals_reconciled": 2 * len(matches)},
            "scope": report["scope"], "limits": report["limits"]}


def public_passing_motifs(summary, teams):
    from app.analytics.passing_motifs import motif_label, period_segment

    if summary.get('schema') != 'Touchline.PassingMotifs.v1' or summary.get('max_gap_seconds') != 5:
        raise ValueError('Invalid passing motif schema')
    public = {'schema': summary['schema'], 'max_gap_seconds': 5, 'players': {}, 'chains': [], 'audit': {}}
    totals = {team: dict.fromkeys(('chains', 'chain_passes', 'windows'), 0) for team in teams}
    seen = set()
    for chain in summary['chains']:
        row = {k: chain[k] for k in ('id', 'period', 'possession', 'team')}
        if (type(row['period']) is not int or row['period'] not in (1, 2, 3, 4) or row['team'] not in teams
                or type(row['possession']) is not int or row['possession'] <= 0 or len(chain['passes']) < 3):
            raise ValueError('Invalid passing motif chain')
        row['passes'] = []
        for source in chain['passes']:
            p = {k: source[k] for k in ('index', 'player', 'recipient', 'minute', 'second', 'clock_ms', 'segment', 'start', 'end')}
            if (any(type(p[k]) is not int or p[k] < 0 for k in ('index', 'minute', 'second', 'clock_ms', 'segment'))
                    or not p['index'] or p['index'] in seen or p['second'] >= 60
                    or p['minute'] != p['clock_ms'] // 60000 + {1: 0, 2: 45, 3: 90, 4: 105}[row['period']]
                    or p['second'] != p['clock_ms'] // 1000 % 60
                    or p['segment'] != period_segment(row['period'], p['minute'])):
                raise ValueError('Invalid passing motif clock/order')
            for key in ('player', 'recipient'):
                actor = p[key]
                if not isinstance(actor, str) or not re.fullmatch(r'[1-9][0-9]*', actor) or not isinstance(summary['players'].get(actor), str):
                    raise ValueError('Invalid passing motif actor')
                public['players'][actor] = summary['players'][actor]
            if p['player'] == p['recipient']:
                raise ValueError('Invalid passing motif self-pass')
            for key in ('start', 'end'):
                point = p[key]
                if point is not None and (not isinstance(point, list) or len(point) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                        or not 0 <= point[0] <= 120 or not 0 <= point[1] <= 80):
                    raise ValueError('Invalid passing motif location')
            if row['passes']:
                previous = row['passes'][-1]
                if p['index'] <= previous['index'] or previous['recipient'] != p['player'] or not 0 <= p['clock_ms'] - previous['clock_ms'] <= 5000:
                    raise ValueError('Invalid passing motif continuity')
            seen.add(p['index']); row['passes'].append(p)
        if row['id'] != f"{row['period']}-{row['passes'][0]['index']}":
            raise ValueError('Invalid passing motif identity')
        row['motifs'] = [motif_label([p['player'] for p in row['passes'][i:i + 3]] + [row['passes'][i + 2]['recipient']]) for i in range(len(row['passes']) - 2)]
        if row['motifs'] != chain['motifs']:
            raise ValueError('Invalid passing motif labels')
        totals[row['team']]['chains'] += 1; totals[row['team']]['chain_passes'] += len(row['passes']); totals[row['team']]['windows'] += len(row['motifs'])
        public['chains'].append(row)
    for team in teams:
        audit = {k: summary['audit'][team][k] for k in ('passes', 'completed_passes', 'eligible_passes', 'chain_passes', 'chains', 'windows')}
        if any(type(v) is not int or v < 0 for v in audit.values()) or not audit['passes'] >= audit['completed_passes'] >= audit['eligible_passes'] >= audit['chain_passes']:
            raise ValueError('Invalid passing motif audit')
        if any(audit[k] != v for k, v in totals[team].items()):
            raise ValueError('Invalid passing motif totals')
        public['audit'][team] = audit
    return public


def public_shot_sequences(sequences, teams):
    """Whitelist ordered event derivatives; never publish source frames or IDs."""
    if sequences.get("schema") != "Touchline.ShotSequences.v1":
        raise ValueError("Invalid shot sequence schema")
    public = {"schema": sequences["schema"], "phases": [],
              "unassigned_shots": {team: sequences["unassigned_shots"][team] for team in teams}}
    if any(type(n) is not int or n < 0 for n in public["unassigned_shots"].values()):
        raise ValueError("Invalid shot sequence coverage")
    seen = set()
    for phase in sequences["phases"]:
        row = {key: phase[key] for key in ("id", "period", "possession", "team", "shots", "goals", "xg", "opponent_events")}
        if (type(row["period"]) is not int or row["period"] not in (1, 2, 3, 4)
                or type(row["possession"]) is not int or row["possession"] <= 0
                or row["id"] != f"{row['period']}-{row['possession']}" or row["id"] in seen
                or row["team"] not in teams or not phase["actions"]
                or any(type(row[k]) is not int or row[k] < 0 for k in ("shots", "goals", "opponent_events"))):
            raise ValueError("Invalid shot sequence identity/counts")
        seen.add(row["id"])
        row["actions"] = []
        previous = 0
        for action in phase["actions"]:
            a = {key: action[key] for key in ("index", "minute", "second", "kind", "player", "recipient", "start", "end", "outcome", "xg")}
            if (type(a["index"]) is not int or a["index"] <= previous
                    or type(a["minute"]) is not int or a["minute"] < 0
                    or type(a["second"]) is not int or not 0 <= a["second"] < 60
                    or a["kind"] not in ("Pass", "Carry", "Dribble", "Shot")
                    or any(not isinstance(a[k], str) for k in ("player", "recipient", "outcome"))
                    or a["kind"] == "Shot" and (type(a["xg"]) not in (int, float) or not math.isfinite(a["xg"]) or not 0 <= a["xg"] <= 1)
                    or a["kind"] != "Shot" and a["xg"] is not None):
                raise ValueError("Invalid shot sequence action")
            for key in ("start", "end"):
                point = a[key]
                if point is not None and (not isinstance(point, list) or len(point) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                        or not 0 <= point[0] <= 120 or not 0 <= point[1] <= 80):
                    raise ValueError("Invalid shot sequence coordinate")
            previous = a["index"]
            row["actions"].append(a)
        shots = [a for a in row["actions"] if a["kind"] == "Shot"]
        if (not shots or row["actions"][-1]["kind"] != "Shot" or row["shots"] != len(shots)
                or row["goals"] != sum(a["outcome"] == "Goal" for a in shots)
                or type(row["xg"]) not in (int, float) or not math.isfinite(row["xg"])
                or not math.isclose(row["xg"], sum(a["xg"] for a in shots), abs_tol=1e-9)):
            raise ValueError("Invalid shot sequence reconciliation")
        public["phases"].append(row)
    return public


def public_exploration(source):
    """Project local exploration results onto public, derived-only schemas."""
    def read(relative):
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Public summary escapes exploration root")
        return json.loads(path.read_text())

    def fields(row, names):
        return {key: row[key] for key in names.split()}

    catalog = read("index.json")
    public_catalog = fields(catalog, "schema source revision source_url pitch scope")
    public_catalog["matches"] = [fields(m, "id date teams score competition") for m in catalog["matches"]]
    payloads = {"matches.json": public_catalog}
    for match in public_catalog["matches"]:
        identity = match["id"]
        if not re.fullmatch(r"statsbomb-[0-9]+", identity):
            raise ValueError("Invalid public match identity")
        data = read(f"matches/{identity}.json")
        if data["id"] != identity or data["teams"] != match["teams"]:
            raise ValueError("Public match summary does not match catalog")
        public = fields(data, "id date teams players")
        public["segments"] = []
        for segment in data["segments"]:
            row = fields(segment, "segment period team shots goals xg unlocated_shots passes completed_passes nodes edges")
            row["cells"] = [fields(c, "cell shots goals xg") for c in segment["cells"]]
            public["segments"].append(row)
        if "shot_sequences" in data:
            public["shot_sequences"] = public_shot_sequences(data["shot_sequences"], match["teams"])
        if "passing_motifs" in data:
            public["passing_motifs"] = public_passing_motifs(data['passing_motifs'], match['teams'])
            for team, audit in public['passing_motifs']['audit'].items():
                if any(audit[k] != sum(s[k] for s in public['segments'] if s['team'] == team) for k in ('passes', 'completed_passes')):
                    raise ValueError('Invalid passing motif source totals')
        payloads[f"matches/{identity}.json"] = public
    comparisons = read("comparisons.json")
    public = fields(comparisons, "schema team_source player_source")
    for kind, columns, metrics in (
        ("teams", "id name league season matches date_start date_end", "points_per_match goals_for_per_match goals_against_per_match win_rate clean_sheet_rate"),
        ("players", "id name league season position minutes matches date_end", "passes_per90 progressive_passes_per90 shots_per90 goals_per90 defensive_duels_per90 interceptions_per90 pass_completion duel_win_rate"),
    ):
        public[kind] = [fields(row, columns) | {"metrics": fields(row["metrics"], metrics)} for row in comparisons[kind]]
    payloads["comparisons.json"] = public
    validation = read("validation.json")
    public = fields(validation, "schema scope limits source")
    public["source"] = fields(validation["source"], "provider source_url model_family input_sha256")
    public["source"]["parameters"] = fields(validation["source"]["parameters"], "rounds max_depth learning_rate seed")
    public["source"]["code_sha256"] = {
        key: validation["source"]["code_sha256"][key]
        for key in ("app/analytics/exploration.py", "app/ml/feature_engineering.py", "app/ml/model.py")
    }
    public["leagues"] = []
    for league in validation["leagues"]:
        row = fields(league, "id name training_rows training_end test_start test_end model seed")
        row["periods"] = []
        for period in league["periods"]:
            p = fields(period, "month count correct baseline_correct log_loss_sum brier_sum baseline_brier_sum accuracy")
            p["calibration"] = {key: [fields(b, "count probability_sum hits") for b in period["calibration"][key]] for key in ("top", "home", "draw", "away")}
            row["periods"].append(p)
        public["leagues"].append(row)
    payloads["validation.json"] = public
    return payloads
