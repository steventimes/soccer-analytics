"""Public aggregate experiment, literature, and market schemas."""

import json
import math
import re
from datetime import date
from urllib.parse import urlparse


def public_spatial_style(report):
    """Expose paired aggregate retrieval scores without identities or fitted bases."""
    leagues = ("England", "France", "Germany", "Italy", "Spain")

    if report.get("schema") != "SoccerAnalytics.SpatialStyle.v1" or not re.fullmatch(r"[a-f0-9]{20}", report.get("run_id", "")):
        raise ValueError("Invalid spatial style report")
    public = {key: report[key] for key in ("schema", "run_id", "paper_url", "scope", "method", "limits")}
    public["source"] = {key: report["source"][key] for key in ("provider", "source_url", "event_zip_sha256", "exposure_cache_sha256", "inventory_sha256", "warehouse_signature")}
    for key in ("event_zip_sha256", "exposure_cache_sha256", "inventory_sha256", "warehouse_signature"):
        if not re.fullmatch(r"[a-f0-9]{64}", public["source"][key]):
            raise ValueError("Invalid spatial style source digest")
    for url in (public["source"]["source_url"], public["paper_url"]):
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Invalid spatial style source URL")
    public["cohort"] = {key: report["cohort"][key] for key in ("players", "same_club_both_periods", "exposure_source")}
    public["cohort"]["by_league"] = {league: report["cohort"]["by_league"][league] for league in leagues}
    counts = [public["cohort"]["players"], *public["cohort"]["by_league"].values()]
    if any(type(n) is not int or n <= 0 for n in counts) or sum(counts[1:]) != counts[0] or public["cohort"]["same_club_both_periods"] is not True:
        raise ValueError("Invalid spatial style cohort")
    public["windows"] = {block: {key: report["windows"][block][key] for key in ("start", "end")} for block in ("early", "late")}
    w = public["windows"]
    for block in w.values():
        for value in block.values(): date.fromisoformat(value)
    if not "2017-07-01" <= w["early"]["start"] <= w["early"]["end"] < "2018-01-01" <= w["late"]["start"] <= w["late"]["end"] < "2018-07-01":
        raise ValueError("Invalid spatial style chronology")
    public["parameters"] = {key: report["parameters"][key] for key in ("grid", "sigma_cells", "minimum_minutes_each_period", "distance", "basis_scale", "nmf_max_iter")}
    public["parameters"]["seeds"] = list(report["parameters"]["seeds"])
    if "component_selection" in report["parameters"]:
        provenance = report["parameters"]["component_selection"]
        if (provenance.get("policy") != "fixed_reference_configuration"
                or provenance.get("reference") != report["paper_url"]
                or provenance.get("local_70_percent_variance_selection_performed") is not False
                or report["parameters"].get("smoothing_units") != "grid_cells"
                or report["parameters"].get("display_interpolation") != "nearest"):
            raise ValueError("Invalid spatial component provenance or display units")
        public["parameters"].update(component_selection={key: provenance[key] for key in (
                                         "policy", "reference", "local_70_percent_variance_selection_performed")},
                                     smoothing_units="grid_cells", display_interpolation="nearest")
    if (public["parameters"]["grid"] != 50 or public["parameters"]["sigma_cells"] != 1
            or public["parameters"]["minimum_minutes_each_period"] != 900
            or public["parameters"]["distance"] != "Manhattan" or public["parameters"]["basis_scale"] != "unit_L1"
            or public["parameters"]["nmf_max_iter"] != 1000):
        raise ValueError("Invalid spatial style benchmark parameters")
    seeds = public["parameters"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2 ** 32 for s in seeds):
        raise ValueError("Invalid spatial style seeds")
    if "nmf_init" in report["parameters"]:
        params = report["parameters"]
        if params["nmf_init"] not in ("random", "nndsvda") or params.get("nmf_solver") != "cd" or params.get("nmf_tol") != 1e-4:
            raise ValueError("Invalid spatial style initialization")
        public["parameters"].update({key: params[key] for key in ("nmf_init", "nmf_solver", "nmf_tol")})
    public["summary"], seen = [], set()
    features = {"frequency": 4, "heatmaps": 12500, "nmf_start_only": 18, "nmf_start_end": 18}
    for summary in report["summary"]:
        row = {key: summary[key] for key in ("scope", "method", "pool", "features", "runs", "queries", "candidate_min", "candidate_max", "random_top1")}
        identity = (row["scope"], row["method"], row["pool"])
        expected_queries = public["cohort"]["players"] if row["scope"] == "all" else public["cohort"]["by_league"].get(row["scope"])
        if (identity in seen or expected_queries is None or row["method"] not in features or row["pool"] not in ("same_league", "same_position")
                or any(type(row[key]) is not int or row[key] <= 0 for key in ("features", "runs", "queries", "candidate_min", "candidate_max"))
                or row["queries"] != expected_queries or row["features"] != features[row["method"]]
                or row["runs"] != (len(seeds) if row["method"].startswith("nmf_") else 1)
                or not 1 <= row["candidate_min"] <= row["candidate_max"] <= row["queries"]
                or type(row["random_top1"]) not in (int, float) or not math.isfinite(row["random_top1"])
                or not 1 / row["candidate_max"] - 1e-12 <= row["random_top1"] <= 1 / row["candidate_min"] + 1e-12):
            raise ValueError("Invalid spatial style aggregate")
        for metric in ("top1", "top5", "mrr"):
            values = {key: summary[metric][key] for key in ("min", "mean", "max")}
            if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values())
                    or not 0 <= values["min"] <= values["max"] <= 1
                    or not values["min"] - 1e-12 <= values["mean"] <= values["max"] + 1e-12):
                raise ValueError("Invalid spatial style metric")
            row[metric] = values
        seen.add(identity); public["summary"].append(row)
    expected = {(scope, method, pool) for scope in ("all", *leagues) for method in features for pool in ("same_league", "same_position")}
    if seen != expected:
        raise ValueError("Invalid spatial style comparison coverage")
    diagnostics = [d for row in report["runs"] for d in row["diagnostics"].values()]
    public["convergence"] = {"fitted_blocks": len(diagnostics), "warnings": sum(len(d["warnings"]) for d in diagnostics),
                             "zero_basis_components": sum(d["zero_basis_components"] for d in diagnostics)}
    if "seed_stability" in report:
        source = report["seed_stability"]
        if (source.get("initialization") != "random" or source.get("metric") != "cosine"
                or public["parameters"].get("nmf_init") != "random"):
            raise ValueError("Invalid spatial style seed diagnostic")
        pairs, seen = [], set()
        components = {"shots": 4, "crosses": 4, "take_ons": 5, "passes": 5}
        expected = {(method, kind, a, b) for method in ("nmf_start_only", "nmf_start_end")
                    for kind in components for i, a in enumerate(seeds) for b in seeds[i + 1:]}
        for item in source["pairs"]:
            row = {key: item[key] for key in ("method", "kind", "seed_a", "seed_b", "components", "unassessed_pairs", "mean_cosine", "min_cosine")}
            identity = (row["method"], row["kind"], row["seed_a"], row["seed_b"])
            if identity not in expected or identity in seen or type(row["components"]) is not int or row["components"] != components.get(row["kind"]):
                raise ValueError("Invalid spatial style seed pair")
            n = row["components"]
            values, dead = list(item["matched_cosines"]), list(item["dead_components"])
            if (len(dead) != 2 or any(type(v) is not int or not 0 <= v <= n for v in dead)
                    or len(values) != n - max(dead) or type(row["unassessed_pairs"]) is not int
                    or row["unassessed_pairs"] != max(dead)
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)):
                raise ValueError("Invalid spatial style seed components")
            for key, expected_value in (("mean_cosine", sum(values) / len(values) if values else None), ("min_cosine", min(values) if values else None)):
                value = row[key]
                if ((expected_value is None and value is not None) or (expected_value is not None and
                        (type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-12)))):
                    raise ValueError("Invalid spatial style seed statistic")
            row.update(matched_cosines=values, dead_components=dead)
            pairs.append(row); seen.add(identity)
        if seen != expected:
            raise ValueError("Invalid spatial style seed coverage")
        public["seed_stability"] = {"initialization": "random", "metric": "cosine", "pairs": pairs,
                                    "alignment": "Maximum total cosine one-to-one matching of nonzero early-period basis components; dead components excluded."}
    return public


def public_literature(catalog):
    """Export sourced reading notes without local PDFs, paths or scratch notes."""
    if catalog.get("schema") != "SoccerAnalytics.FootballLiterature.v1":
        raise ValueError("Unrecognized football literature catalog")
    checked = date.fromisoformat(catalog["checked_at"])
    public = {key: catalog[key] for key in ("schema", "checked_at", "scope")}
    seen = set()
    strings = ("id", "citation_key", "title", "authors", "venue", "topic", "publication_status",
               "read_scope", "source_url", "publication_url", "data", "validation", "limitations", "application")
    for group in ("papers", "candidates"):
        public[group] = []
        for paper in catalog[group]:
            row = {key: paper[key] for key in strings + ("year", "priority", "evidence_pages")}
            if (any(not isinstance(row[key], str) or not row[key].strip() for key in strings)
                    or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", row["id"]) or row["id"] in seen
                    or type(row["year"]) is not int or not 1900 <= row["year"] <= checked.year
                    or row["topic"] not in ("visualization", "sports-analytics")
                    or row["publication_status"] not in ("published", "preprint")
                    or type(row["priority"]) is not int or row["priority"] not in (1, 2, 3)
                    or not isinstance(row["evidence_pages"], list)
                    or any(type(page) is not int or page < 1 for page in row["evidence_pages"])
                    or group == "papers" and (row["read_scope"] == "metadata_only" or not row["evidence_pages"])
                    or group == "candidates" and row["read_scope"] != "metadata_only"):
                raise ValueError("Invalid or duplicate literature evidence")
            for key in ("source_url", "publication_url"):
                url = urlparse(row[key])
                if url.scheme != "https" or not url.hostname or url.username or url.password:
                    raise ValueError("Literature source requires a public HTTPS URL")
            if "source_checks" in paper:
                checks = paper["source_checks"]
                if not isinstance(checks, list) or not 1 <= len(checks) <= 4:
                    raise ValueError("Literature source checks require 1–4 public references")
                row["source_checks"] = []
                for check in checks:
                    if not isinstance(check, dict) or any(not isinstance(check.get(key), str) or not check[key].strip() for key in ("label", "url", "note")):
                        raise ValueError("Literature source check is incomplete")
                    url = urlparse(check["url"])
                    if url.scheme != "https" or not url.hostname or url.username or url.password:
                        raise ValueError("Literature source check requires a public HTTPS URL")
                    row["source_checks"].append({key: check[key] for key in ("label", "url", "note")})
            seen.add(row["id"])
            public[group].append(row)
    if not public["papers"]:
        raise ValueError("No reviewed football literature")
    return public


def public_market_reference(report):
    """Keep paired aggregate scores; never export match-level odds or predictions."""
    from app.analytics.market_reference import BOOKS, LEAGUES, OUTCOMES, SCHEMA, SOURCE_NOTES, SOURCE_WARNING

    if (report.get("schema") != SCHEMA or report.get("class_order") != list(OUTCOMES)
            or not re.fullmatch(r"[a-f0-9]{20}", report.get("run_id", ""))):
        raise ValueError("Unrecognized football market reference")
    rows, seen = [], set()
    for item in report["results"]:
        row = {key: item[key] for key in ("competition", "season", "book", "population", "paired_matches",
                "test_start", "test_end", "training_rows", "training_end", "cohort_sha256")}
        identity = (row["competition"], row["season"], row["book"])
        if identity in seen or row["competition"] not in LEAGUES or row["book"] not in BOOKS or row["season"] not in (2024, 2025, 2026):
            raise ValueError("Invalid or duplicate market cohort")
        seen.add(identity)
        row["excluded"] = {key: item["excluded"][key] for key in ("provider_stale_warning", "incomplete_pre_close", "incomplete_closing", "incomplete_both")}
        counts = [row[key] for key in ("population", "paired_matches", "training_rows")] + list(row["excluded"].values())
        if (any(type(n) is not int or n < 0 for n in counts)
                or row["population"] != row["paired_matches"] + sum(row["excluded"].values())
                or not row["training_end"] < row["test_start"] <= row["test_end"]
                or not re.fullmatch(r"[a-f0-9]{64}", row["cohort_sha256"])):
            raise ValueError("Inconsistent market cohort accounting or chronology")
        row["metrics"] = {}
        for stage in ("prior", "pre_close", "closing"):
            scores = item["metrics"][stage]
            if row["paired_matches"] == 0:
                if scores is not None:
                    raise ValueError("Empty market cohort cannot have scores")
                row["metrics"][stage] = None
                continue
            scores = {key: scores[key] for key in ("log_loss", "brier", "accuracy", "mean_overround")}
            if (any(type(scores[k]) not in (int, float) or not math.isfinite(scores[k]) or scores[k] < 0 for k in ("log_loss", "brier", "accuracy"))
                    or scores["brier"] > 2 or scores["accuracy"] > 1
                    or stage == "prior" and scores["mean_overround"] is not None
                    or stage != "prior" and (type(scores["mean_overround"]) not in (int, float) or not math.isfinite(scores["mean_overround"]) or scores["mean_overround"] <= -1)):
                raise ValueError("Invalid market probability score")
            row["metrics"][stage] = scores
        rows.append(row)
    source = report["source"]
    if source["provider"] != "football-data.co.uk" or source["notes_url"] != SOURCE_NOTES or source["warning_url"] != SOURCE_WARNING:
        raise ValueError("Unexpected football market source")
    return {key: report[key] for key in ("schema", "run_id", "class_order", "scope", "method", "limits", "pinnacle_warning_from")} | {
        "source": {key: source[key] for key in ("provider", "notes_url", "warning_url", "input_sha256", "snapshot_retrieved_at")},
        "results": rows}


def public_representation_comparisons(report, raw):
    """Keep conditional loss ranges and reconcile their cohort and point scores."""
    keys = ('method', 'reference', 'test_sequences', 'test_matches', 'replicates', 'seed',
            'rng', 'scipy_version', 'quantiles', 'interval_method', 'weighting',
            'seed_aggregation', 'cohort_sha256', 'predictions_sha256', 'method_url',
            'engine_url', 'scope', 'limits')
    public = {key: raw[key] for key in keys}
    summaries = {row['method']: row for row in report['summaries']}
    test = report['data']['splits']['test']
    if (public['method'] != 'paired_match_case_bootstrap' or public['reference'] not in summaries
            or any(type(public[key]) is not int for key in ('test_sequences', 'test_matches', 'replicates', 'seed'))
            or public['test_sequences'] != test['sequences_used'] or public['test_matches'] != test['matches_used']
            or not 2 <= public['test_matches'] <= public['test_sequences']
            or public['replicates'] < 200 or public['seed'] < 0
            or public['rng'] != 'numpy.PCG64' or public['quantiles'] != [.025, .975]
            or public['interval_method'] != 'percentile' or public['weighting'] != 'pooled_sampled_passes'
            or public['seed_aggregation'] != 'mean_loss_across_fixed_seeds'
            or not isinstance(public['scipy_version'], str) or not public['scipy_version']
            or not isinstance(public['scope'], str) or not public['scope']
            or not isinstance(public['limits'], list) or not public['limits']
            or any(not isinstance(s, str) or not s for s in public['limits'])
            or any(not isinstance(public[k], str) or not re.fullmatch('[a-f0-9]{64}', public[k])
                   for k in ('cohort_sha256', 'predictions_sha256'))
            or public['method_url'] != 'https://journal.r-project.org/articles/RJ-2023-015/'
            or public['engine_url'] != 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html'):
        raise ValueError('Invalid paired representation resampling metadata or cohort')
    reference = summaries[public['reference']]
    seen, rows = set(), []
    metrics = ('completion_log_loss', 'completion_brier', 'forward_displacement_mae_pitch_pct')
    for item in raw['comparisons']:
        name = item['method']
        if (name not in summaries or name == public['reference'] or name in seen
                or type(item['runs']) is not int or item['runs'] != summaries[name]['runs']):
            raise ValueError('Invalid paired representation comparison methods')
        seen.add(name)
        row = {'method': name, 'runs': item['runs']}
        for metric in metrics:
            result = {key: item[metric][key] for key in ('difference', 'lower', 'upper')}
            expected = summaries[name]['test'][metric]['mean'] - reference['test'][metric]['mean']
            tolerance = 2e-4 if metric == 'forward_displacement_mae_pitch_pct' else 2e-6
            if (any(type(v) not in (int, float) or not math.isfinite(v) for v in result.values())
                    or result['lower'] > result['upper']
                    or not math.isclose(result['difference'], expected, abs_tol=tolerance, rel_tol=0)):
                raise ValueError('Invalid paired representation interval or score accounting')
            row[metric] = result
        rows.append(row)
    if seen != set(summaries) - {public['reference']}:
        raise ValueError('Incomplete paired representation comparisons')
    public['comparisons'] = rows
    return public


def public_representation_summary(report):
    """Accept a full local run receipt, but publish only explicit aggregate fields."""
    public = {
        key: report[key]
        for key in (
            "run_id",
            "observed_at",
            "code_sha256",
            "summaries",
            "input_fields",
            "targets",
            "limits",
        )
    }
    config_fields = (
        "target_league",
        "train_cap",
        "epochs",
        "seeds",
        "encoder",
        "hidden_size",
        "event_embedding_size",
        "sequence_length",
        "batch_size",
        "learning_rate",
        "optimizer",
        "mask_rate",
        "multi_task_loss",
    )
    public["configuration"] = {
        key: report["configuration"][key] for key in config_fields
    }
    if "tabular_baselines" in report["configuration"]:
        public["configuration"]["tabular_baselines"] = {
            key: report["configuration"]["tabular_baselines"][key]
            for key in ("inputs", "tree", "mlp")
        }
    if "environment" in report:
        public["environment"] = {
            key: report["environment"][key]
            for key in (
                "python",
                "torch",
                "numpy",
                "sklearn",
                "device",
                "torch_threads",
            )
        }
        if 'scipy' in report['environment']:
            public['environment']['scipy'] = report['environment']['scipy']
    data = report["data"]
    public["data"] = {
        key: data[key]
        for key in (
            "provider",
            "season",
            "target_league",
            "validation_start",
            "test_start",
            "inputs",
        )
    }
    split_fields = (
        "sequences_available",
        "sequences_used",
        "matches_used",
        "date_start",
        "date_end",
        "leagues",
        "completion_rate",
    )
    public["data"]["splits"] = {
        split: {key: data["splits"][split][key] for key in split_fields}
        for split in ("source", "train", "validation", "test")
    }
    if 'test_comparisons' in report:
        public['test_comparisons'] = public_representation_comparisons(report, report['test_comparisons'])
    return public


def public_model_lab(experiments):
    """Publish per-candidate aggregate metrics; never fitted state or individual rows."""
    labels = {"scores": "比分模型", "ratings": "实力评级", "calibration": "概率校准", "trees": "树与特征搜索"}
    groups = []
    def metric(value, name):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid model-lab {name}")
        if name in ("accuracy", "rps") and value > 1 or name == "brier_score" and value > 2:
            raise ValueError(f"Out-of-range model-lab {name}")
        return value
    for name, payload in experiments.items():
        if name not in labels or payload.get("errors") or not payload.get("results"):
            raise ValueError("Incomplete model-lab experiment")
        for experiment in payload["results"]:
            count = experiment["holdout_rows"]
            if type(count) is not int or count <= 0:
                raise ValueError("Invalid model-lab holdout cohort")
            if name == "calibration":
                start = experiment["final_windows"]["evaluation_first_kickoff"]
                end = experiment["final_windows"]["evaluation_last_kickoff"]
            else:
                start, end = experiment["holdout_first_kickoff"], experiment["holdout_last_kickoff"]
            if not isinstance(start, str) or not isinstance(end, str) or start > end:
                raise ValueError("Invalid model-lab holdout interval")
            rows, ids = [], set()
            for candidate in experiment["results"]:
                model = candidate.get("model", candidate.get("method"))
                scope = candidate.get("scope", "")
                key = f"{model}:{scope}" if scope else model
                if not isinstance(model, str) or not model or key in ids:
                    raise ValueError("Invalid or duplicate model-lab candidate")
                ids.add(key)
                raw = candidate.get("holdout_metrics")
                holdout = ({k: metric(raw[k], k) for k in ("log_loss", "brier_score", "accuracy", "rps")} if raw is not None else
                           {k: metric(candidate["holdout_" + k], k) for k in ("log_loss", "brier_score", "accuracy", "rps")})
                if raw is not None and raw["rows"] != count or candidate.get("holdout_rows", count) != count:
                    raise ValueError("Model-lab candidate cohort mismatch")
                rows.append({"id": key, "model": model, "scope": scope,
                             "selection_log_loss": metric(candidate["selection_log_loss"], "log_loss"), "holdout": holdout})
            selected = experiment["selected"]
            selected_id = f"{selected['model']}:{selected['scope']}" if isinstance(selected, dict) else selected
            if selected_id not in ids:
                raise ValueError("Selected model-lab candidate missing")
            sources = [{k: snapshot[k] for k in ("provider", "source", "season", "competition", "fetched_at", "sha256")}
                       for snapshot in experiment["source_snapshots"]]
            if not sources or any(not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) for row in sources):
                raise ValueError("Invalid model-lab source identity")
            groups.append({"id": f"{name}-{experiment['competition']}", "label": labels[name], "kind": name,
                           "competition": experiment["competition"], "seasons": payload["seasons"], "holdout_rows": count,
                           "holdout_start": start, "holdout_end": end, "selected": selected_id,
                           "selected_on": experiment["selected_on"], "sources": sources, "candidates": rows})
    public = {"schema": "Touchline.ModelLab.v1", "scope": "Historical offline experiments; development-selected candidates, frozen mappings, chronological holdouts. Groups have distinct input and warmup policies; do not pool their rows or rank across groups.", "groups": groups}
    json.dumps(public, allow_nan=False)
    return public
