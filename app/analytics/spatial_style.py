"""Chronological, closed-set on-ball style retrieval; independent of live models."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import warnings

from .spatial_style_data import KINDS, LEAGUES, CUT, load_data
from .artifacts import file_digest
from app.file_io import write_json_atomic

COMPONENTS = {"shots": 4, "crosses": 4, "take_ons": 5, "passes": 5}


def fit_style_vectors(early, late, seed, start_only=False):
    matrices, basis, diagnostics = {"early": [], "late": []}, {}, {}
    with threadpool_limits(limits=1):
        for kind in KINDS:
            inputs = {}
            for block, maps in (("early", early), ("late", late)):
                values = maps[kind][:, :1] if start_only and kind == "passes" else maps[kind]
                inputs[block] = np.asarray(values).reshape(len(values), -1)
                if not np.isfinite(inputs[block]).all() or np.any(inputs[block] < 0):
                    raise ValueError("Invalid nonnegative spatial features")
            if inputs["early"].shape[1] != inputs["late"].shape[1]:
                raise ValueError("Early and late spatial grids differ")
            model = NMF(n_components=COMPONENTS[kind], init="random", solver="cd", max_iter=1000, tol=1e-4, random_state=seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                train = model.fit_transform(inputs["early"])
                test = model.transform(inputs["late"])
            mass = model.components_.sum(axis=1)
            # Fix NMF's arbitrary scale: a loading is mass in a unit-L1 spatial basis.
            basis[kind] = np.divide(model.components_, mass[:, None], out=np.zeros_like(model.components_), where=mass[:, None] > 0)
            matrices["early"].append(train * mass)
            matrices["late"].append(test * mass)
            diagnostics[kind] = {"iterations": int(model.n_iter_), "warnings": [str(w.message) for w in caught],
                                 "zero_basis_components": int(np.count_nonzero(mass == 0))}
    return {block: np.concatenate(values, axis=1) for block, values in matrices.items()} | {"basis": basis, "diagnostics": diagnostics}


def match_basis(first, second):
    """Match active components by cosine; dead components provide no evidence."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (first.ndim != 2 or first.shape != second.shape or not all(first.shape)
            or any(not np.isfinite(a).all() or np.any(a < 0) for a in (first, second))):
        raise ValueError("Invalid spatial bases")
    normalized, dead = [], []
    for a in (first, second):
        norms = np.linalg.norm(a, axis=1)
        active = norms > 0
        normalized.append(a[active] / norms[active, None])
        dead.append(int((~active).sum()))
    similarity = np.clip(normalized[0] @ normalized[1].T, 0, 1)
    rows, columns = linear_sum_assignment(-similarity)
    values = similarity[rows, columns].tolist()
    return {'matched_cosines': values, 'dead_components': dead,
            'unassessed_pairs': len(first) - len(values),
            'mean_cosine': float(np.mean(values)) if values else None,
            'min_cosine': min(values) if values else None}


def retrieval_scores(gallery, queries, players):
    gallery, queries = np.asarray(gallery), np.asarray(queries)
    if (gallery.shape != queries.shape or gallery.ndim != 2 or len(gallery) != len(players) or not len(players)
            or not np.isfinite(gallery).all() or not np.isfinite(queries).all()):
        raise ValueError("Invalid paired style retrieval inputs")
    distances = cdist(queries, gallery, metric="cityblock")
    result = {}
    for pool in ("same_league", "same_position"):
        records = []
        for i, player in enumerate(players):
            mask = np.array([p["league"] == player["league"] and (pool == "same_league" or p["position"] == player["position"]) for p in players])
            d = distances[i, mask]; correct = distances[i, i]
            tied = np.isclose(d, correct, atol=1e-10, rtol=0)
            before = int(np.count_nonzero((d < correct) & ~tied)); ties = int(tied.sum())
            if not ties:
                raise ValueError("True player absent from retrieval tie group")
            records.append({"top1": min(max(1 - before, 0), ties) / ties,
                            "top5": min(max(5 - before, 0), ties) / ties,
                            "mrr": float(np.mean(1 / np.arange(before + 1, before + ties + 1))),
                            "random_top1": 1 / len(d), "candidates": len(d)})
        result[pool] = {key: float(np.mean([r[key] for r in records])) for key in ("top1", "top5", "mrr", "random_top1")}
        result[pool].update(queries=len(players), candidate_min=min(r["candidates"] for r in records), candidate_max=max(r["candidates"] for r in records))
    return result


def run(data, output, scratch, seeds=(7, 17, 42)):
    """Fit on early actions only and publish aggregate retrieval comparisons."""
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2 ** 32 for s in seeds):
        raise ValueError("Style seeds must be distinct nonnegative integers")
    if not data["dates"]["early"]["end"] < CUT <= data["dates"]["late"]["start"]:
        raise ValueError("Style retrieval windows overlap")
    output, scratch = Path(output), Path(scratch)
    output.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    players, maps = data["players"], data["maps"]
    cohort = [{**p, "minutes": data["minutes"][i].tolist()} for i, p in enumerate(players)]
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}
    parameters = {"grid": data["grid"], "sigma_cells": data["sigma"], "minimum_minutes_each_period": data["minimum_minutes"],
                  "components": COMPONENTS, "seeds": list(seeds), "distance": "Manhattan", "basis_scale": "unit_L1", "nmf_max_iter": 1000,
                  "nmf_init": "random", "nmf_solver": "cd", "nmf_tol": 1e-4,
                  "component_selection": {"policy": "fixed_reference_configuration",
                                          "reference": "https://tomdecroos.github.io/reports/ecml19_tomd.pdf",
                                          "local_70_percent_variance_selection_performed": False},
                  "smoothing_units": "grid_cells", "display_interpolation": "nearest"}
    inputs = {"source": data["source"], "cohort_sha256": hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest(),
              "extractor_sha256": file_digest(Path(__file__).with_name("spatial_style_data.py")), "experiment_sha256": file_digest(Path(__file__)),
              "parameters": parameters, "versions": versions}
    run_id = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    rows, figure_basis, stability, artifact_hashes = [], None, [], {}

    def record(name, seed, gallery, queries, diagnostics):
        scores = {"all": retrieval_scores(gallery, queries, players)}
        for league in LEAGUES:
            indices = [i for i, p in enumerate(players) if p["league"] == league]
            if indices:
                scores[league] = retrieval_scores(gallery[indices], queries[indices], [players[i] for i in indices])
        rows.append({"method": name, "seed": seed, "features": int(gallery.shape[1]), "scores": scores, "diagnostics": diagnostics})
        print(f"{name}, seed={seed}: same-league Top1={scores['all']['same_league']['top1']:.4f}, same-position Top1={scores['all']['same_position']['top1']:.4f}", flush=True)
        with (scratch / "notes.txt").open("a") as handle:
            handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

    print("Compare action frequency and spatial distributions on the same closed-set cohort; these are style stability proxies, not ability scores.", flush=True)
    frequency = {block: np.column_stack([maps[block][kind][:, 0].sum(axis=(1, 2)) for kind in KINDS]) for block in maps}
    full = {block: np.concatenate([maps[block][kind].reshape(len(players), -1) for kind in KINDS], axis=1) for block in maps}
    record("frequency", None, frequency["early"], frequency["late"], {})
    record("heatmaps", None, full["early"], full["late"], {})
    for start_only in (True, False):
        bases = {}
        for seed in seeds:
            method = "nmf_start_only" if start_only else "nmf_start_end"
            print(f"Fit {method} with seed {seed}; only pre-{CUT} actions enter basis fitting.", flush=True)
            fitted = fit_style_vectors(maps["early"], maps["late"], seed, start_only)
            bases[seed] = fitted["basis"]
            record(method, seed, fitted["early"], fitted["late"], fitted["diagnostics"])
            artifact = scratch / f"{method}-{seed}.npz"
            np.savez_compressed(artifact, early=fitted["early"], late=fitted["late"], **fitted["basis"])
            artifact_hashes[artifact.name] = file_digest(artifact)
            if not start_only and seed == seeds[-1]:
                figure_basis = fitted["basis"]["passes"]
        for a, b in combinations(seeds, 2):
            for kind in KINDS:
                stability.append({'method': method, 'kind': kind, 'seed_a': a, 'seed_b': b,
                                  'components': COMPONENTS[kind], **match_basis(bases[a][kind], bases[b][kind])})
    summaries = []
    for scope in rows[0]["scores"]:
        for method in ("frequency", "heatmaps", "nmf_start_only", "nmf_start_end"):
            runs = [r for r in rows if r["method"] == method]
            for pool in ("same_league", "same_position"):
                scores = [r["scores"][scope][pool] for r in runs]
                summary = {"scope": scope, "method": method, "pool": pool, "features": runs[0]["features"], "runs": len(runs),
                           **{key: scores[0][key] for key in ("queries", "candidate_min", "candidate_max", "random_top1")}}
                for metric in ("top1", "top5", "mrr"):
                    values = [s[metric] for s in scores]
                    summary[metric] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
                summaries.append(summary)
    report = {"schema": "SoccerAnalytics.SpatialStyle.v1", "run_id": run_id, "source": data["source"], "inputs": inputs,
              "windows": data["dates"], "parameters": parameters, "cohort": {"players": len(players), "by_league": dict(Counter(p["league"] for p in players)),
              "same_club_both_periods": True, "exposure_source": "supplied_playerank_minutes"}, "audit": data["audit"], "runs": rows, "summary": summaries,
              "paper_url": "https://tomdecroos.github.io/reports/ecml19_tomd.pdf",
              "scope": "Wyscout 2017/18 五大联赛，同一俱乐部、两段各至少 900 分钟的非门将闭集识别。2018-01-01 前拟合热图基底，之后的动作只做转换和检索。",
              "method": "50×50 网格，每90分钟频率，Gaussian sigma=1 格；传球起点与终点分开拼接。固定 4/4/5/5 个 NMF 分量，使用 random 初始化和固定种子，基底 L1 归一化，使用 Manhattan 距离。同距名次按均匀随机排序分摊。",
              "seed_stability": {'initialization': 'random', 'metric': 'cosine',
                                 'alignment': 'Maximum total cosine one-to-one matching of nonzero early-period basis components; dead components excluded.',
                                 'pairs': stability},
              "limits": ["这是同赛季前后段识别稳定性，不是论文的跨赛季复现，也不是球员能力、效率或转会潜力评分。",
                         "Wyscout 的 Ground attacking duel 用作持球进攻对抗起点，不等于原论文中的带球运输动作；不补造连续带球。定位球与门将排除。",
                         "样本按两段出场时间和同队条件筛选，结果仅适用于该闭集；球队、位置和战术的影响仍可能保留。",
                         "传球起终点是两个边际分布，不保存逐次起终点配对；没有无球活动或站位时长数据。",
                         "NMF 使用三个明确随机起点。种子范围只反映初始化差异，不是置信区间；未据后段分数选择网格、分量或平滑参数。",
                         "分量相似度先做最大总余弦的一对一匹配，只比较跨联赛共享的早段基底，零分量不计为稳定。高相似度不证明球风识别准确，也不检验时间漂移。",
                         "同位置候选池仍限定同联赛。总体指标按球员加权。每个联赛沿用同一个跨联赛早段基底，未分别拟合。"]}
    write_json_atomic(scratch / "cohort.json", cohort)
    write_json_atomic(scratch / "run-receipt.json", inputs | {"run_id": run_id, "artifact_sha256": artifact_hashes})
    write_json_atomic(output / "spatial-style.json", report)
    from .visualization.spatial_style import figures
    figures(report, figure_basis, output / "spatial-style", seeds[-1])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/football.sqlite3"))
    parser.add_argument("--sources", type=Path, default=Path(".tmp/open-data/wyscout"))
    parser.add_argument("--exposure-cache", type=Path, default=Path(".tmp/assessments/player-data.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--scratch", type=Path, default=Path(".tmp/spatial-style"))
    args = parser.parse_args()
    result = run(load_data(args.database, args.sources, args.exposure_cache), args.output, args.scratch)
    print(json.dumps({"run_id": result["run_id"], "players": result["cohort"]["players"]}))
