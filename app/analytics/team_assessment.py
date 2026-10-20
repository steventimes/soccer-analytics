"""Opponent-adjusted Poisson attack/defense ratings and observed team styles."""

from collections import defaultdict
import json
import math
from pathlib import Path


def poisson_ratings(matches, output, as_of="2026-10-09"):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import PoissonRegressor
    from sklearn.metrics import mean_poisson_deviance
    from threadpoolctl import threadpool_limits
    import joblib

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ratings = []
    evaluations = []
    failures = []
    groups = defaultdict(list)
    for row in matches:
        groups[row["competition"]].append(row)

    def expand(rows):
        x = []
        y = []
        dates = []
        for r in rows:
            for home in [True, False]:
                x.append(
                    [
                        r["home_id" if home else "away_id"],
                        r["away_id" if home else "home_id"],
                        int(home),
                    ]
                )
                y.append(r["home_goals" if home else "away_goals"])
                dates.append(r["date"])
        return np.array(x, dtype=object), np.array(y, float), dates

    for league, rows in groups.items():
        train = [
            r for r in rows if 2016 <= r["season"] <= 2023 and r["date"] < "2024-07-01"
        ]
        valid = [
            r
            for r in rows
            if r["season"] == 2024 and "2024-07-01" <= r["date"] < "2025-07-01"
        ]
        test = [
            r
            for r in rows
            if r["season"] == 2025 and "2025-07-01" <= r["date"] < "2026-07-01"
        ]
        if len(train) < 100 or len(valid) < 30 or len(test) < 30:
            continue
        best = None
        trials = []
        for alpha in [0.03, 0.3, 3.0]:
            x, y, dates = expand(train)
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
                x[:, :2]
            )
            xx = np.column_stack([encoder.transform(x[:, :2]), x[:, 2].astype(float)])
            xv, yv, _ = expand(valid)
            vx = np.column_stack([encoder.transform(xv[:, :2]), xv[:, 2].astype(float)])
            reference = np.datetime64("2024-07-01")
            weights = np.exp(
                -np.log(2)
                * (reference - np.array(dates, dtype="datetime64[D]")).astype(float)
                / 365.25
            )
            try:
                with threadpool_limits(limits=1):
                    model = PoissonRegressor(alpha=alpha, max_iter=500).fit(
                        xx, y, sample_weight=weights
                    )
                score = float(
                    mean_poisson_deviance(yv, np.maximum(0.0001, model.predict(vx)))
                )
                trials.append({"alpha": alpha, "validation_poisson_deviance": score})
                if best is None or score < best["score"]:
                    best = {"alpha": alpha, "score": score}
            except Exception as exc:
                failures.append({"league": league, "alpha": alpha, "message": str(exc)})
        if not best:
            continue
        fitted = train + valid
        x, y, dates = expand(fitted)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            x[:, :2]
        )
        xx = np.column_stack([encoder.transform(x[:, :2]), x[:, 2].astype(float)])
        weights = np.exp(
            -np.log(2)
            * (
                np.datetime64("2025-07-01") - np.array(dates, dtype="datetime64[D]")
            ).astype(float)
            / 365.25
        )
        with threadpool_limits(limits=1):
            model = PoissonRegressor(alpha=best["alpha"], max_iter=500).fit(
                xx, y, sample_weight=weights
            )
        xt, yt, _ = expand(test)
        pred = model.predict(
            np.column_stack([encoder.transform(xt[:, :2]), xt[:, 2].astype(float)])
        )
        means = [
            np.mean([r["away_goals"] for r in fitted]),
            np.mean([r["home_goals"] for r in fitted]),
        ]
        baseline = np.array([means[int(home)] for home in xt[:, 2]])
        evaluations.append(
            {
                "league": league,
                "train_matches": len(train),
                "validation_matches": len(valid),
                "test_matches": len(test),
                "trials": trials,
                "selected_alpha": best["alpha"],
                "test_poisson_deviance": float(
                    mean_poisson_deviance(yt, np.maximum(pred, 0.0001))
                ),
                "home_away_mean_deviance": float(
                    mean_poisson_deviance(yt, np.maximum(baseline, 0.0001))
                ),
                "training_label_end": max(r["date"] for r in fitted),
                "test_start": min(r["date"] for r in test),
            }
        )
        # Deployment fit is separate, after test evaluation. Its dated ratings use
        # only results already available at the requested cutoff.
        recent = [r for r in rows if r["season"] >= 2021 and r["date"] < as_of]
        x, y, dates = expand(recent)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            x[:, :2]
        )
        xx = np.column_stack([encoder.transform(x[:, :2]), x[:, 2].astype(float)])
        weights = np.exp(
            -np.log(2)
            * (np.datetime64(as_of) - np.array(dates, dtype="datetime64[D]")).astype(
                float
            )
            / 365.25
        )
        with threadpool_limits(limits=1):
            model = PoissonRegressor(alpha=best["alpha"], max_iter=500).fit(
                xx, y, sample_weight=weights
            )
        attack = dict(
            zip(encoder.categories_[0], model.coef_[: len(encoder.categories_[0])])
        )
        defense = dict(
            zip(encoder.categories_[1], model.coef_[len(encoder.categories_[0]) : -1])
        )
        latest = max(r["season"] for r in rows)
        current = {r["home_id"]: r["home"] for r in rows if r["season"] == latest}
        current.update({r["away_id"]: r["away"] for r in rows if r["season"] == latest})
        a_mean = float(np.mean(list(attack.values())))
        d_mean = float(np.mean(list(defense.values())))
        strengths = {tid: attack.get(tid, 0) - defense.get(tid, 0) for tid in current}
        for tid, name in current.items():
            ratings.append(
                {
                    "league": league,
                    "entity_id": tid,
                    "name": name,
                    "as_of": as_of,
                    "latest_season": latest,
                    "attack_multiplier": float(math.exp(attack.get(tid, 0) - a_mean)),
                    "goals_conceded_multiplier": float(
                        math.exp(defense.get(tid, 0) - d_mean)
                    ),
                    "neutral_expected_goals_for": float(
                        math.exp(model.intercept_ + attack.get(tid, 0) + d_mean)
                    ),
                    "neutral_expected_goals_against": float(
                        math.exp(model.intercept_ + defense.get(tid, 0) + a_mean)
                    ),
                    "relative_strength_percentile": 100
                    * (
                        sum(v < strengths[tid] for v in strengths.values())
                        + 0.5 * sum(v == strengths[tid] for v in strengths.values())
                    )
                    / len(strengths),
                    "weighted_match_exposure": float(
                        sum(
                            w
                            for r, w in zip(recent, weights[::2])
                            if tid in {r["home_id"], r["away_id"]}
                        )
                    ),
                    "scale_note": "Attack: larger stronger. Conceded multiplier: smaller stronger. Relative within league, not a global league-strength scale.",
                }
            )
        joblib.dump(
            {
                "encoder": encoder,
                "model": model,
                "league": league,
                "as_of": as_of,
                "half_life_days": 365.25,
                "selected_by": "validation_poisson_deviance",
            },
            output / f"poisson-{league}.joblib",
        )
    return {
        "ratings": ratings,
        "experiments": evaluations,
        "failures": failures,
        "notes": [
            "Poisson log mean contains team attack, opponent defense and home effect.",
            "Regularization selected before held-out season; deployment ratings then refit on already-observed results.",
            "Recency weights use one-year half life; promoted/low-exposure teams receive regularized estimates.",
            "Poisson count variance is a modeling assumption, not a guarantee of calibrated uncertainty.",
        ],
    }


def tactical_profiles(matches, team_styles, team_names=None):
    profiles = {}
    for record in team_styles:
        mid = int(record["match_id"])
        match = matches.get(str(mid), matches.get(mid))
        if not match:
            continue
        key = (match["league"], record["team_id"])
        if key not in profiles:
            profiles[key] = {
                "league": match["league"],
                "entity_id": record["team_id"],
                "matches": 0,
                "passes": 0,
                "progressive_passes": 0,
                "shots": 0,
                "defensive_duels": 0,
                "interceptions": 0,
                "duels": 0,
                "duel_wins": 0,
                "pass_length_sum": 0.0,
                "passes_with_length": 0,
                "long_passes": 0,
            }
        p = profiles[key]
        p["matches"] += 1
        for k in p:
            if k not in {"league", "entity_id", "matches"}:
                p[k] += record.get(k, 0)
    result = []
    for p in profiles.values():
        result.append(
            {
                **p,
                "features": {
                    "passes_per_match": p["passes"] / p["matches"],
                    "progressive_passes_per_match": p["progressive_passes"]
                    / p["matches"],
                    "shots_per_match": p["shots"] / p["matches"],
                    "defensive_duels_per_match": p["defensive_duels"] / p["matches"],
                    "interceptions_per_match": p["interceptions"] / p["matches"],
                    "long_pass_share": (
                        p["long_passes"] / p["passes"] if p["passes"] else 0.0
                    ),
                    "mean_pass_length": (
                        p["pass_length_sum"] / p["passes_with_length"]
                        if p["passes_with_length"]
                        else 0.0
                    ),
                },
            }
        )
    for row in result:
        row["name"] = (team_names or {}).get(str(row["entity_id"]))
    return result


def cluster_styles(profiles, output=None):
    import joblib
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    features = list(profiles[0]["features"])
    x = np.array([[p["features"][f] for f in features] for p in profiles])
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    candidates = []
    mixtures = []
    with threadpool_limits(limits=1):
        for k in range(2, 7):
            model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z)
            candidates.append(
                {
                    "k": k,
                    "silhouette": float(silhouette_score(z, model.labels_)),
                    "labels": model.labels_,
                    "model": model,
                }
            )
            gm = GaussianMixture(
                n_components=k, covariance_type="diag", random_state=42, n_init=3
            ).fit(z)
            mixtures.append(
                {"k": k, "bic": float(gm.bic(z)), "labels": gm.predict(z), "model": gm}
            )
        best = max(candidates, key=lambda c: c["silhouette"])
        mixture = min(mixtures, key=lambda c: c["bic"])
        pca = PCA(n_components=2).fit(z)
        coordinates = pca.transform(z)
    for p, label, gl, coord in zip(
        profiles, best["labels"], mixture["labels"], coordinates
    ):
        p["style"] = {
            "kmeans_cluster": int(label),
            "mixture_cluster": int(gl),
            "pca": coord.tolist(),
        }
    if output:
        joblib.dump(
            {
                "features": features,
                "scaler": scaler,
                "kmeans": best["model"],
                "gmm": mixture["model"],
                "pca": pca,
                "scope": "Historical 2017/18 descriptive team styles",
            },
            output,
        )
    return {
        "profiles": profiles,
        "features": features,
        "kmeans_candidates": [
            {k: v for k, v in c.items() if k not in {"labels", "model"}}
            for c in candidates
        ],
        "gmm_candidates": [
            {k: v for k, v in c.items() if k not in {"labels", "model"}}
            for c in mixtures
        ],
        "selected_k": best["k"],
        "selected_gmm_k": mixture["k"],
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "scope": "Descriptive full-season 2017/18 event styles; clusters have no good/bad ordering.",
    }
