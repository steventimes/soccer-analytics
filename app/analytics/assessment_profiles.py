"""Transparent, exposure-aware observed profiles and unsupervised playing styles."""

from collections import defaultdict
import json
from pathlib import Path
from .assessment_data import (
    COUNT_FIELDS,
    fit_priors,
    player_features,
    beta_estimate,
    gamma_estimate,
    rates,
)


def season_profiles(panels, minimum_minutes=450):
    groups = {}
    for row in panels:
        key = (row["entity_id"], row["league"])
        if key not in groups:
            groups[key] = {
                **row,
                **dict.fromkeys(COUNT_FIELDS + ("minutes", "matches"), 0),
                "teams": defaultdict(float),
            }
        g = groups[key]
        g["teams"][row["team"]] += row["minutes"]
        g["as_of"] = max(g["as_of"], row["as_of"])
        g["age"] = row["age"]
        for name in COUNT_FIELDS + ("minutes", "matches"):
            g[name] += row[name]
    result = []
    for g in groups.values():
        teams = g.pop("teams")
        g["team"] = max(teams, key=teams.get)
        if g["minutes"] >= minimum_minutes:
            result.append(g)
    return result


def profile_scores(profiles, priors):
    import numpy as np
    from scipy.stats import beta, gamma

    result = []
    groups = defaultdict(list)
    for row in profiles:
        feature = player_features(row, priors)
        record = {
            **row,
            "posterior_features": feature,
            "observed_metrics": rates(row),
            "intervals": {},
        }
        prior = priors.get(
            (row["league"], row["position"]),
            priors.get(("*", row["position"]), priors[("*", "*")]),
        )
        for name, wins, attempts in [
            ("pass_completion", "accurate_passes", "passes"),
            ("duel_win_rate", "duel_wins", "duels"),
            ("keeper_success_rate", "keeper_success", "keeper_attempts"),
        ]:
            estimation = beta_estimate(
                row[wins], row[attempts], min(0.999, max(0.001, prior.get(name) or 0.5))
            )
            record["intervals"][name] = {
                **estimation,
                "conditional_ci95": beta.ppf(
                    [0.025, 0.975], estimation["alpha"], estimation["beta"]
                ).tolist(),
            }
        for field in ["goals", "progressive_passes", "interceptions"]:
            estimation = gamma_estimate(
                row[field],
                row["minutes"],
                max(prior.get(field + "_per90") or 0.001, 0.001),
            )
            record["intervals"][field + "_per90"] = {
                **estimation,
                "conditional_ci95": gamma.ppf(
                    [0.025, 0.975], estimation["alpha"], scale=1 / estimation["rate"]
                ).tolist(),
            }
        result.append(record)
        groups[(row["league"], row["position"])].append(record)
    dimensions = {
        "attacking_output": ["goals_per90"],
        "progression_and_distribution": ["progressive_passes_per90", "pass_completion"],
        "recorded_defensive_output": [
            "defensive_duels_per90",
            "interceptions_per90",
            "duel_win_rate",
        ],
    }
    weights = {
        "Forward": [0.5, 0.35, 0.15],
        "Midfielder": [0.2, 0.5, 0.3],
        "Defender": [0.1, 0.3, 0.6],
    }
    for group in groups.values():
        for row in group:
            scores = {}

            def percentile(field):
                values = np.array([r["posterior_features"][field] for r in group])
                value = row["posterior_features"][field]
                return float(
                    100
                    * (np.sum(values < value) + 0.5 * np.sum(values == value))
                    / len(values)
                )

            if row["position"] == "Goalkeeper":
                scores = {
                    "distribution": (
                        percentile("pass_completion") if row["passes"] >= 20 else None
                    ),
                    "recorded_save_outcomes": (
                        percentile("keeper_success_rate")
                        if row["keeper_attempts"] >= 10
                        else None
                    ),
                }
                overall = None
            else:
                for dimension, fields in dimensions.items():
                    sufficient = not (
                        "duel_win_rate" in fields and row["duels"] < 10
                    ) and not ("pass_completion" in fields and row["passes"] < 20)
                    scores[dimension] = (
                        float(np.mean([percentile(f) for f in fields]))
                        if sufficient
                        else None
                    )
                w = weights.get(row["position"])
                overall = (
                    sum(scores[k] * weight for k, weight in zip(dimensions, w))
                    if w and all(v is not None for v in scores.values())
                    else None
                )
            row["evaluation"] = {
                "cohort": row["league"] + "/" + row["position"],
                "cohort_players": len(group),
                "dimensions": scores,
                "observed_index": overall,
                "index_definition": "role-weighted percentile of observed posterior metrics; a descriptive index, not CA, market value or intrinsic ability",
                "weights": weights.get(row["position"]),
                "goalkeeper_note": (
                    "Save outcomes are provider-labelled and not adjusted for shot difficulty"
                    if row["position"] == "Goalkeeper"
                    else None
                ),
            }
    return result


def styles(profiles, output):
    import joblib
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA, NMF
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors
    from threadpoolctl import threadpool_limits

    rows = [p for p in profiles if p["position"] != "Goalkeeper"]
    features = [
        "passes_per90",
        "progressive_passes_per90",
        "shots_per90",
        "goals_per90",
        "defensive_duels_per90",
        "interceptions_per90",
        "duels_per90",
        "pass_completion",
        "duel_win_rate",
    ]
    x = np.array([[p["posterior_features"][f] for f in features] for p in rows])
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    validation = np.array([int(p["entity_id"]) % 5 == 0 for p in rows])
    train = ~validation
    # Refit scaling on the unsupervised training subset before choosing k.
    scaler.fit(x[train])
    z = scaler.transform(x)
    k_candidates = []
    g_candidates = []
    with threadpool_limits(limits=1):
        for k in range(2, 8):
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z[train])
            labels = km.predict(z[validation])
            score = (
                silhouette_score(z[validation], labels)
                if 1 < len(set(labels)) < len(labels)
                else None
            )
            k_candidates.append(
                {
                    "k": k,
                    "validation_silhouette": (
                        float(score) if score is not None else None
                    ),
                    "model": km,
                }
            )
            gm = GaussianMixture(
                n_components=k, covariance_type="diag", n_init=3, random_state=42
            ).fit(z[train])
            g_candidates.append(
                {"k": k, "training_bic": float(gm.bic(z[train])), "model": gm}
            )
        selected = max(
            (r for r in k_candidates if r["validation_silhouette"] is not None),
            key=lambda r: r["validation_silhouette"],
        )
        km = selected["model"]
        labels = km.predict(z)
        again = KMeans(n_clusters=selected["k"], n_init=10, random_state=43).fit(
            z[train]
        )
        gm = min(g_candidates, key=lambda r: r["training_bic"])
        probabilities = gm["model"].predict_proba(z)
        pca = PCA(n_components=2, random_state=42).fit(z[train])
        coords = pca.transform(z)
        scale = np.maximum(np.quantile(x[train], 0.95, axis=0), 0.01)
        nonnegative = np.clip(x / scale, 0, 2)
        nmf = NMF(n_components=4, init="nndsvda", max_iter=1000, random_state=42).fit(
            nonnegative[train]
        )
        contributions = nmf.transform(nonnegative)
    centroids = []
    for cluster in range(selected["k"]):
        members = np.flatnonzero(labels == cluster)
        mean = x[members].mean(axis=0)
        centroids.append(
            {
                "cluster": cluster,
                "players": len(members),
                "mean_observed_features": dict(zip(features, map(float, mean))),
            }
        )
    for i, row in enumerate(rows):
        row["style"] = {
            "kmeans_cluster": int(labels[i]),
            "mixture_cluster": int(np.argmax(probabilities[i])),
            "mixture_membership": probabilities[i].tolist(),
            "pca": coords[i].tolist(),
            "nmf_weights": contributions[i].tolist(),
        }
    # Neighbours compare historical players in the same broad role, without price claims.
    for position in {p["position"] for p in rows}:
        indices = [i for i, p in enumerate(rows) if p["position"] == position]
        near = NearestNeighbors(
            n_neighbors=min(6, len(indices)), metric="euclidean"
        ).fit(z[indices])
        distances, neighbours = near.kneighbors(z[indices])
        for local, i in enumerate(indices):
            rows[i]["similar_players"] = [
                {
                    "player_id": rows[indices[j]]["entity_id"],
                    "name": rows[indices[j]]["name"],
                    "league": rows[indices[j]]["league"],
                    "distance": float(d),
                }
                for d, j in zip(distances[local], neighbours[local])
                if rows[indices[j]]["entity_id"] != rows[i]["entity_id"]
            ][:5]
    report = {
        "features": features,
        "kmeans_candidates": [
            {k: v for k, v in r.items() if k != "model"} for r in k_candidates
        ],
        "selected_kmeans_k": selected["k"],
        "gmm_candidates": [
            {k: v for k, v in r.items() if k != "model"} for r in g_candidates
        ],
        "selected_gmm_k": gm["k"],
        "kmeans_seed_stability_ari": float(
            adjusted_rand_score(km.predict(z[validation]), again.predict(z[validation]))
        ),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "nmf_components": nmf.components_.tolist(),
        "centroids": centroids,
        "notes": [
            "Clusters describe styles and activity, not a good-to-bad ranking.",
            "Scaling fits the unsupervised training subset; k-means k uses held-out silhouette; GMM k uses training BIC.",
            "Neighbour suggestions are historical role-compatible style comparisons, not current transfer recommendations.",
        ],
    }
    joblib.dump(
        {
            "features": features,
            "scaler": scaler,
            "kmeans": km,
            "gmm": gm["model"],
            "pca": pca,
            "nmf": nmf,
            "nmf_scale": scale,
            "scope": "Historical 2017/18 descriptive outfield styles",
        },
        Path(output).with_suffix(".joblib"),
    )
    Path(output).write_text(json.dumps(report, indent=2))
    return report
