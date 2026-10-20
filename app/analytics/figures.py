"""Load verified scouting inputs and orchestrate source-backed football figures."""

from __future__ import annotations
import json
import os
from pathlib import Path
from app.data_service.archive import read_asset_json
from .visualization.football import (
    completed_pass_network,
    shot_map,
    passing_network,
    visible_space,
    running_position,
)


def generate_scouting_figures(
    data_root: Path,
    scouting_root: Path,
    output_dir: Path,
    *,
    match_id: int | None = None,
    network_window: tuple[int, int] | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    if network_window is not None and not 0 <= network_window[0] < network_window[1]:
        raise ValueError("Network window must have 0<=start<end first-half minutes")
    data_root, scouting_root = Path(data_root), Path(scouting_root)
    root = data_root / "statsbomb"
    manifest = json.loads((root / "manifest.json").read_text())
    index = {entry["path"]: entry for entry in manifest["files"]}
    report = json.loads((scouting_root / "statsbomb" / "report.json").read_text())
    if match_id is None:
        choices = [
            row
            for row in report["matches"]
            if row["home_team"] == "Spain"
            and row["away_team"] == "England"
            and row["date"] == "2024-07-14"
        ]
        chosen = (
            choices[0]
            if choices
            else next(row for row in reversed(report["matches"]) if row["360_events"])
        )
        match_id = chosen["match_id"]
    else:
        chosen = next(row for row in report["matches"] if row["match_id"] == match_id)
    events = read_asset_json(root, f"data/events/{match_id}.json", index)
    frames = read_asset_json(root, f"data/three-sixty/{match_id}.json", index)
    lineups = read_asset_json(root, f"data/lineups/{match_id}.json", index)
    lineup_names = {
        p["player_id"]: (
            p.get("player_nickname") or p["player_name"],
            p.get("jersey_number"),
        )
        for t in lineups
        for p in t["lineup"]
    }
    paths = shot_map(events, chosen, scouting_root=scouting_root, output_dir=output_dir)
    paths.extend(
        passing_network(
            events,
            chosen,
            lineup_names,
            network_window,
            scouting_root=scouting_root,
            output_dir=output_dir,
        )
    )
    paths.extend(
        visible_space(
            events, frames, scouting_root=scouting_root, output_dir=output_dir, chosen=chosen
        )
    )
    paths.extend(running_position(scouting_root=scouting_root, output_dir=output_dir))
    return paths
