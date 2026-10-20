"""Additional public body-pose archives described by SkillCorner's verified manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import requests

from app.data_service.archive import _request, collect_repository, read_asset_json
from app.file_io import write_json_atomic as _atomic_write_json

logger = logging.getLogger(__name__)


def collect_bodypose(skillcorner_root: Path, *, refresh=False) -> dict:
    root = Path(skillcorner_root)
    if not (root / "manifest.json").exists():
        collect_repository("skillcorner", root)
    contract = read_asset_json(root, "data/bodypose/MANIFEST.json")
    if not contract["base_url"].startswith(
        "https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/"
    ):
        raise ValueError("Unexpected public bodypose origin")
    output = root / "external-bodypose"
    output.mkdir(exist_ok=True)
    files = []
    for name, entry in contract["files"].items():
        target = output / Path(name).name
        url = contract["base_url"] + "/" + name
        sha = hashlib.sha256()
        if target.exists() and not refresh:
            with target.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    sha.update(chunk)
            if (
                target.stat().st_size != entry["size_bytes"]
                or sha.hexdigest() != entry["sha256"]
            ):
                raise ValueError("Cached bodypose checksum mismatch")
        else:
            response = _request(requests.Session(), url, stream=True)
            temporary = None
            count = 0
            try:
                with tempfile.NamedTemporaryFile(dir=output, delete=False) as handle:
                    temporary = Path(handle.name)
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
                        sha.update(chunk)
                        count += len(chunk)
                if count != entry["size_bytes"] or sha.hexdigest() != entry["sha256"]:
                    raise ValueError("Published bodypose checksum mismatch")
                os.replace(temporary, target)
                temporary = None
            finally:
                response.close()
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        files.append(
            {
                "path": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha.hexdigest(),
                "source_url": url,
            }
        )
        logger.info(
            "Bodypose archive %s: %.2f GB", target.name, target.stat().st_size / 1e9
        )
    report = {
        "source": "SkillCorner public bodypose",
        "frame_rate_fps": 25,
        "files": files,
        "bytes": sum(item["bytes"] for item in files),
        "complete": True,
        "metadata_url": "https://github.com/SkillCorner/opendata/blob/master/data/bodypose/MANIFEST.json",
    }
    _atomic_write_json(output / "manifest.json", report)
    return report


def analyze_bodypose(skillcorner_root: Path, output_dir: Path) -> dict:
    import math
    import zipfile
    from collections import Counter

    try:
        import orjson

        loads = orjson.loads
    except ImportError:
        loads = json.loads
    root = Path(skillcorner_root) / "external-bodypose"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    reports = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                sha.update(chunk)
        if sha.hexdigest() != entry["sha256"]:
            raise ValueError("Bodypose source checksum mismatch")
        frames = player_frames = pose_player_frames = joint_observations = 0
        periods = Counter()
        errors = Counter()
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith(".jsonl")
                and "__MACOSX" not in Path(name).parts
                and not Path(name).name.startswith("._")
            ]
            if not members:
                raise ValueError("Bodypose ZIP has no JSONL")
            for member in members:
                with archive.open(member) as handle:
                    for line in handle:
                        frame = loads(line)
                        frames += 1
                        periods[str(frame.get("period"))] += 1
                        for player in frame.get("player_data", []):
                            player_frames += 1
                            joints = player.get("joints")
                            if not isinstance(joints, dict) or not joints:
                                continue
                            pose_player_frames += 1
                            joint_observations += len(joints)
                            # Systematic 1-in-25 frame sampling characterizes uncertainty, not motion.
                            if frames % 25 == 0:
                                for joint in joints.values():
                                    error = joint.get("p90_mae_cm")
                                    if (
                                        isinstance(error, (int, float))
                                        and math.isfinite(error)
                                        and error >= 0
                                    ):
                                        errors[min(199, int(error))] += 1
        row = {
            "file": entry["path"],
            "frames": frames,
            "frames_by_period": dict(periods),
            "player_frames": player_frames,
            "pose_player_frames": pose_player_frames,
            "pose_coverage_fraction": (
                pose_player_frames / player_frames if player_frames else None
            ),
            "joint_observations": joint_observations,
            "sampled_joint_error_histogram_cm": dict(errors),
        }
        reports.append(row)
        logger.info(
            "Bodypose analyzed %s: %d frames, %d joints",
            entry["path"],
            frames,
            joint_observations,
        )
    result = {
        "source": "SkillCorner",
        "matches": reports,
        "frame_rate_fps": 25,
        "notes": [
            "XY positions are interpolated from 10Hz; joint predictions are not ground-truth biomechanics.",
            "Z is reliable relative to the player centroid, not necessarily to the pitch.",
            "p90_mae_cm is a supplied predicted joint-error radius, sampled here once per 25 frames.",
            "Joint-error histogram uses one-centimeter bins; the final bin includes all errors >=199cm.",
            "Do not join bodypose and XY tracking by raw frame number; their frame rates differ.",
        ],
    }
    _atomic_write_json(output_dir / "bodypose-quality.json", result)
    return result
