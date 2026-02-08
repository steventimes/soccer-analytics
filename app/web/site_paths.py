from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def site_root() -> Path:
    return repo_root() / "docs"


def data_dir() -> Path:
    return site_root() / "data"