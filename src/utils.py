from pathlib import Path


def get_project_root(marker: str = ".here") -> Path:
    """Returns the project root directory."""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root (with {marker}) not found.")


def get_assets_path() -> Path:
    """Returns the project assets directory."""
    return get_project_root() / "assets/"


def get_samples_path() -> Path:
    return get_assets_path() / "sample-sets"
