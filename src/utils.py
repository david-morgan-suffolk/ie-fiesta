import os
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


def get_files(path: str | Path, type: str = ".pdf") -> list[tuple[str, str]]:
    return [
        (os.path.join(path, file), file.split(".")[0])
        for file in os.listdir(path)
        if file.endswith(type)
    ]


def get_sample_path() -> Path:
    return get_assets_path() / "sample-sets"


def get_sample_docs() -> list[Path]:
    return [Path(i) for i, _ in get_files(get_sample_path())]


def get_sample_images(suffix: str = ".png") -> list[Path]:
    dirs = [item for item in get_sample_path().iterdir() if item.is_dir()]
    files = []
    for d in dirs:
        for file in d.iterdir():
            if file.is_file and file.suffix == suffix:
                files.append(file)

    return files
