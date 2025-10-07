import os
from pathlib import Path

import numpy as np


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


def get_tests_path() -> Path:
    return get_assets_path() / "sample-tests"


def get_sample_docs() -> list[Path]:
    return [Path(i) for i, _ in get_files(get_sample_path())]


def get_sample_dirs():
    for d in get_sample_path().iterdir():
        if d.is_dir():
            yield d


def get_sample_images(suffix: str = ".png") -> list[Path]:
    files = []
    for d in get_sample_dirs():
        for file in d.iterdir():
            if file.is_file and file.suffix == suffix:
                files.append(file)

    return files


def get_sample_batch() -> dict[Path, list[Path]]:
    values = {}
    for d in get_sample_dirs():
        files = []
        for file in d.iterdir():
            if file.is_file and file.suffix == ".png":
                files.append(file)

        values[d] = files
    return values


def get_training_dir() -> Path:
    p = get_project_root() / ".training"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gcd(a, b):
    return abs(a) if b == 0 else gcd(b, a % b)


def show(imgs):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
