import torch
from loguru import logger as log
from torchvision import datasets

from helpers import plot
from utils import get_assets_path

torch.manual_seed(0)

ROOT = get_assets_path() / "coco"
IMAGES = str(ROOT / "images")
LABELS = str(ROOT / "instances.json")


def main():
    dataset = datasets.CocoDetection(IMAGES, LABELS)
    sample = dataset[0]
    log.info(f"\n{sample = }")
    img, target = sample
    log.info(
        f"\n{type(img) = }\n{type(target) = }\n{type(target[0]) = }\n{target[0].keys() = }"
    )
    plot([dataset[0]])


if __name__ == "__main__":
    main()
