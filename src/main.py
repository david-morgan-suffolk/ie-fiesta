import math

import torch
from loguru import logger as log
from PIL import Image
from torchvision import transforms as t
from transformers import (
    LayoutLMv2Processor,
)

from utils import get_assets_path, get_sample_images, get_tests_path

torch.manual_seed(0)

ROOT = get_assets_path() / "coco"
IMAGES = str(ROOT / "images")
LABELS = str(ROOT / "instances.json")

width: int = 0
height: int = 0


def resize(a: int, b: int, target_a: int = 224) -> tuple[int, int]:
    new_a = math.floor((a / b) * target_a)
    new_b = math.floor((b / a) * new_a)
    return (new_a, new_b)


def main():
    files = get_sample_images()
    log.info(f"Starting Processor for {len(files)} files")
    batch = tuple[str, list[any]]
    result_path = get_tests_path()

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
    to_img = t.ToPILImage()
    file = files[0]
    input_img = Image.open(file).convert("RGB")
    width, height = input_img.size
    new_sizes = resize(width, height, 224)
    input_img.resize(size=new_sizes)
    encoding = processor(images=input_img, return_tensors="pt", truncation=True)

    images = encoding.get("image")
    if isinstance(images, torch.Tensor):
        img = to_img(images.squeeze())
        loc = result_path / f"{file.stem}.JPEG"
        log.info(loc)
        img.save(loc, "JPEG")
        log.info(img)


if __name__ == "__main__":
    main()
