import math

import torch
from loguru import logger as log
from PIL import Image
from torchvision import transforms as t
from transformers import (
    LayoutLMv2ImageProcessor,
    LayoutLMv2Processor,
    LayoutLMv2Tokenizer,
)

from utils import get_assets_path, get_sample_batch, get_tests_path

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


def main(save: bool):
    batch = get_sample_batch()
    log.info(f"Starting Processor for {len(batch.keys())} docs")
    result_path = get_tests_path()
    # each source is a single pdf doc set
    to_img = t.ToPILImage()
    image_processor = LayoutLMv2ImageProcessor()
    tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(image_processor, tokenizer)

    for batch_source, batch_images in batch.items():
        # here we only need the first img to get this data from, since we are mounting all of this into a 3d buffer
        width, height = Image.open(batch_images[0]).size
        # remap the sizes for the model (flipped for orientation)
        t_height, t_width = resize(height, width, 224)
        # first we load the batch of images that we plan to push into the model
        batch = []
        for f in batch_images[:2]:
            img = Image.open(f).convert("RGB").resize(size=(t_width, t_height))
            batch.append(img)

        img = batch[0]
        img_path = batch_images[0]
        encoding = processor(images=img, return_tensors="pt", truncation=True)

        for k, v in encoding.items():
            log.info(f"{k}={v.shape}")

        # TODO: The plotting and resampling needs to be flushed out
        image = encoding.get("image")
        if save and isinstance(image, torch.Tensor):
            img = to_img(image.squeeze())
            loc = result_path / f"{img_path.stem}.JPEG"
            img.save(loc, "JPEG")


if __name__ == "__main__":
    main(False)
