from pathlib import Path

import torch
import torchvision
from loguru import logger as log
from PIL import Image
from torchvision.models.detection.faster_rcnn import FasterRCNN

from dataset import TitleBlockDataset
from helpers import plot_results, resize
from utils import get_assets_path, get_sample_batch, get_tests_path

torch.manual_seed(0)

ROOT = get_assets_path() / "coco"
IMAGES = str(ROOT / "images")
LABELS = str(ROOT / "instances.json")

width: int = 0
height: int = 0


def use_layoutlmv2(save: bool, print_logs: bool) -> None:
    from transformers import (
        LayoutLMv2ImageProcessor,
        LayoutLMv2Processor,
        LayoutLMv2Tokenizer,
    )

    batch = get_sample_batch()
    log.info(f"Starting Processor for {len(batch.keys())} docs")
    result_path = get_tests_path()
    # each source is a single pdf doc set
    image_processor = LayoutLMv2ImageProcessor()
    tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(image_processor, tokenizer)

    for _, batch_images in batch.items():
        # here we only need the first img to get this data from, since we are mounting all of this into a 3d buffer
        width, height = Image.open(batch_images[0]).size
        # remap the sizes for the model (flipped for orientation)
        t_height, t_width = resize(height, width, 224)
        # first we load the batch of images that we plan t o push into the model
        batch = []
        og_batch = []
        for f in batch_images[:2]:
            og_img = Image.open(f).convert("RGB")
            resized_img = og_img.resize(size=(t_width, t_height))
            batch.append(resized_img)
            og_batch.append(og_img)

        # for now we grab the first image to test
        og_img = og_batch[0]
        og_path = batch_images[0]
        encoding = processor(images=og_img, return_tensors="pt", truncation=True)

        if print_logs:
            for k, v in encoding.items():
                log.info(f"{k}={v.shape}")

        if save:
            log.info("Plotting New Image")
            img = plot_results(og_img, encoding, (width, height), tokenizer)
            file_path = result_path / f"{og_path.stem}.jpeg"
            img.save(file_path, "jpeg")


def use_fasterrcnn(
    features: list[str],
) -> None:
    log.info("Starting Faster RCNN Model")
    mobilenet = torchvision.models.mobilenet_v3_large(weights="DEFAULT")
    # remove the mapping channel as we'll setup our own
    backbone = mobilenet.features
    backbone.out_channels = 960
    model = FasterRCNN(backbone, num_classes=len(features))


def get_coco_path() -> Path:
    return get_assets_path() / "coco"


def get_coco_file(c: str = "single") -> Path:
    return get_coco_path() / f"{c}-set/result.json"


if __name__ == "__main__":
    from torchvision import transforms as T

    data_transforms = T.Compose([T.ToTensor()])

    coco = get_coco_file(c="single")

    ds = TitleBlockDataset(
        anno_file=coco, img_dir=coco.parent / "images", transforms=data_transforms
    )
    # example using the dataset
    from torch.utils.data import DataLoader

    loader = DataLoader(ds)
    log.info(loader)
    # features = ["background", "titleblock", "viewport"]
    # use_fasterrcnn(features)
    # use_layoutlmv2(save=True, print_logs=True)
