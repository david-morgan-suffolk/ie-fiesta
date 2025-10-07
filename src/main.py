import math

import torch
import torchvision
from loguru import logger as log
from PIL import Image
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.feature_extraction import (
    get_graph_node_names,
)

from helpers import plot_results
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

    for batch_source, batch_images in batch.items():
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


def use_fasterrcnn(feature_count: int = 3) -> None:
    log.info("Starting Faster RCNN Model")
    mobilenet = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    )
    backbone = mobilenet.features
    backbone.out_channels = 960  # not really sure how these layering stuff works here

    model = FasterRCNN(backbone, num_classes=feature_count)
    train_nodes, eval_nodes = get_graph_node_names(model)
    log.info(f"Checking Nodes\nTraining Nodes:{train_nodes}\nEval Nodes:{eval_nodes}")

    return
    model.eval()
    log.info("\nTesting model with mock data...")
    dummy_input = [torch.rand(3, 800, 600)]

    predictions = None
    with torch.no_grad():
        predictions = model(dummy_input)

    log.info(f"Mock test complete\n{predictions}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total trainable parameters: {total_params:,}")

    batch = get_sample_batch()
    log.info(f"Batch size of {len(batch.keys())} docs")
    result_path = get_tests_path()


if __name__ == "__main__":
    use_fasterrcnn(feature_count=3)

    # use_layoutlmv2(save=True, print_logs=True)
