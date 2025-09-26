import matplotlib.pyplot as plt
import torch
from loguru import logger as log
from PIL import Image
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection

from utils import get_assets_path, get_sample_images

torch.manual_seed(0)

ROOT = get_assets_path() / "coco"
IMAGES = str(ROOT / "images")
LABELS = str(ROOT / "instances.json")


# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(
        scores.tolist(), labels.tolist(), boxes.tolist(), colors, strict=False
    ):
        ax.add_patch(
            plt.plot.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        text = f"{label}: {score:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


def main():
    log.info("Starting Model")
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )
    files = get_sample_images()

    for file_path in files:
        log.info(f"Adding file\n{file_path}")

        image = Image.open(file_path).convert("RGB")
        width, height = image.size
        image.resize((int(width * 0.5), int(height * 0.5)))
        feature_extractor = DetrFeatureExtractor()
        encoding = feature_extractor(image, return_tensors="pt")
        log.info(encoding["pixel_values"].shape)
        with torch.no_grad():
            outputs = model(**encoding)
            width, height = image.size
            results = feature_extractor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=[(height, width)]
            )[0]
            plot_results(image, results["scores"], results["labels"], results["boxes"])

        return


if __name__ == "__main__":
    main()
