import json
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class TitleBlockDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        anno_file: Path,
        transforms: Callable | None = None,
    ) -> None:
        """
        Args:
            img_dir (str): Path to the directory where images are stored.
            anno_file (str): Path to the COCO format JSON annotation file.
            transforms (callable, optional): A pipeline.
        """
        self.img_dir = img_dir
        self.transforms = transforms

        coco_data = None
        with open(anno_file) as f:
            coco_data = json.load(f)

        if coco_data is None:
            raise ValueError("Annotation file was not found")

        self.images = coco_data["images"]
        self.annotations = coco_data["annotations"]

        self.img_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[image_id] = []

            self.img_id_to_annotations[image_id].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> tuple[torch.Tensor, dict]:
        """
        Retrieves an image and its corresponding annotations.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            A tuple containing:
            - image (torch.Tensor): The image as a tensor.
            - target (dict): A dictionary containing 'boxes' and 'labels'.
        """
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_name = img_info["file_name"]
        img_path = self.img_dir / img_name
        img = Image.open(img_path).convert("RGB")
        annos = self.img_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in annos:
            # COCO format for bbox is [x_min, y_min, width, height]
            x_min, y_min, width, height = ann["bbox"]
            # Torchvision models expect [x_min, y_min, x_max, y_max]
            x_max = x_min + width
            y_max = y_min + height

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # Throwing this in since gemini claims its useful metrics like COCOeval
        target["image_id"] = torch.tensor([img_id])

        if self.transforms:
            # Add in boxes and labels ?
            image = self.transforms(img)

        return image, target
