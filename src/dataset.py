from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ResizeBoxes:
    """Keep aspect ratio, resize shorter side to short_size, rescale boxes."""
    def __init__(self, short_size: int = 1024, max_size: int = 2000):
        self.short = short_size
        self.max = max_size

    def __call__(self, img: Image.Image, target: dict):
        w, h = img.size
        short, long = (h, w) if h < w else (w, h)
        scale = min(self.short / short, self.max / long)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= (new_w / w)
            boxes[:, [1, 3]] *= (new_h / h)
            target["boxes"] = boxes

        target["orig_size"] = torch.tensor([h, w], dtype=torch.int64)
        target["size"] = torch.tensor([new_h, new_w], dtype=torch.int64)
        return img, target

class ToTensor:
    def __call__(self, img: Image.Image, target: dict):
        arr = np.array(img, copy=False)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return tensor, target

class ComposeDet:
    def __init__(self, ops: List[Callable]):
        self.ops = ops
    def __call__(self, img, target):
        for op in self.ops:
            img, target = op(img, target)
        return img, target

def default_transforms():
    return ComposeDet([ResizeBoxes(1024), ToTensor()])

class TitleBlockDataset(Dataset):
    """
    COCO-style dataset. Maps JSON category_id (0,1,...) -> labels (1,2,...) so 0 remains background.
    Resolves image paths by basename under img_dir, ignoring Label Studio prefixes.
    """

    def __init__(
        self,
        anno_file: Path,
        img_dir: Path,
        transforms: Optional[Callable] = None,
        keep_only_category_id: int | None = None,  # e.g., 0 to keep only "titleblock"
    ) -> None:
        self.img_dir = Path(img_dir)
        self.transforms = transforms or default_transforms()

        with open(anno_file, "r") as f:
            coco = json.load(f)

        self.images = coco.get("images", [])
        self.annotations = coco.get("annotations", [])
        self.categories = coco.get("categories", [])

        # group annotations by image
        self.by_img: Dict[int, List[dict]] = {}
        for a in self.annotations:
            if keep_only_category_id is not None and a["category_id"] != keep_only_category_id:
                continue
            self.by_img.setdefault(a["image_id"], []).append(a)

        self.ids = [img["id"] for img in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def _resolve_path(self, file_name: str) -> Path:
        # Use only the basename from Label Studio-style paths
        return self.img_dir / Path(file_name).name

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        info = self.images[idx]
        img_id = info["id"]
        path = self._resolve_path(info["file_name"])
        img = Image.open(path).convert("RGB")

        anns = self.by_img.get(img_id, [])
        boxes, labels, area, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            # +1 so labels start at 1 (0 is background)
            labels.append(int(a["category_id"]) + 1)
            area.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area_t = torch.tensor(area, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id]),
            "area": area_t,
            "iscrowd": iscrowd_t,
        }

        img, target = self.transforms(img, target)
        return img, target

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)
