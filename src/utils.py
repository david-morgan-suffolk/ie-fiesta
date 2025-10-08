from __future__ import annotations
import random
from pathlib import Path
from typing import List

import torch
from torchvision.ops import box_iou

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"model_epoch{epoch}.pt"
    torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, path)
    return path

@torch.no_grad()
def ap50_single_image(pred_boxes, pred_scores, gt_boxes, score_thresh=0.05, iou_thresh=0.5):
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 1.0
    if pred_boxes.numel() == 0 and gt_boxes.numel() > 0:
        return 0.0
    keep = pred_scores >= score_thresh
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    if pred_boxes.numel() == 0:
        return 0.0

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]

    ious = box_iou(pred_boxes, gt_boxes) if gt_boxes.numel() > 0 else torch.zeros((len(pred_boxes), 0))
    matched_gt = set()
    tp = fp = 0
    for i in range(len(pred_boxes)):
        if gt_boxes.numel() == 0:
            fp += 1; continue
        j = torch.argmax(ious[i]).item()
        if ious[i, j] >= iou_thresh and j not in matched_gt:
            tp += 1; matched_gt.add(j)
        else:
            fp += 1
    fn = max(0, gt_boxes.shape[0] - len(matched_gt))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (precision + recall) / 2.0  # quick proxy

@torch.no_grad()
def eval_ap50(model, dataloader, device):
    model.eval()
    scores: List[float] = []
    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        preds = model(imgs)
        for p, t in zip(preds, targets):
            pb = p["boxes"].cpu()
            ps = p["scores"].cpu()
            gb = t["boxes"].cpu()
            scores.append(ap50_single_image(pb, ps, gb))
    return float(torch.tensor(scores).mean().item()) if scores else 0.0
