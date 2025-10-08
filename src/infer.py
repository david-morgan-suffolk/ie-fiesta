from __future__ import annotations
import json
from pathlib import Path

import torch
from PIL import Image

from dataset import default_transforms
from models import build_frcnn_mobilenet

@torch.no_grad()
def load_model(weights: Path, num_classes: int, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_frcnn_mobilenet(num_classes=num_classes).to(device)
    state = torch.load(weights, map_location=device)
    if "model" in state:  # in case you pass a checkpoint dict
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model, device

def infer_folder(model, device, img_dir: Path, out_json: Path, score_thr: float = 0.5):
    t = default_transforms()
    out = []
    for p in sorted(Path(img_dir).glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = Image.open(p).convert("RGB")
        x, _ = t(img, {})
        pred = model([x.to(device)])[0]
        keep = pred["scores"] >= score_thr
        boxes = pred["boxes"][keep].cpu()
        scores = pred["scores"][keep].cpu().tolist()

        H, W = x.shape[1], x.shape[2]
        xyxy = boxes.clone()
        xyxy[:, [0, 2]] /= W
        xyxy[:, [1, 3]] /= H

        out.append({"file_name": p.name,
                    "boxes_xyxy_norm": xyxy.tolist(),
                    "scores": scores})
    out_json.write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--img_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_classes", type=int, default=3)  # 2 if --titleblock_only during training
    ap.add_argument("--score_thr", type=float, default=0.5)
    args = ap.parse_args()

    model, device = load_model(args.weights, num_classes=args.num_classes)
    infer_folder(model, device, args.img_dir, args.out, args.score_thr)
    print(f"Wrote {args.out}")
