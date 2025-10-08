from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TitleBlockDataset, collate_fn, default_transforms
from models import build_frcnn_mobilenet
from utils import seed_everything, save_checkpoint, eval_ap50

def train_one_epoch(model, loader, optimizer, device, epoch: int, log_every: int = 50):
    model.train()
    total = 0.0
    for i, (imgs, targets) in enumerate(loader, 1):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(imgs, targets)
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
        if i % log_every == 0:
            print(f"[epoch {epoch:02d} step {i:04d}] loss={loss.item():.4f}")
    return total / max(1, len(loader))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", type=Path, required=True)
    p.add_argument("--val_json", type=Path, required=True)
    p.add_argument("--img_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--titleblock_only", action="store_true",
                   help="Keep only category_id==0 (titleblock) and drop others.")
    args = p.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keep_cat = 0 if args.titleblock_only else None
    train_ds = TitleBlockDataset(args.train_json, args.img_dir,
                                 transforms=default_transforms(),
                                 keep_only_category_id=keep_cat)
    val_ds   = TitleBlockDataset(args.val_json,   args.img_dir,
                                 transforms=default_transforms(),
                                 keep_only_category_id=keep_cat)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    # num_classes includes background
    num_classes = 2 if args.titleblock_only else 3
    model = build_frcnn_mobilenet(num_classes=num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[8, 10], gamma=0.1)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device, epoch)
        ap50 = eval_ap50(model, val_loader, device)
        print(f"epoch {epoch:02d} | loss={tr_loss:.4f} | AP50={ap50:.4f}")
        sched.step()
        save_checkpoint(model, optim, epoch, args.outdir)
        if ap50 > best:
            best = ap50
            (args.outdir / "best_model.pt").parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.outdir / "best_model.pt")
            print(f"✓ new best AP50={best:.4f} (saved best_model.pt)")

if __name__ == "__main__":
    main()
