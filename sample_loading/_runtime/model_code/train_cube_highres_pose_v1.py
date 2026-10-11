#!/usr/bin/env python3
"""Train a high-resolution, image-only initial tube-pose estimator.

The model deliberately excludes robot state so the estimated tube cannot
follow a commanded arm trajectory after a missed grasp. Labels are offline
only; runtime input is the declared cam_high RGB stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--keypoints", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def cube_target(row: dict[str, Any]) -> np.ndarray:
    cube = np.asarray(row["cube_pose"], dtype=np.float32)
    return np.concatenate((cube[:3, 3], cube[:3, 2])).astype(np.float32)


class CubeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        target_mean: np.ndarray,
        target_std: np.ndarray,
        image_size: tuple[int, int],
        augment: bool,
    ):
        self.rows = rows
        self.target_mean = torch.from_numpy(target_mean)
        self.target_std = torch.from_numpy(target_std)
        ops: list[Any] = [transforms.Resize(image_size, antialias=True)]
        if augment:
            ops.append(
                transforms.ColorJitter(
                    brightness=(0.80, 1.20),
                    contrast=(0.80, 1.20),
                    saturation=(0.85, 1.15),
                    hue=(-0.02, 0.02),
                )
            )
        ops.extend(
            (
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
        )
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        with Image.open(row["images"]["cam_high"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        raw = torch.from_numpy(cube_target(row))
        return {
            "image": image_tensor,
            "target": (raw - self.target_mean) / self.target_std,
            "target_raw": raw,
        }


class CubeSpatialEncoder(nn.Module):
    def __init__(self, keypoints: int):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.appearance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.local_projection = nn.Conv2d(256, 48, kernel_size=1)
        self.heatmaps = nn.Conv2d(256, keypoints, kernel_size=1)
        nn.init.normal_(self.heatmaps.weight, std=0.001)
        nn.init.zeros_(self.heatmaps.bias)
        self.keypoints = keypoints
        self.output_dim = 128 + keypoints * (4 + 48)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        appearance = self.appearance(features)
        logits = self.heatmaps(features)
        batch, keypoints, height, width = logits.shape
        probabilities = torch.softmax(logits.flatten(2) / 0.15, dim=-1)
        probabilities_2d = probabilities.reshape(batch, keypoints, height, width)
        ys = torch.linspace(
            -1.0, 1.0, height, device=image.device, dtype=features.dtype
        )
        xs = torch.linspace(
            -1.0, 1.0, width, device=image.device, dtype=features.dtype
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        mean_x = (probabilities_2d * grid_x).sum(dim=(-2, -1))
        mean_y = (probabilities_2d * grid_y).sum(dim=(-2, -1))
        var_x = (
            probabilities_2d * (grid_x - mean_x[..., None, None]).square()
        ).sum(dim=(-2, -1))
        var_y = (
            probabilities_2d * (grid_y - mean_y[..., None, None]).square()
        ).sum(dim=(-2, -1))
        local = self.local_projection(features)
        local = torch.einsum("bkhw,bchw->bkc", probabilities_2d, local)
        return torch.cat(
            (appearance, mean_x, mean_y, var_x, var_y, local.flatten(1)),
            dim=-1,
        )


class CubeHighPoseNet(nn.Module):
    def __init__(self, keypoints: int = 20):
        super().__init__()
        self.encoder = CubeSpatialEncoder(keypoints)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 6),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


def angle_error_deg(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = nn.functional.normalize(prediction[:, 3:6], dim=-1)
    truth = nn.functional.normalize(target[:, 3:6], dim=-1)
    cosine = (pred * truth).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    translation = (
        torch.linalg.norm(prediction[:, :3] - target[:, :3], dim=-1) * 1000
    )
    axis = angle_error_deg(prediction, target)
    result: dict[str, Any] = {}
    for name, values in (
        ("cube_translation_mm", translation),
        ("cube_axis_deg", axis),
    ):
        values = values.detach().float().cpu()
        result[name] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p90": float(torch.quantile(values, 0.90)),
            "p95": float(torch.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    result["score"] = (
        result["cube_translation_mm"]["mean"]
        + 0.3 * result["cube_axis_deg"]["mean"]
    )
    return result


def loss_fn(
    output: torch.Tensor,
    target_normalized: torch.Tensor,
    target_raw: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    prediction = output.float() * target_std + target_mean
    base = nn.functional.smooth_l1_loss(output.float(), target_normalized.float())
    translation = torch.linalg.norm(
        prediction[:, :3] - target_raw[:, :3], dim=-1
    ).mean()
    pred_axis = nn.functional.normalize(prediction[:, 3:6], dim=-1)
    true_axis = nn.functional.normalize(target_raw[:, 3:6], dim=-1)
    axis = (1.0 - (pred_axis * true_axis).sum(dim=-1)).mean()
    return base + 3.0 * translation / 0.01 + 0.5 * axis


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses: list[float] = []
    predictions = []
    targets = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_norm = batch["target"].to(device, non_blocking=True)
        target_raw = batch["target_raw"].to(device, non_blocking=True)
        output = model(image)
        losses.append(
            float(loss_fn(output, target_norm, target_raw, target_mean, target_std))
        )
        predictions.append((output.float() * target_std + target_mean).cpu())
        targets.append(batch["target_raw"])
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return float(np.mean(losses)), metrics(prediction, target)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    index_path = args.dataset_dir / "scenes.jsonl"
    rows = []
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row.get("source") == "official_random_initial_reset_2000"
                and int(row.get("replica", 0)) == 0
            ):
                rows.append(row)
    splits = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "validation", "holdout")
    }
    if any(not value for value in splits.values()):
        raise RuntimeError({name: len(value) for name, value in splits.items()})

    train_targets = np.stack([cube_target(row) for row in splits["train"]])
    target_mean = train_targets.mean(axis=0).astype(np.float32)
    target_std = train_targets.std(axis=0).astype(np.float32)
    target_std = np.maximum(target_std, 1e-4)

    datasets = {
        name: CubeDataset(
            subset,
            target_mean,
            target_std,
            (args.height, args.width),
            augment=name == "train",
        )
        for name, subset in splits.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }

    device = torch.device(args.device)
    model = CubeHighPoseNet(args.keypoints).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mean_tensor = torch.from_numpy(target_mean).to(device)
    std_tensor = torch.from_numpy(target_std).to(device)

    run_config = {
        "schema_version": 1,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_index_sha256": sha256_file(index_path),
        "source_filter": "official_random_initial_reset_2000 replica=0",
        "split_counts": {name: len(value) for name, value in splits.items()},
        "runtime_inputs": ["cam_high"],
        "offline_labels_only": ["cube_pose"],
        "image_size": [args.height, args.width],
        "keypoints": args.keypoints,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
    }
    atomic_json(args.output_dir / "run_config.json", run_config)
    np.savez(
        args.output_dir / "normalization.npz",
        target_mean=target_mean,
        target_std=target_std,
    )

    best_score = float("inf")
    best_epoch = 0
    history_path = args.output_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            image = batch["image"].to(device, non_blocking=True)
            target_norm = batch["target"].to(device, non_blocking=True)
            target_raw = batch["target_raw"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(image)
                loss = loss_fn(
                    output, target_norm, target_raw, mean_tensor, std_tensor
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
        scheduler.step()

        validation_loss, validation_metrics = evaluate(
            model, loaders["validation"], device, mean_tensor, std_tensor
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)

        checkpoint = {
            "schema_version": 1,
            "architecture": "CubeHighPoseNet",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "image_size": (args.height, args.width),
            "keypoints": args.keypoints,
            "run_config": run_config,
            "validation_metrics": validation_metrics,
        }
        atomic_torch_save(args.output_dir / "latest.ckpt", checkpoint)
        if validation_metrics["score"] < best_score:
            best_score = float(validation_metrics["score"])
            best_epoch = epoch
            atomic_torch_save(args.output_dir / "best.ckpt", checkpoint)

    best = torch.load(
        args.output_dir / "best.ckpt", map_location=device, weights_only=False
    )
    model.load_state_dict(best["model_state"])
    evaluation = {
        "schema_version": 1,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "metrics": {},
    }
    for name in ("validation", "holdout"):
        split_loss, split_metrics = evaluate(
            model, loaders[name], device, mean_tensor, std_tensor
        )
        evaluation["metrics"][name] = {
            "loss": split_loss,
            **split_metrics,
        }
    evaluation["best_checkpoint_sha256"] = sha256_file(
        args.output_dir / "best.ckpt"
    )
    evaluation["normalization_sha256"] = sha256_file(
        args.output_dir / "normalization.npz"
    )
    atomic_json(args.output_dir / "evaluation.json", evaluation)
    print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main()
