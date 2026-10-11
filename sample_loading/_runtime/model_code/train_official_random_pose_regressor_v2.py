#!/usr/bin/env python3
"""Train a spatial multi-view pose regressor on independent official resets."""
from __future__ import annotations

import argparse
import csv
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

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--keypoints", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
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


def pose_target(row: dict[str, Any]) -> np.ndarray:
    cube = np.asarray(row["cube_pose"], dtype=np.float32)
    rack = np.asarray(row["rack_pose"], dtype=np.float32)
    if cube.shape != (4, 4) or rack.shape != (4, 4):
        raise ValueError("Expected 4x4 cube and rack poses")
    cube_axis = cube[:3, 2].copy()
    cube_axis /= max(float(np.linalg.norm(cube_axis)), 1e-8)
    rack_sixd = np.concatenate((rack[:3, 0], rack[:3, 1]))
    return np.concatenate(
        (cube[:3, 3], cube_axis, rack[:3, 3], rack_sixd)
    ).astype(np.float32)


class PoseDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, Any]],
        target_mean: np.ndarray,
        target_std: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        image_size: tuple[int, int],
        augment: bool,
    ):
        self.root = root
        self.rows = rows
        self.target_mean = torch.from_numpy(target_mean)
        self.target_std = torch.from_numpy(target_std)
        self.state_mean = torch.from_numpy(state_mean)
        self.state_std = torch.from_numpy(state_std)
        ops: list[Any] = [transforms.Resize(image_size, antialias=True)]
        if augment:
            ops.append(
                transforms.ColorJitter(
                    brightness=(0.90, 1.10),
                    contrast=(0.90, 1.10),
                    saturation=(0.95, 1.05),
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
        images = []
        for camera in CAMERAS:
            with Image.open(self.root / row["images"][camera]) as image:
                images.append(self.transform(image.convert("RGB")))
        target_raw = torch.from_numpy(pose_target(row))
        state_raw = torch.tensor(row["robot_qpos"], dtype=torch.float32)
        if state_raw.shape != (14,):
            raise ValueError(f"Expected qpos shape 14, got {tuple(state_raw.shape)}")
        return {
            "images": torch.stack(images),
            "state": (state_raw - self.state_mean) / self.state_std,
            "target": (target_raw - self.target_mean) / self.target_std,
            "target_raw": target_raw,
            "scene_index": torch.tensor(row["scene_index"], dtype=torch.int64),
        }


class SpatialViewEncoder(nn.Module):
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
        self.local_projection = nn.Conv2d(256, 32, kernel_size=1)
        self.heatmaps = nn.Conv2d(256, keypoints, kernel_size=1)
        nn.init.normal_(self.heatmaps.weight, std=0.001)
        nn.init.zeros_(self.heatmaps.bias)
        self.keypoints = keypoints
        self.output_dim = 128 + keypoints * (2 + 2 + 32)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        appearance = self.appearance(features)
        logits = self.heatmaps(features)
        batch, keypoints, height, width = logits.shape
        probabilities = torch.softmax(logits.flatten(2) / 0.20, dim=-1)
        probabilities_2d = probabilities.reshape(batch, keypoints, height, width)
        ys = torch.linspace(
            -1.0, 1.0, height, device=images.device, dtype=features.dtype
        )
        xs = torch.linspace(
            -1.0, 1.0, width, device=images.device, dtype=features.dtype
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


class MultiViewSpatialPoseNet(nn.Module):
    def __init__(self, keypoints: int = 12, target_dim: int = 15):
        super().__init__()
        self.view_encoder = SpatialViewEncoder(keypoints)
        self.state_encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        fused_dim = len(CAMERAS) * self.view_encoder.output_dim + 64
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(384, target_dim),
        )

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = images.shape
        encoded = self.view_encoder(
            images.reshape(batch * views, channels, height, width)
        )
        encoded = encoded.reshape(batch, views * encoded.shape[-1])
        return self.head(torch.cat((encoded, self.state_encoder(state)), dim=-1))


def sixd_to_rotation(value: torch.Tensor) -> torch.Tensor:
    first = nn.functional.normalize(value[..., 0:3], dim=-1)
    second_raw = value[..., 3:6]
    second = nn.functional.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def angle_deg(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = nn.functional.normalize(first, dim=-1)
    second = nn.functional.normalize(second, dim=-1)
    cosine = (first * second).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def rotation_angle_rad(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = first.transpose(-1, -2) @ second
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) / 2.0
    return torch.acos(cosine.clamp(-1.0, 1.0))


def metrics_from_tensors(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    values = {
        "cube_translation_mm": torch.linalg.norm(
            prediction[:, 0:3] - target[:, 0:3], dim=-1
        ) * 1000,
        "cube_axis_deg": angle_deg(prediction[:, 3:6], target[:, 3:6]),
        "rack_translation_mm": torch.linalg.norm(
            prediction[:, 6:9] - target[:, 6:9], dim=-1
        ) * 1000,
        "rack_rotation_deg": torch.rad2deg(
            rotation_angle_rad(
                sixd_to_rotation(prediction[:, 9:15]),
                sixd_to_rotation(target[:, 9:15]),
            )
        ),
    }
    result: dict[str, Any] = {}
    for name, tensor in values.items():
        tensor = tensor.float().cpu()
        result[name] = {
            "mean": float(tensor.mean()),
            "median": float(tensor.median()),
            "p90": float(torch.quantile(tensor, 0.90)),
            "p95": float(torch.quantile(tensor, 0.95)),
            "max": float(tensor.max()),
        }
    result["score"] = (
        result["cube_translation_mm"]["mean"]
        + result["rack_translation_mm"]["mean"]
        + 0.5 * result["cube_axis_deg"]["mean"]
        + 0.5 * result["rack_rotation_deg"]["mean"]
    )
    return result


def training_loss(
    output_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    target_raw: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = output_normalized.float() * target_std + target_mean
    base = nn.functional.smooth_l1_loss(
        output_normalized.float(), target_normalized.float()
    )
    cube_translation = torch.linalg.norm(
        prediction[:, 0:3] - target_raw[:, 0:3], dim=-1
    ).mean() / 0.05
    rack_translation = torch.linalg.norm(
        prediction[:, 6:9] - target_raw[:, 6:9], dim=-1
    ).mean() / 0.05
    cube_axis = 1.0 - (
        nn.functional.normalize(prediction[:, 3:6], dim=-1)
        * nn.functional.normalize(target_raw[:, 3:6], dim=-1)
    ).sum(dim=-1).mean()
    rack_rotation = rotation_angle_rad(
        sixd_to_rotation(prediction[:, 9:15]),
        sixd_to_rotation(target_raw[:, 9:15]),
    ).mean() / math.pi
    total = (
        base
        + 0.50 * cube_translation
        + 0.50 * rack_translation
        + 0.20 * cube_axis
        + 0.20 * rack_rotation
    )
    return total, {
        "base": float(base.detach()),
        "cube_translation": float(cube_translation.detach()),
        "rack_translation": float(rack_translation.detach()),
        "cube_axis": float(cube_axis.detach()),
        "rack_rotation": float(rack_rotation.detach()),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[float, dict]:
    model.eval()
    losses: list[float] = []
    predictions = []
    targets = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        target_normalized = batch["target"].to(device, non_blocking=True)
        target_raw = batch["target_raw"].to(device, non_blocking=True)
        output = model(images, state)
        loss, _ = training_loss(
            output, target_normalized, target_raw, target_mean, target_std
        )
        losses.append(float(loss))
        predictions.append((output.float() * target_std + target_mean).cpu())
        targets.append(batch["target_raw"])
    return float(np.mean(losses)), metrics_from_tensors(
        torch.cat(predictions), torch.cat(targets)
    )


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"Refusing non-empty output without --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = [
        json.loads(line)
        for line in (dataset_dir / "scenes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    expected = json.loads(
        (dataset_dir / "collection_config.json").read_text(encoding="utf-8")
    )["num_scenes"]
    if len(rows) != expected:
        raise RuntimeError(f"Dataset incomplete: {len(rows)} of {expected} scenes")
    if len({int(row["scene_index"]) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate scene_index detected")
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "holdout")
    }
    if any(not split_rows for split_rows in by_split.values()):
        raise RuntimeError(
            f"Empty split: {dict((k, len(v)) for k, v in by_split.items())}"
        )

    train_targets = np.stack([pose_target(row) for row in by_split["train"]])
    train_states = np.stack(
        [np.asarray(row["robot_qpos"], dtype=np.float32) for row in by_split["train"]]
    )
    target_mean = train_targets.mean(axis=0).astype(np.float32)
    target_std = train_targets.std(axis=0).astype(np.float32)
    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = train_states.std(axis=0).astype(np.float32)
    target_std[target_std < 1e-4] = 1.0
    state_std[state_std < 1e-4] = 1.0
    np.savez(
        output_dir / "normalization.npz",
        target_mean=target_mean,
        target_std=target_std,
        state_mean=state_mean,
        state_std=state_std,
    )

    run_config = {
        "schema_version": 1,
        "dataset_dir": str(dataset_dir),
        "dataset_index_sha256": sha256_file(dataset_dir / "scenes.jsonl"),
        "collection_config_sha256": sha256_file(
            dataset_dir / "collection_config.json"
        ),
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
        "image_size": [args.height, args.width],
        "keypoints": args.keypoints,
        "runtime_inputs": [
            "cam_high", "cam_left_wrist", "cam_right_wrist", "joints", "gripper"
        ],
        "offline_labels_only": ["cube_pose", "rack_pose"],
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise RuntimeError("Resume run_config mismatch")
    else:
        atomic_json(config_path, run_config)

    datasets = {
        split: PoseDataset(
            dataset_dir, split_rows, target_mean, target_std,
            state_mean, state_std, (args.height, args.width),
            augment=split == "train",
        )
        for split, split_rows in by_split.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }

    device = torch.device(args.device)
    model = MultiViewSpatialPoseNet(args.keypoints).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.03
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mean_tensor = torch.from_numpy(target_mean).to(device)
    std_tensor = torch.from_numpy(target_std).to(device)
    start_epoch = 1
    best_score = math.inf
    best_epoch = -1
    latest_path = output_dir / "latest.ckpt"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        print(f"Resuming at epoch {start_epoch}", flush=True)

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        component_values: dict[str, list[float]] = {}
        for batch in loaders["train"]:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target_normalized = batch["target"].to(device, non_blocking=True)
            target_raw = batch["target_raw"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(images, state)
            loss, parts = training_loss(
                output, target_normalized, target_raw, mean_tensor, std_tensor
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
            for name, value in parts.items():
                component_values.setdefault(name, []).append(value)
        scheduler.step()

        validation_loss, validation_metrics = evaluate(
            model, loaders["validation"], device, mean_tensor, std_tensor
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "validation_metrics": validation_metrics,
            "train_loss_components": {
                k: float(np.mean(v)) for k, v in component_values.items()
            },
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)

        score = float(validation_metrics["score"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            atomic_torch_save(
                output_dir / "best.ckpt",
                {
                    "schema_version": 2,
                    "architecture": "MultiViewSpatialPoseNet",
                    "model_state": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "image_size": [args.height, args.width],
                    "keypoints": args.keypoints,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "runtime_inputs": run_config["runtime_inputs"],
                    "offline_labels_only": run_config["offline_labels_only"],
                },
            )
        atomic_torch_save(
            latest_path,
            {
                "schema_version": 2,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
            },
        )
        if epoch % 10 == 0:
            atomic_torch_save(
                output_dir / f"epoch_{epoch:03d}.ckpt",
                {
                    "schema_version": 2,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
            )

    checkpoint = torch.load(
        output_dir / "best.ckpt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = {}
    for split in ("validation", "holdout"):
        loss, metrics = evaluate(
            model, loaders[split], device, mean_tensor, std_tensor
        )
        final_metrics[split] = {"loss": loss, **metrics}
    result = {
        "schema_version": 2,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "metrics": final_metrics,
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "best_checkpoint_sha256": sha256_file(output_dir / "best.ckpt"),
        "normalization_sha256": sha256_file(output_dir / "normalization.npz"),
    }
    atomic_json(output_dir / "evaluation.json", result)
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    flat = [
        {
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "validation_loss": row["validation_loss"],
            "lr": row["lr"],
            "val_score": row["validation_metrics"]["score"],
            "cube_translation_mm": row["validation_metrics"][
                "cube_translation_mm"
            ]["mean"],
            "rack_translation_mm": row["validation_metrics"][
                "rack_translation_mm"
            ]["mean"],
            "cube_axis_deg": row["validation_metrics"]["cube_axis_deg"]["mean"],
            "rack_rotation_deg": row["validation_metrics"][
                "rack_rotation_deg"
            ]["mean"],
        }
        for row in records
    ]
    with (output_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
