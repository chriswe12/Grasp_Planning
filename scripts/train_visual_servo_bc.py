#!/usr/bin/env python3
"""Train the first camera-frame residual visual-servo behavior-cloning policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment,misc]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.d405_wrist_camera import (  # noqa: E402
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
    D405WristCameraConfig,
)
from grasp_planning.rl.live_observation_randomization import (  # noqa: E402
    LiveObservationRandomizationCfg,
    LiveObservationRandomizer,
)
from grasp_planning.rl.visual_servo_dataset import (  # noqa: E402
    ANGULAR_ACTION_SCALE_RAD_S,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    LINEAR_ACTION_SCALE_M_S,
    EpisodeGroupedBatchSampler,
    LocalityBlockBatchSampler,
    MmapVisualServoFrameDataset,
    VisualServoFrameDataset,
)
from grasp_planning.rl.visual_servo_policy import ResidualVisualServoPolicy  # noqa: E402


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    max_train_episodes: int
    max_validation_episodes: int
    episode_cache_size: int
    num_workers: int
    episode_shuffle_block_size: int
    shared_goal: bool
    amp: bool
    training_cache_dir: str | None
    mmap_shuffle_block_size: int
    live_observation_randomization: bool
    observation_profile: str


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    writer: object | None = None,
    epoch: int = 0,
    phase: str = "train",
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    progress_every_batches: int = 25,
    shared_goal: bool = True,
    live_observation_randomization: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    linear_absolute_sum = 0.0
    angular_absolute_sum = 0.0
    sample_count = 0
    epoch_started_at = time.monotonic()
    batch_count_total = len(loader)
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    live_randomizers: dict[int, LiveObservationRandomizer] = {}
    for batch_index, batch in enumerate(loader):
        if "live_rgb" in batch:
            live_rgb = (
                batch["live_rgb"]
                .to(device=device, non_blocking=True)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            live_depth = (
                batch["live_depth"]
                .to(device=device, non_blocking=True)
                .unsqueeze(1)
                .sub_(DEPTH_MIN_M)
                .div_(DEPTH_MAX_M - DEPTH_MIN_M)
                .clamp_(0.0, 1.0)
            )
            goal_selection = slice(0, 1) if shared_goal else slice(None)
            goal_rgb = (
                batch["goal_rgb"][goal_selection]
                .to(device=device, non_blocking=True)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            goal_depth = (
                batch["goal_depth"][goal_selection]
                .to(device=device, non_blocking=True)
                .unsqueeze(1)
                .sub_(DEPTH_MIN_M)
                .div_(DEPTH_MAX_M - DEPTH_MIN_M)
                .clamp_(0.0, 1.0)
            )
            inputs = {
                "live_rgbd": torch.cat((live_rgb, live_depth), dim=1),
                "goal_rgbd": torch.cat((goal_rgb, goal_depth), dim=1),
                **{
                    name: batch[name].to(device=device, non_blocking=True)
                    for name in (
                        "joint_positions",
                        "progress",
                        "nominal_twist_camera",
                    )
                },
            }
        else:
            inputs = {
                name: batch[name].to(device=device, non_blocking=True)
                for name in (
                    "live_rgbd",
                    "goal_rgbd",
                    "joint_positions",
                    "progress",
                    "nominal_twist_camera",
                )
            }
        observation_size = (
            VISUAL_SERVO_OBSERVATION_HEIGHT,
            VISUAL_SERVO_OBSERVATION_WIDTH,
        )
        for image_name in ("live_rgbd", "goal_rgbd"):
            if tuple(inputs[image_name].shape[-2:]) != observation_size:
                inputs[image_name] = F.interpolate(
                    inputs[image_name], size=observation_size, mode="area"
                )
        if training and live_observation_randomization:
            live_rgbd = inputs["live_rgbd"]
            batch_size = int(live_rgbd.shape[0])
            randomizer = live_randomizers.get(batch_size)
            if randomizer is None:
                randomizer = LiveObservationRandomizer(
                    LiveObservationRandomizationCfg(),
                    num_envs=batch_size,
                    device=device,
                )
                live_randomizers[batch_size] = randomizer
            else:
                randomizer.sample(torch.arange(batch_size, device=device))
            live_rgb = live_rgbd[:, :3].permute(0, 2, 3, 1)
            live_depth_m = (
                live_rgbd[:, 3:4]
                .permute(0, 2, 3, 1)
                .mul(DEPTH_MAX_M - DEPTH_MIN_M)
                .add(DEPTH_MIN_M)
            )
            live_rgb, live_depth_m = randomizer.apply(live_rgb, live_depth_m)
            live_depth = (
                live_depth_m.sub(DEPTH_MIN_M)
                .div(DEPTH_MAX_M - DEPTH_MIN_M)
                .clamp(0.0, 1.0)
            )
            inputs["live_rgbd"] = torch.cat(
                (live_rgb, live_depth), dim=-1
            ).permute(0, 3, 1, 2)
        target = batch["residual_twist_camera"].to(device=device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            enabled=amp,
        ):
            prediction = model(**inputs)
            loss = loss_fn(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
        batch_count = int(target.shape[0])
        error = torch.abs(prediction.detach() - target)
        if writer is not None and batch_index == 0:
            writer.add_histogram(
                f"{phase}/predicted_residual_normalized",
                prediction.detach().cpu(),
                epoch,
            )
            writer.add_histogram(
                f"{phase}/expert_residual_normalized",
                target.detach().cpu(),
                epoch,
            )
        loss_sum += float(loss.detach()) * batch_count
        linear_absolute_sum += float(error[:, :3].mean(dim=1).sum())
        angular_absolute_sum += float(error[:, 3:].mean(dim=1).sum())
        sample_count += batch_count
        completed_batches = batch_index + 1
        if (
            completed_batches == 1
            or completed_batches % progress_every_batches == 0
            or completed_batches == batch_count_total
        ):
            elapsed_s = max(time.monotonic() - epoch_started_at, 1.0e-6)
            batches_per_s = completed_batches / elapsed_s
            remaining_s = (
                (batch_count_total - completed_batches) / batches_per_s
                if batches_per_s > 0.0
                else float("inf")
            )
            print(
                f"[{phase.upper()}] epoch={epoch} "
                f"batch={completed_batches}/{batch_count_total} "
                f"frames={sample_count} rate={sample_count / elapsed_s:.1f} frames/s "
                f"eta_min={remaining_s / 60.0:.1f}",
                flush=True,
            )
    return {
        "loss": loss_sum / sample_count,
        "linear_mae_mm_s": (
            linear_absolute_sum / sample_count * LINEAR_ACTION_SCALE_M_S * 1000.0
        ),
        "angular_mae_deg_s": (
            angular_absolute_sum
            / sample_count
            * ANGULAR_ACTION_SCALE_RAD_S
            * 180.0
            / np.pi
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/visual_servo_bc"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-validation-episodes", type=int, default=0)
    parser.add_argument(
        "--episode-cache-size",
        type=int,
        default=2,
        help="Maximum decompressed episodes retained in host RAM per split.",
    )
    parser.add_argument(
        "--training-cache-dir",
        type=Path,
        default=None,
        help="Use a preprocessed contiguous mmap cache instead of episode NPZ files.",
    )
    parser.add_argument(
        "--mmap-shuffle-block-size",
        type=int,
        default=8192,
        help="Contiguous mmap locality window whose frames are shuffled together.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Parallel DataLoader workers used to decompress and prepare episode batches.",
    )
    parser.add_argument(
        "--episode-shuffle-block-size",
        type=int,
        default=64,
        help="Shuffle nearby episode files in blocks to avoid cold random reads across the dataset.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=25,
        help="Print throughput and ETA at this batch interval.",
    )
    parser.add_argument(
        "--shared-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode one goal image per batch; valid when every episode uses the fixed goal.",
    )
    parser.add_argument(
        "--live-observation-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Randomize only live RGB-D during training; goal and validation images "
            "remain canonical."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use automatic mixed precision; defaults to enabled on CUDA.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model, optimizer, scaler, epoch, and best metric from a checkpoint.",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.episode_cache_size < 1
        or args.num_workers < 0
        or args.episode_shuffle_block_size < 1
        or args.mmap_shuffle_block_size < args.batch_size
        or args.progress_every_batches < 1
    ):
        parser.error(
            "--epochs, --batch-size, --episode-cache-size, and "
            "--progress-every-batches must be positive; --num-workers must be nonnegative."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but this Python environment cannot access CUDA. "
            "Run through the Isaac container or choose --device cpu."
        )
    amp_enabled = device.type == "cuda" if args.amp is None else bool(args.amp)
    if amp_enabled and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device.")
    config = TrainingConfig(
        dataset_dir=str(args.dataset_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_train_episodes=args.max_train_episodes,
        max_validation_episodes=args.max_validation_episodes,
        episode_cache_size=args.episode_cache_size,
        num_workers=args.num_workers,
        episode_shuffle_block_size=args.episode_shuffle_block_size,
        shared_goal=bool(args.shared_goal),
        amp=amp_enabled,
        training_cache_dir=(
            str(args.training_cache_dir.resolve())
            if args.training_cache_dir is not None
            else None
        ),
        mmap_shuffle_block_size=args.mmap_shuffle_block_size,
        live_observation_randomization=bool(args.live_observation_randomization),
        observation_profile=D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    )
    print(f"[TRAIN] Loading successful train episodes from {args.dataset_dir}.", flush=True)
    using_mmap_cache = args.training_cache_dir is not None
    if using_mmap_cache:
        if args.max_train_episodes or args.max_validation_episodes:
            parser.error("Episode limits cannot be used with --training-cache-dir.")
        train_dataset = MmapVisualServoFrameDataset(
            args.training_cache_dir, split="train"
        )
        validation_dataset = MmapVisualServoFrameDataset(
            args.training_cache_dir, split="validation"
        )
    else:
        train_dataset = VisualServoFrameDataset(
            args.dataset_dir,
            split="train",
            max_episodes=args.max_train_episodes,
            cache_episodes=args.episode_cache_size,
            raw_images=True,
        )
        validation_dataset = VisualServoFrameDataset(
            args.dataset_dir,
            split="validation",
            max_episodes=args.max_validation_episodes,
            cache_episodes=args.episode_cache_size,
            raw_images=True,
        )
    loader_worker_options = (
        {
            "num_workers": args.num_workers,
            "persistent_workers": True,
            "prefetch_factor": 2,
        }
        if args.num_workers > 0
        else {"num_workers": 0}
    )
    if using_mmap_cache:
        train_batch_sampler = LocalityBlockBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            block_size=args.mmap_shuffle_block_size,
            shuffle=True,
            seed=args.seed,
        )
        validation_batch_sampler = LocalityBlockBatchSampler(
            validation_dataset,
            batch_size=args.batch_size,
            block_size=args.mmap_shuffle_block_size,
            shuffle=False,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=device.type == "cuda",
            **loader_worker_options,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=validation_batch_sampler,
            pin_memory=device.type == "cuda",
            **loader_worker_options,
        )
    else:
        train_batch_sampler = EpisodeGroupedBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            episode_shuffle_block_size=args.episode_shuffle_block_size,
        )
        validation_batch_sampler = EpisodeGroupedBatchSampler(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            episode_shuffle_block_size=args.episode_shuffle_block_size,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=device.type == "cuda",
            **loader_worker_options,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=validation_batch_sampler,
            pin_memory=device.type == "cuda",
            **loader_worker_options,
        )
    model = ResidualVisualServoPolicy().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.output_dir / "tensorboard"
    writer = (
        SummaryWriter(log_dir=str(tensorboard_dir))
        if SummaryWriter is not None
        else None
    )
    if writer is None:
        print(
            "[TRAIN] TensorBoard is unavailable in this Python environment; "
            "continuing with console, history.json, and checkpoint metrics.",
            flush=True,
        )
    preview = train_dataset[0]
    if writer is not None:
        preview_live = VisualServoFrameDataset._rgbd(
            preview["live_rgb"].numpy(),
            preview["live_depth"].numpy(),
        )
        preview_goal = VisualServoFrameDataset._rgbd(
            preview["goal_rgb"].numpy(),
            preview["goal_depth"].numpy(),
        )
        writer.add_image("examples/live_rgb", preview_live[:3], 0)
        writer.add_image("examples/live_depth", preview_live[3:4], 0)
        writer.add_image("examples/goal_rgb", preview_goal[:3], 0)
        writer.add_image("examples/goal_depth", preview_goal[3:4], 0)
        writer.add_text("configuration", json.dumps(asdict(config), indent=2), 0)
    if isinstance(train_dataset, VisualServoFrameDataset):
        train_dataset.clear_cache()
    history_path = args.output_dir / "history.json"
    history = []
    start_epoch = 1
    best_validation_loss = float("inf")
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        checkpoint_dataset_dir = checkpoint.get("training_config", {}).get(
            "dataset_dir"
        )
        if checkpoint_dataset_dir is None:
            raise ValueError(
                f"{args.resume_checkpoint} is a legacy checkpoint without dataset identity; "
                "refusing to resume it automatically against a potentially different dataset."
            )
        if Path(checkpoint_dataset_dir).resolve() != args.dataset_dir.resolve():
            raise ValueError(
                f"Checkpoint dataset mismatch: checkpoint={checkpoint_dataset_dir} "
                f"requested={args.dataset_dir.resolve()}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint.get("metrics", {})
                .get("validation", {})
                .get("loss", float("inf")),
            )
        )
        best_checkpoint_path = args.output_dir / "best.pt"
        if (
            "best_validation_loss" not in checkpoint
            and best_checkpoint_path.exists()
        ):
            best_checkpoint = torch.load(
                best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            best_validation_loss = min(
                best_validation_loss,
                float(
                    best_checkpoint.get(
                        "best_validation_loss",
                        best_checkpoint.get("metrics", {})
                        .get("validation", {})
                        .get("loss", float("inf")),
                    )
                ),
            )
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(
            f"[TRAIN] Resumed {args.resume_checkpoint} at epoch {start_epoch}; "
            f"best_validation_loss={best_validation_loss:.6f}.",
            flush=True,
        )
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}, "
            f"which is not below requested --epochs {args.epochs}."
        )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            phase="train",
            scaler=scaler,
            amp=amp_enabled,
            progress_every_batches=args.progress_every_batches,
            shared_goal=bool(args.shared_goal),
            live_observation_randomization=bool(
                args.live_observation_randomization
            ),
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            writer=writer,
            epoch=epoch,
            phase="validation",
            amp=amp_enabled,
            progress_every_batches=args.progress_every_batches,
            shared_goal=bool(args.shared_goal),
            live_observation_randomization=False,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        for phase, metrics in (("train", train_metrics), ("validation", validation_metrics)):
            for metric_name, value in metrics.items():
                if writer is not None:
                    writer.add_scalar(f"{phase}/{metric_name}", value, epoch)
        if writer is not None:
            writer.add_scalar(
                "optimization/learning_rate",
                optimizer.param_groups[0]["lr"],
                epoch,
            )
        print(f"[TRAIN] {json.dumps(record)}", flush=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "training_config": asdict(config),
            "epoch": epoch,
            "metrics": record,
            "best_validation_loss": min(
                best_validation_loss,
                validation_metrics["loss"],
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else {},
            "action_frame": "d405_camera_optical",
            "action_semantics": "normalized_residual_camera_twist",
            "linear_action_scale_m_s": LINEAR_ACTION_SCALE_M_S,
            "angular_action_scale_rad_s": ANGULAR_ACTION_SCALE_RAD_S,
            "rotation_camera_in_tcp": list(
                D405WristCameraConfig().rotation_camera_in_calibration_parent
            ),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        history_path.write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
    history_path.write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    if writer is not None:
        writer.close()
    print(
        f"[TRAIN] Complete: train_frames={len(train_dataset)} "
        f"validation_frames={len(validation_dataset)} best={best_validation_loss:.6f} "
        f"checkpoint={args.output_dir / 'best.pt'} "
        f"tensorboard={tensorboard_dir if writer is not None else 'disabled'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
