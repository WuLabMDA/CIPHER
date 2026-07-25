from __future__ import annotations

import datetime
import os
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from pretraining.config import get_config
from pretraining.losses import SelfSupervisedLoss
from pretraining.models import SSLHead
from pretraining.optimizers import WarmupCosineSchedule
from pretraining.utils.augmentations import (
    random_axial_rotation,
    random_block_augmentation,
)
from pretraining.utils.data import get_loaders


def setup_runtime(args) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this training pipeline.")

    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    args.rank = 0
    args.world_size = 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            timeout=datetime.timedelta(hours=2),
        )
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda", 0)

    args.device = device
    torch.backends.cudnn.benchmark = True
    return device


def seed_everything(seed: int, rank: int) -> None:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)


def build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    global_step: int,
    epoch: int,
    best_reconstruction_loss: float,
) -> None:
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "best_reconstruction_loss": best_reconstruction_loss,
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint["state_dict"]))

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return (
        int(checkpoint.get("global_step", 0)),
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_reconstruction_loss", float("inf"))),
    )


def assert_finite(name: str, tensor: torch.Tensor, filenames) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values in {name}; files={filenames}")


def make_tensorboard_images(
    original: torch.Tensor,
    augmented: torch.Tensor,
    reconstructed: torch.Tensor,
    reconstructed_original: torch.Tensor,
) -> list[torch.Tensor]:
    middle_slice = original.shape[-1] // 2

    def normalized_slice(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().float().cpu().numpy()[0, 0, :, :, middle_slice]
        array = array - array.min()
        return array / (array.max() + 1e-8)

    original_slice = normalized_slice(original)
    augmented_slice = normalized_slice(augmented)
    reconstructed_slice = normalized_slice(reconstructed)
    reconstructed_original_slice = normalized_slice(reconstructed_original)

    difference = original_slice - reconstructed_original_slice
    maximum = max(float(np.max(np.abs(difference))), 1e-8)
    difference_for_colormap = np.clip(0.5 + 0.5 * difference / maximum, 0.0, 1.0)
    difference_rgb = matplotlib.colormaps["seismic"](difference_for_colormap)[..., :3]

    return [
        torch.from_numpy(original_slice[None, ...]).float(),
        torch.from_numpy(augmented_slice[None, ...]).float(),
        torch.from_numpy(reconstructed_slice[None, ...]).float(),
        torch.from_numpy(difference_rgb).permute(2, 0, 1).float(),
    ]


def validate(model, loader, loss_function, args):
    model.eval()
    total_loss = 0.0
    reconstruction_loss = 0.0
    batch_count = 0
    images = None

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs = batch["image"].to(args.device, non_blocking=True)
            first, first_rotation = random_axial_rotation(inputs)
            second, second_rotation = random_axial_rotation(inputs)
            first_augmented = random_block_augmentation(first, args)
            second_augmented = random_block_augmentation(second, args)

            with autocast(enabled=args.amp):
                first_rotation_logits, first_embedding, first_reconstruction = model(
                    first_augmented
                )
                second_rotation_logits, second_embedding, second_reconstruction = model(
                    second_augmented
                )
                _, _, reconstructed_original = model(first)

                rotation_logits = torch.cat(
                    [first_rotation_logits, second_rotation_logits], dim=0
                )
                rotation_targets = torch.cat(
                    [first_rotation, second_rotation], dim=0
                )
                reconstructions = torch.cat(
                    [first_reconstruction, second_reconstruction], dim=0
                )
                targets = torch.cat([first, second], dim=0)

                loss, component_losses = loss_function(
                    rotation_logits,
                    rotation_targets,
                    first_embedding,
                    second_embedding,
                    reconstructions,
                    targets,
                )

            total_loss += float(loss.item())
            reconstruction_loss += float(component_losses[2].item())
            batch_count += 1

            if batch_index == 0 and args.rank == 0:
                images = make_tensorboard_images(
                    first,
                    first_augmented,
                    first_reconstruction,
                    reconstructed_original,
                )

    totals = torch.tensor(
        [total_loss, reconstruction_loss, float(batch_count)],
        dtype=torch.float64,
        device=args.device,
    )
    if args.distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    if totals[2].item() == 0:
        raise RuntimeError(
            "Validation produced zero batches. Reduce batch_size or add validation cases."
        )

    return (
        totals[0].item() / totals[2].item(),
        totals[1].item() / totals[2].item(),
        images,
    )


def log_validation(writer, global_step, total_loss, reconstruction_loss, images):
    if writer is None:
        return
    writer.add_scalar("validation/loss_total", total_loss, global_step)
    writer.add_scalar("validation/loss_reconstruction", reconstruction_loss, global_step)
    if images is not None:
        writer.add_image("validation/original", images[0], global_step)
        writer.add_image("validation/augmented", images[1], global_step)
        writer.add_image("validation/reconstruction", images[2], global_step)
        writer.add_image("validation/difference", images[3], global_step)


def main() -> None:
    args = get_config()
    device = setup_runtime(args)
    seed_everything(args.seed, args.rank)

    output_dir = Path(args.logdir)
    writer = None
    if args.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.resolve()}")
        print(f"World size: {args.world_size}")
        writer = SummaryWriter(str(output_dir))

    train_loader, validation_loader = get_loaders(args)
    total_steps = args.epochs * len(train_loader)

    model = SSLHead(args).to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = build_optimizer(model, args)
    scheduler = None
    if args.lr_decay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=args.warmup_steps,
                total_steps=total_steps,
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: (1.0 - min(step, total_steps) / total_steps) ** 0.9,
            )

    global_step = 0
    start_epoch = 0
    best_reconstruction_loss = float("inf")
    if args.resume:
        global_step, start_epoch, best_reconstruction_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
        )
        if args.rank == 0:
            print(f"Resumed from {args.resume} at step {global_step}.")

    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

    local_crop_batch_size = args.batch_size * args.sw_batch_size
    loss_function = SelfSupervisedLoss(local_crop_batch_size, args).to(device)
    scaler = GradScaler(enabled=args.amp)
    last_validation_step = -1

    for epoch in range(start_epoch, args.epochs):
        if global_step >= total_steps:
            break
        if args.distributed and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_losses: list[float] = []
        epoch_reconstruction_losses: list[float] = []

        for batch_index, batch in enumerate(train_loader):
            if global_step >= total_steps:
                break
            step_start = time.time()
            inputs = batch["image"].to(device, non_blocking=True)
            metadata = batch.get("image_meta_dict", {})
            filenames = metadata.get("filename_or_obj", "unknown")

            first, first_rotation = random_axial_rotation(inputs)
            second, second_rotation = random_axial_rotation(inputs)
            first_augmented = random_block_augmentation(first, args)
            second_augmented = random_block_augmentation(second, args)

            assert_finite("first_augmented", first_augmented, filenames)
            assert_finite("second_augmented", second_augmented, filenames)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                first_rotation_logits, first_embedding, first_reconstruction = model(
                    first_augmented
                )
                second_rotation_logits, second_embedding, second_reconstruction = model(
                    second_augmented
                )

                for name, tensor in (
                    ("first_rotation_logits", first_rotation_logits),
                    ("first_embedding", first_embedding),
                    ("first_reconstruction", first_reconstruction),
                    ("second_rotation_logits", second_rotation_logits),
                    ("second_embedding", second_embedding),
                    ("second_reconstruction", second_reconstruction),
                ):
                    assert_finite(name, tensor, filenames)

                rotation_logits = torch.cat(
                    [first_rotation_logits, second_rotation_logits], dim=0
                )
                rotation_targets = torch.cat(
                    [first_rotation, second_rotation], dim=0
                )
                reconstructions = torch.cat(
                    [first_reconstruction, second_reconstruction], dim=0
                )
                targets = torch.cat([first, second], dim=0)

                loss, component_losses = loss_function(
                    rotation_logits,
                    rotation_targets,
                    first_embedding,
                    second_embedding,
                    reconstructions,
                    targets,
                )

            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            epoch_losses.append(float(loss.item()))
            epoch_reconstruction_losses.append(float(component_losses[2].item()))

            if args.rank == 0 and writer is not None:
                writer.add_scalar("train/loss_total_step", loss.item(), global_step)
                writer.add_scalar(
                    "train/loss_rotation_step", component_losses[0].item(), global_step
                )
                writer.add_scalar(
                    "train/loss_contrastive_step", component_losses[1].item(), global_step
                )
                writer.add_scalar(
                    "train/loss_reconstruction_step",
                    component_losses[2].item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )

                if global_step % args.log_every == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} "
                        f"step={global_step}/{total_steps} "
                        f"loss={loss.item():.5f} "
                        f"reconstruction={component_losses[2].item():.5f} "
                        f"seconds={time.time() - step_start:.2f}"
                    )

            if global_step % args.eval_num == 0:
                validation_total, validation_reconstruction, images = validate(
                    model, validation_loader, loss_function, args
                )
                last_validation_step = global_step

                if args.rank == 0:
                    log_validation(
                        writer,
                        global_step,
                        validation_total,
                        validation_reconstruction,
                        images,
                    )
                    print(
                        f"validation step={global_step}: "
                        f"total={validation_total:.5f}, "
                        f"reconstruction={validation_reconstruction:.5f}"
                    )

                    if validation_reconstruction < best_reconstruction_loss:
                        best_reconstruction_loss = validation_reconstruction
                        save_checkpoint(
                            output_dir / "model_best_reconstruction.pt",
                            model,
                            optimizer,
                            scheduler,
                            global_step=global_step,
                            epoch=epoch,
                            best_reconstruction_loss=best_reconstruction_loss,
                        )
                        print("Saved new best reconstruction checkpoint.")

        if args.rank == 0 and writer is not None:
            writer.add_scalar("train/loss_total_epoch", np.mean(epoch_losses), epoch + 1)
            writer.add_scalar(
                "train/loss_reconstruction_epoch",
                np.mean(epoch_reconstruction_losses),
                epoch + 1,
            )

    if global_step != last_validation_step:
        validation_total, validation_reconstruction, images = validate(
            model, validation_loader, loss_function, args
        )
        if args.rank == 0:
            log_validation(
                writer,
                global_step,
                validation_total,
                validation_reconstruction,
                images,
            )
            if validation_reconstruction < best_reconstruction_loss:
                best_reconstruction_loss = validation_reconstruction
                save_checkpoint(
                    output_dir / "model_best_reconstruction.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step=global_step,
                    epoch=args.epochs - 1,
                    best_reconstruction_loss=best_reconstruction_loss,
                )

    if args.rank == 0:
        save_checkpoint(
            output_dir / "model_final.pt",
            model,
            optimizer,
            scheduler,
            global_step=global_step,
            epoch=args.epochs,
            best_reconstruction_loss=best_reconstruction_loss,
        )
        torch.save(
            unwrap_model(model).state_dict(),
            output_dir / "final_model_state_dict.pth",
        )
        if writer is not None:
            writer.close()

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
