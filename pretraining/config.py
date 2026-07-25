from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def _add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        dest=name,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CIPHER self-supervised pretraining configuration"
    )

    parser.add_argument("--config", default="pretraining/configs/pretrain.yaml")
    parser.add_argument("--logdir", default="outputs/pretraining")
    parser.add_argument("--datalist", default="pretraining/jsons/SwinUNETRPretraining.json")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-num", dest="eval_num", type=int, default=10_000)
    parser.add_argument("--log-every", dest="log_every", type=int, default=50)
    parser.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=500)

    parser.add_argument("--in-channels", dest="in_channels", type=int, default=1)
    parser.add_argument("--feature-size", dest="feature_size", type=int, default=48)
    parser.add_argument("--spatial-dims", dest="spatial_dims", type=int, default=3)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=2)
    parser.add_argument("--window-size", dest="window_size", type=int, default=7)
    parser.add_argument("--depths", nargs=4, type=int, default=[2, 2, 2, 2])
    parser.add_argument("--num-heads", dest="num_heads", nargs=4, type=int, default=[3, 6, 12, 24])
    parser.add_argument("--dropout-path-rate", dest="dropout_path_rate", type=float, default=0.0)
    _add_bool_argument(parser, "use_v2", default=True, help_text="Use the SwinUNETR v2 residual blocks.")
    _add_bool_argument(parser, "use_checkpoint", default=False, help_text="Use gradient checkpointing.")

    parser.add_argument("--a-min", dest="a_min", type=float, default=-1000.0)
    parser.add_argument("--a-max", dest="a_max", type=float, default=400.0)
    parser.add_argument("--b-min", dest="b_min", type=float, default=0.0)
    parser.add_argument("--b-max", dest="b_max", type=float, default=1.0)
    _add_bool_argument(parser, "apply_spacing", default=False, help_text="Resample images inside the loader.")
    parser.add_argument("--space-x", dest="space_x", type=float, default=0.8)
    parser.add_argument("--space-y", dest="space_y", type=float, default=0.8)
    parser.add_argument("--space-z", dest="space_z", type=float, default=2.5)
    parser.add_argument("--roi-x", dest="roi_x", type=int, default=96)
    parser.add_argument("--roi-y", dest="roi_y", type=int, default=96)
    parser.add_argument("--roi-z", dest="roi_z", type=int, default=32)

    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--sw-batch-size", dest="sw_batch_size", type=int, default=2)
    parser.add_argument(
        "--dataset-type",
        dest="dataset_type",
        choices=["standard", "cache", "smartcache"],
        default="standard",
    )
    parser.add_argument("--cache-rate", dest="cache_rate", type=float, default=0.5)
    parser.add_argument(
        "--smartcache-replace-rate",
        dest="smartcache_replace_rate",
        type=float,
        default=1.0,
    )
    parser.add_argument("--smartcache-cache-num", dest="smartcache_cache_num", type=int, default=None)

    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=6e-6)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    _add_bool_argument(parser, "lr_decay", default=True, help_text="Enable learning-rate scheduling.")
    parser.add_argument(
        "--lr-schedule",
        dest="lr_schedule",
        choices=["warmup_cosine", "poly"],
        default="warmup_cosine",
    )
    _add_bool_argument(parser, "grad_clip", default=False, help_text="Clip gradient norm.")
    parser.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=1.0)
    _add_bool_argument(parser, "amp", default=False, help_text="Enable CUDA automatic mixed precision.")

    parser.add_argument("--rotation-weight", dest="rotation_weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", dest="contrastive_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", dest="reconstruction_weight", type=float, default=1.0)
    parser.add_argument(
        "--contrastive-temperature",
        dest="contrastive_temperature",
        type=float,
        default=0.5,
    )
    parser.add_argument("--max-drop", dest="max_drop", type=float, default=0.3)
    parser.add_argument("--max-block-size", dest="max_block_size", type=float, default=0.25)
    parser.add_argument("--block-tolerance", dest="block_tolerance", type=float, default=0.05)

    parser.add_argument("--dist-url", dest="dist_url", default="env://")
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=0)
    return parser


def _normalize_yaml_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {str(key).replace("-", "_"): value for key, value in config.items()}


def get_config() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="pretraining/configs/pretrain.yaml")
    preliminary, _ = pre_parser.parse_known_args()

    parser = build_parser()
    config_path = Path(preliminary.config)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML configuration must be a mapping: {config_path}")

        loaded = _normalize_yaml_keys(loaded)
        valid_destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(loaded) - valid_destinations)
        if unknown:
            raise ValueError(
                "Unknown configuration field(s): " + ", ".join(unknown)
            )
        parser.set_defaults(**loaded)

    args = parser.parse_args()
    args.config = str(config_path)

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.a_max <= args.a_min:
        raise ValueError("a_max must be greater than a_min.")
    if args.b_max <= args.b_min:
        raise ValueError("b_max must be greater than b_min.")
    if args.eval_num <= 0 or args.log_every <= 0:
        raise ValueError("eval_num and log_every must be positive.")
    if args.batch_size <= 0 or args.sw_batch_size <= 0:
        raise ValueError("batch_size and sw_batch_size must be positive.")

    return args
