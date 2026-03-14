#!/usr/bin/env python3
"""Prepare a reproducible resume directory from an existing training run.

This clones checkpoint artifacts into a fresh output directory, optionally
resuming from the previous best checkpoint, and rewrites the optimizer learning
rate so `--train_from_last_checkpoint 1` resumes with the requested value.

Example:
    uv run python scripts/prepare_resume_checkpoint.py \
        --source-run-dir checkpoints/log_2026-03-14(12:27:57) \
        --output-run-dir checkpoints/log_2026-03-14(12:27:57)_resume_lr1e-4 \
        --resume-from best \
        --learning-rate 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


CHECKPOINT_FILES = {
    "last": "last_checkpoint.pt",
    "best": "last_best_checkpoint.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fresh checkpoint directory for reproducible resume with a new LR."
    )
    parser.add_argument("--source-run-dir", required=True, help="Existing checkpoint directory to clone from.")
    parser.add_argument("--output-run-dir", required=True, help="New checkpoint directory to create.")
    parser.add_argument(
        "--resume-from",
        choices=("last", "best"),
        default="last",
        help="Which source checkpoint should become the new last_checkpoint.pt.",
    )
    parser.add_argument(
        "--learning-rate",
        required=True,
        type=float,
        help="Optimizer LR to write into the resume checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    parser.add_argument(
        "--config",
        help="Optional training config to copy into the output directory and patch with the new init_learning_rate.",
    )
    return parser.parse_args()


def resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    return path


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def rewrite_optimizer_lr(checkpoint: dict, learning_rate: float) -> dict:
    optimizer = checkpoint.get("optimizer")
    if optimizer is None:
        raise KeyError("Checkpoint does not contain optimizer state.")

    for group in optimizer.get("param_groups", []):
        group["lr"] = learning_rate
        if "initial_lr" in group:
            group["initial_lr"] = learning_rate
    return checkpoint


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    torch.save(checkpoint, path)


def copy_or_patch_config(
    source_run_dir: Path,
    output_run_dir: Path,
    learning_rate: float,
    config_path: str | None = None,
) -> Path | None:
    source_config = Path(config_path).expanduser().resolve() if config_path else source_run_dir / "config.yaml"
    if not source_config.exists():
        return None

    try:
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    except Exception:
        shutil.copy2(source_config, output_run_dir / "config.yaml")
        return output_run_dir / "config.yaml"

    if isinstance(config, dict):
        config["init_learning_rate"] = learning_rate
        rendered = yaml.safe_dump(config, sort_keys=False)
        (output_run_dir / "config.yaml").write_text(rendered, encoding="utf-8")
        return output_run_dir / "config.yaml"
    else:
        shutil.copy2(source_config, output_run_dir / "config.yaml")
        return output_run_dir / "config.yaml"


def write_metadata(
    source_run_dir: Path,
    output_run_dir: Path,
    resume_from: str,
    learning_rate: float,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": os.fspath(source_run_dir),
        "output_run_dir": os.fspath(output_run_dir),
        "resume_from": resume_from,
        "source_checkpoint": CHECKPOINT_FILES[resume_from],
        "learning_rate": learning_rate,
    }
    (output_run_dir / "resume_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    source_run_dir = resolve_run_dir(args.source_run_dir)
    output_run_dir = Path(args.output_run_dir).expanduser().resolve()
    ensure_output_dir(output_run_dir, overwrite=args.overwrite)

    source_resume_checkpoint = source_run_dir / CHECKPOINT_FILES[args.resume_from]
    if not source_resume_checkpoint.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {source_resume_checkpoint}")

    resume_checkpoint = rewrite_optimizer_lr(
        load_checkpoint(source_resume_checkpoint),
        args.learning_rate,
    )
    save_checkpoint(resume_checkpoint, output_run_dir / "last_checkpoint.pt")

    source_best_checkpoint = source_run_dir / CHECKPOINT_FILES["best"]
    if args.resume_from == "best":
        save_checkpoint(resume_checkpoint, output_run_dir / "last_best_checkpoint.pt")
    elif source_best_checkpoint.exists():
        shutil.copy2(source_best_checkpoint, output_run_dir / "last_best_checkpoint.pt")

    output_config = copy_or_patch_config(
        source_run_dir,
        output_run_dir,
        args.learning_rate,
        config_path=args.config,
    )
    write_metadata(
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
        resume_from=args.resume_from,
        learning_rate=args.learning_rate,
    )

    print(f"Prepared resume directory: {output_run_dir}")
    print(f"Resume source: {source_resume_checkpoint}")
    print(f"New optimizer learning rate: {args.learning_rate}")
    print()
    print("Launch with:")
    if output_config is not None:
        print(
            "uv run python train.py --config "
            f"{output_config} --checkpoint_dir {output_run_dir} --train_from_last_checkpoint 1"
        )
    else:
        print(
            "uv run python train.py --config /path/to/your/config.yaml "
            f"--checkpoint_dir {output_run_dir} --train_from_last_checkpoint 1"
        )


if __name__ == "__main__":
    main()
