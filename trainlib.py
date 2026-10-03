"""Shared training controllers factored out of train-lora.py for reuse.

Standing up the "factor shared module first" decision (2026-06-18): the
encoder-alignment / alignment-as-decoder-aux finetuner (see
research/awesome-align-embeddinggemma.md) reuses train-lora's debugged
machinery instead of reimplementing it. This module is the canonical home.

Migration status: each extraction is deduped from train-lora.py only after a
pixi-gemma4 `train-lora.py --help` load smoke (do not break the main trainer on
an unverified edit), and committed separately so any regression is bisectable.

Planned contents (extract in this order, smoke train-lora after each):
  1. SophiaG (below) — self-contained. [done: train-lora imports it, smoke-verified]
  2. LengthBucketedBatchSampler, LogicalMixBatchSampler (logical batches) —
     done. Decoupled from PromptVariantDataset by injecting a precomputed
     `lengths` list: both samplers only ever read self.lengths (self.dataset
     was a dead field), so the prompt-variant-specific PromptVariantDataset +
     _example_token_length stay in train-lora.py, which now materializes
     lengths at the call site and passes them in.
  3. ValidationController — early-stop + exact-val recheck (this commit). Owns the
     stop decision + state; the host (RegionScaleTrainer) supplies effect hooks and,
     until commit 4, the reactive LR (anneal/rebound) it consults on exhaustion.
     Behavior pinned by tests/unit/test_val_control_characterization.py.
  4. PatienceLrController — reactive LR anneal/rebound that consumes validation state
     (done). Owns the anneal/rebound decisions + watch state; the cumulative applied LR
     scale stays host-side (maintained by the optimizer mutation). RegionScaleTrainer's
     _update_val_cycle_progress / _maybe_adjust_patience_lr_rebound / _try_patience_lr_anneal
     now delegate to it. Behavior pinned by the same characterization golden.
  5. RefEmbedAux — the reference-embedding aux loss core (pool + cosine/mse + head)
     (done). The host keeps the target cache, the collator, head construction/save, and
     the CLI. The alignment-as-decoder-aux experiment extends pool_hidden / loss_value to
     be alignment-conditioned (per-target-token toward aligned source spans).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
import threading
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from transformers import TrainerCallback

from checkpoint_mirror import CheckpointMirror
from log_format import info
from run_config import merge_run_config, read_run_config_basis  # noqa: F401

_CHECKPOINT_PREFIX = "checkpoint-"
_TRAINER_STATE = "trainer_state.json"
_MODEL_STATE_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_TRAINER_RESUME_FILES = ("optimizer.pt", "scheduler.pt")


def add_checkpoint_mirror_args(parser):
    parser.add_argument(
        "--checkpoint-mirror",
        help="Atomic output replica: /absolute/path or SSH-alias:/absolute/path; requires rsync and Python 3",
    )
    parser.add_argument(
        "--checkpoint-mirror-interval",
        type=float,
        default=3600,
        help="Seconds of training between replica updates; final output always flushes",
    )
    parser.add_argument(
        "--checkpoint-mirror-timeout",
        type=float,
        default=600,
        help="Timeout per transfer attempt in seconds (two attempts)",
    )
    parser.add_argument(
        "--checkpoint-mirror-ssh",
        default="ssh -o BatchMode=yes -o ConnectTimeout=10",
        help="SSH command, optionally including identity/config options; no shell evaluation",
    )


class CheckpointMirrorCallback(TrainerCallback):
    """Time-bound checkpoint requests and publish only after a completed save."""

    def __init__(self, options):
        self.mirror = (
            CheckpointMirror(
                options.checkpoint_mirror,
                interval_seconds=options.checkpoint_mirror_interval,
                timeout_seconds=options.checkpoint_mirror_timeout,
                ssh_command=options.checkpoint_mirror_ssh,
            )
            if options.checkpoint_mirror
            else None
        )
        if self.mirror:
            import time

            self.mirror.last_success = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.mirror:
            import time

            if time.monotonic() - self.mirror.last_success >= self.mirror.interval:
                control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if self.mirror and state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not is_valid_trainer_checkpoint(checkpoint):
                raise RuntimeError(f"checkpoint mirror refuses incomplete resume state: {checkpoint}")
            self.mirror.save(Path(args.output_dir))
        if self.mirror:
            import time

            if time.monotonic() - self.mirror.last_success >= self.mirror.interval:
                self.mirror.last_success = time.monotonic()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.finish(args.output_dir, is_world_process_zero=state.is_world_process_zero)
        return control

    def finish(self, output_dir, *, is_world_process_zero=True):
        if self.mirror and is_world_process_zero:
            self.mirror.save(Path(output_dir), force=True)


def checkpoint_step(path: str | Path) -> int | None:
    """Return the numeric step for a HuggingFace ``checkpoint-N`` directory."""
    name = Path(path).name
    if not name.startswith(_CHECKPOINT_PREFIX):
        return None
    suffix = name[len(_CHECKPOINT_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def is_valid_trainer_checkpoint(path: str | Path) -> bool:
    """True when ``path`` is complete enough for Trainer checkpoint resume.

    Judged on contents alone. The directory name is how checkpoints are *discovered* and
    ordered, not what makes one resumable, and an archive that saves resume state under
    its own naming is resumable in every sense that matters here.

    Interrupted saves can leave a high-numbered ``checkpoint-N`` directory behind.
    Treat a checkpoint as resumable only after the trainer state, model/adaptor
    weights, optimizer state, and scheduler state all reached disk.
    """
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        return False
    state_path = checkpoint / _TRAINER_STATE
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if int(state.get("global_step") or 0) <= 0:
        return False
    if not any((checkpoint / name).is_file() for name in _MODEL_STATE_FILES):
        return False
    return all((checkpoint / name).is_file() for name in _TRAINER_RESUME_FILES)


def last_valid_trainer_checkpoint(output_dir: str | Path | None) -> Path | None:
    """Return the newest valid ``checkpoint-N`` under ``output_dir``, if any."""
    if not output_dir:
        return None
    root = Path(output_dir)
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        step = checkpoint_step(child)
        if step is not None and is_valid_trainer_checkpoint(child):
            candidates.append((step, child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def detach_best_directory(best_output: Path) -> None:
    """Clear `best` so a real save cannot write through a symlink into a checkpoint.

    `best` may be a symlink into a numbered checkpoint from an earlier phase or run.
    Saving onto it would follow the link and rewrite that checkpoint in place, which is
    both the resume point and the thing another `best` may point at. Unlink the symlink
    itself; never touch its target.
    """
    if best_output.is_symlink():
        best_output.unlink()
    elif best_output.exists():
        shutil.rmtree(best_output)


def link_best_to_checkpoint(best_output: Path, checkpoint: Path) -> None:
    """Point `best` at the selected checkpoint instead of copying its weights again.

    The copy was an exact duplicate: identical config and identical weights. The link is
    relative so the run directory can be moved or rsynced whole, and the target survives
    `save_total_limit` rotation because the trainer records it as `best_model_checkpoint`,
    which the rotation order keeps.
    """
    detach_best_directory(best_output)
    best_output.symlink_to(os.path.relpath(checkpoint, best_output.parent), target_is_directory=True)


def checkpoint_steps(output_dir: str | Path) -> dict[int, Path]:
    """Every ``checkpoint-N`` directory under ``output_dir``, keyed by step."""
    steps: dict[int, Path] = {}
    for child in Path(output_dir).iterdir():
        step = checkpoint_step(child)
        if step is not None and child.is_dir() and not child.is_symlink():
            steps[step] = child
    return steps


def referenced_paths(output_dir: str | Path, depth: int = 2) -> set[Path]:
    """Resolved targets of every symlink inside the run directory, to `depth` levels.

    A link is a claim that its target is still wanted. Collecting the claims once lets
    every retention rule honour them without knowing who made them.
    """
    root = Path(output_dir)
    targets: set[Path] = set()
    for level in range(1, depth + 1):
        for entry in root.glob("/".join(["*"] * level)):
            if not entry.is_symlink():
                continue
            try:
                targets.add(entry.resolve())
            except OSError:
                continue
    return targets


def referenced_checkpoints(output_dir: str | Path, depth: int = 2) -> set[int]:
    """Checkpoint steps that something inside the run directory links to.

    `best` is the obvious case but not the only one: an archive such as
    `major-checkpoints/` may hold links to notable steps rather than copies of them.
    Retention honours the link instead of requiring every archiving scheme to teach the
    pruner about itself.
    """
    root = Path(output_dir)
    steps = {path.resolve(): step for step, path in checkpoint_steps(root).items()}
    return {step for target in referenced_paths(root, depth) if (step := steps.get(target)) is not None}


def major_checkpoints(output_dir: str | Path, archive: str = "major-checkpoints") -> dict[int, Path]:
    """Accuracy-triggered archive entries, keyed by the step in their directory name."""
    root = Path(output_dir) / archive
    entries: dict[int, Path] = {}
    if not root.is_dir():
        return entries
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink() or not child.name.startswith("step-"):
            continue
        head = child.name.split("-")
        if len(head) > 1 and head[1].isdigit():
            entries[int(head[1])] = child
    return entries


def link_recent_majors(
    output_dir: str | Path,
    keep: int = 5,
    archive: str = "major-checkpoints",
    save: str = "save",
) -> list[Path]:
    """Declare the newest `keep` archive entries as wanted, by linking them into `save/`.

    Separating the declaration from the deletion is what makes this safe to run twice and
    easy to override: a human who wants an older entry kept adds a link beside these, and
    the pruner honours it like any other.
    """
    entries = major_checkpoints(output_dir, archive)
    if not entries:
        return []
    directory = Path(output_dir) / save
    directory.mkdir(parents=True, exist_ok=True)
    linked = []
    for step in sorted(entries, reverse=True)[:keep]:
        link = directory / entries[step].name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            continue
        link.symlink_to(os.path.relpath(entries[step], directory), target_is_directory=True)
        linked.append(link)
    return linked


def prune_unreferenced_majors(
    output_dir: str | Path,
    archive: str = "major-checkpoints",
    depth: int = 2,
) -> tuple[list[int], int]:
    """Drop archive entries nothing links to; keep every entry that is claimed."""
    entries = major_checkpoints(output_dir, archive)
    if not entries:
        return [], 0
    wanted = referenced_paths(output_dir, depth)
    dropped, freed = [], 0
    for step in sorted(entries):
        path = entries[step]
        if path.resolve() in wanted:
            continue
        freed += sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
        shutil.rmtree(path)
        dropped.append(step)
    return dropped, freed


def thin_optimizer_state(out: str | Path) -> tuple[list[int], int]:
    """Delete optimizer state from every numbered checkpoint except the terminal one.

    Continuations initialize from model weights and start a fresh optimizer and
    schedule, so only exact resume reads optimizer state, and only from the
    run's last checkpoint (user, 2026-09-27). Weights, trainer state and the
    small scheduler file stay everywhere. Returns the thinned steps and bytes freed.
    """
    steps = checkpoint_steps(out)
    if not steps:
        return [], 0
    terminal = max(steps)
    thinned, freed = [], 0
    for step, path in sorted(steps.items()):
        optimizer = path / "optimizer.pt"
        if step == terminal or not optimizer.is_file() or optimizer.is_symlink():
            continue
        freed += optimizer.stat().st_size
        optimizer.unlink()
        thinned.append(step)
    return thinned, freed


def prune_checkpoints_before_selected(out: str | Path, selected_step: int) -> tuple[list[int], int]:
    """Drop numbered checkpoints older than the selected one and its predecessor.

    Kept: the selected checkpoint, because `best` links to it and it is the resume point;
    the one immediately before it, so checkpoint averaging over the pair stays possible;
    and everything after it, which is the trajectory beyond the selection and the only
    record of what continued training did. Everything older is reproducible from the seed
    at the cost of a rerun, and is what actually fills the disk.
    """
    out = Path(out)
    steps = checkpoint_steps(out)
    earlier = sorted(step for step in steps if step < selected_step)
    keep = {step for step in steps if step >= selected_step}
    if earlier:
        keep.add(earlier[-1])
    keep |= referenced_checkpoints(out)
    # Never leave a run with nothing to resume from. Selection can land on a checkpoint
    # that was saved without optimizer state, and links express inference intent rather
    # than resumability, so neither rule above guarantees a survivor.
    if not any(is_valid_trainer_checkpoint(steps[step]) for step in keep):
        resumable = [step for step in sorted(steps, reverse=True) if is_valid_trainer_checkpoint(steps[step])]
        if resumable:
            keep.add(resumable[0])
    dropped = sorted(step for step in steps if step not in keep)
    freed = 0
    for step in dropped:
        path = steps[step]
        # A symlinked `best` must never be followed into a directory being removed.
        if path.is_symlink():
            continue
        freed += sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
        shutil.rmtree(path)
    return dropped, freed


def _run_metrics():
    """The shared GPU-run metrics helpers, or inert stand-ins when unavailable."""
    here = Path(__file__).resolve().parent
    if str(here / "scripts") not in sys.path:
        sys.path.insert(0, str(here / "scripts"))
    import agents_run_metrics

    return agents_run_metrics


class RunMetricsCallback(TrainerCallback):
    """Publish a platform-upgrade baseline for every tracked training run.

    Registered by default rather than on request, because the metrics that make an upgrade
    comparable are exactly the ones nobody remembers to turn on: four consecutive run
    records were found carrying none. Costs one allocator query per run and is a no-op
    outside a tracked run.
    """

    def __init__(self, phase: str = "train"):
        self._metrics = _run_metrics().RunMetrics(phase)
        self._samples = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._metrics.start()

    def on_train_end(self, args, state, control, **kwargs):
        world = max(1, int(getattr(args, "world_size", 1) or 1))
        per_step = int(args.per_device_train_batch_size) * int(args.gradient_accumulation_steps) * world
        self._metrics.finish(
            steps=int(state.global_step or 0),
            samples=int(state.global_step or 0) * per_step,
            batch_shape=f"{args.per_device_train_batch_size}x{args.gradient_accumulation_steps}",
            world_size=world,
            dtype="bf16"
            if getattr(args, "bf16", False)
            else ("fp16" if getattr(args, "fp16", False) else "fp32"),
            gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
        )


class OrderlyCheckpointCallback(TrainerCallback):
    """Turn process signals into resumable Trainer checkpoints.

    ``SIGTERM`` and ``SIGINT`` request a save and stop at the next completed
    optimizer step. ``SIGHUP`` requests the same save without stopping. Signal
    handlers only mutate small Python state; HuggingFace Trainer performs the
    actual save through its ordinary checkpoint path, after gradients have
    been applied and before the training loop observes the stop request.

    A second terminating signal restores the operating-system default and
    re-sends the signal, preserving an immediate forced-exit escape hatch.
    """

    def __init__(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
        stop_reason: str = "external stop request",
        checkpoint_on_sighup: bool = True,
    ) -> None:
        self._stop_requested = stop_requested
        self._external_stop_reason = str(stop_reason)
        self._checkpoint_on_sighup = bool(checkpoint_on_sighup)
        self._stop_reason: str | None = None
        self._checkpoint_reason: str | None = None
        self._termination_signal: int | None = None
        self._stop_dispatched_step: int | None = None
        self._checkpoint_dispatched_step: int | None = None
        self._saved_steps: set[int] = set()
        self._previous_handlers: dict[int, object] = {}
        self._installed_handlers: dict[int, object] = {}

    @property
    def termination_signal(self) -> int | None:
        return self._termination_signal

    @property
    def interrupted_exit_code(self) -> int | None:
        return None if self._termination_signal is None else 128 + self._termination_signal

    def request_stop(self, reason: str, *, signum: int | None = None) -> None:
        """Request one checkpoint followed by an orderly training stop."""
        if self._stop_reason is None:
            self._stop_reason = str(reason)
        if signum is not None and self._termination_signal is None:
            self._termination_signal = int(signum)

    def request_checkpoint(self, reason: str) -> None:
        """Request one checkpoint while leaving training active."""
        if self._checkpoint_reason is None:
            self._checkpoint_reason = str(reason)

    def _handle_termination_signal(self, signum: int, _frame) -> None:  # noqa: ANN001
        if self._stop_reason is not None:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        name = signal.Signals(signum).name
        self.request_stop(f"signal {name}", signum=signum)
        os.write(
            2,
            (
                f"\n[trainlib] {name} received; checkpointing at the next "
                "optimizer boundary before exit. Send again to force exit.\n"
            ).encode(),
        )

    def _handle_checkpoint_signal(self, signum: int, _frame) -> None:  # noqa: ANN001
        name = signal.Signals(signum).name
        self.request_checkpoint(f"signal {name}")
        os.write(
            2,
            f"\n[trainlib] {name} received; checkpointing at the next optimizer boundary.\n".encode(),
        )

    @contextmanager
    def signal_handlers(self):
        """Install handlers for the duration of the owning Trainer call."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("orderly checkpoint signal handlers require the Python main thread")
        if self._previous_handlers:
            raise RuntimeError("orderly checkpoint signal handlers are already installed")

        handlers = {
            signal.SIGTERM: self._handle_termination_signal,
            signal.SIGINT: self._handle_termination_signal,
        }
        if self._checkpoint_on_sighup and hasattr(signal, "SIGHUP"):
            handlers[signal.SIGHUP] = self._handle_checkpoint_signal
        for signum, handler in handlers.items():
            self._previous_handlers[signum] = signal.getsignal(signum)
            self._installed_handlers[signum] = handler
            signal.signal(signum, handler)
        try:
            yield self
        finally:
            for signum, previous in self._previous_handlers.items():
                if signal.getsignal(signum) == self._installed_handlers[signum]:
                    signal.signal(signum, previous)
            self._previous_handlers.clear()
            self._installed_handlers.clear()

    def _synchronize_requests(self) -> tuple[bool, bool]:
        stop = self._stop_reason is not None
        checkpoint = self._checkpoint_reason is not None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.distributed.get_backend() == "nccl"
                else torch.device("cpu")
            )
            requests = torch.tensor([int(stop), int(checkpoint)], dtype=torch.int32, device=device)
            torch.distributed.all_reduce(requests, op=torch.distributed.ReduceOp.MAX)
            stop, checkpoint = (bool(value) for value in requests.tolist())
            if stop and self._stop_reason is None:
                self._stop_reason = "termination requested by another distributed rank"
            if checkpoint and self._checkpoint_reason is None:
                self._checkpoint_reason = "checkpoint requested by another distributed rank"
        return stop, checkpoint

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, kwargs
        if self._stop_requested is not None and self._stop_requested():
            self.request_stop(self._external_stop_reason)
        stop, checkpoint = self._synchronize_requests()
        step = int(state.global_step)
        if stop:
            control.should_save = True
            control.should_training_stop = True
            self._stop_dispatched_step = step
            info(f"[Checkpoint] Orderly stop at optimizer step {step}: {self._stop_reason}")
        elif checkpoint:
            control.should_save = True
            self._checkpoint_dispatched_step = step
            info(f"[Checkpoint] Checkpoint-only request at optimizer step {step}: {self._checkpoint_reason}")
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, kwargs
        step = int(state.global_step)
        self._saved_steps.add(step)
        if self._checkpoint_dispatched_step == step:
            self._checkpoint_dispatched_step = None
            self._checkpoint_reason = None
        return control

    def require_resumable_checkpoint(self, output_dir: str | Path) -> Path | None:
        """Verify that an orderly stop produced a complete resume checkpoint."""
        step = self._stop_dispatched_step
        if step is None:
            return None
        if step not in self._saved_steps:
            raise RuntimeError(f"orderly stop at step {step} returned without Trainer on_save")
        checkpoint = Path(output_dir) / f"{_CHECKPOINT_PREFIX}{step}"
        if not is_valid_trainer_checkpoint(checkpoint):
            raise RuntimeError(f"orderly stop produced an incomplete resume checkpoint: {checkpoint}")
        return checkpoint

    def raise_if_terminated(self) -> None:
        """Preserve signal-style process status after a verified orderly save."""
        if self.interrupted_exit_code is not None:
            raise SystemExit(self.interrupted_exit_code)


def fork_seed(master_seed: int, name: str) -> int:
    """Return a stable 64-bit seed for one named random stream."""
    digest = hashlib.sha256(f"{int(master_seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def fork_rng(master_seed: int, name: str) -> random.Random:
    """Return an independent Python PRNG stream forked from a master seed."""
    return random.Random(fork_seed(master_seed, name))


def select_validation_rows(
    rows: Sequence,
    *,
    limit: int = 0,
    seed: int,
    policy: str = "shuffle",
) -> list:
    """Select a validation view without making input order the default policy."""
    if limit < 0:
        raise ValueError("validation row limit must be nonnegative")
    selected = list(rows)
    if policy == "shuffle":
        fork_rng(seed, "validation-selection").shuffle(selected)
    elif policy != "head":
        raise ValueError(f"unsupported validation selection policy: {policy!r}")
    return selected[:limit] if limit else selected


def _scalar_field(row: Mapping, field: str, row_index: int) -> object:
    """Resolve one dotted scalar field used in a split identity."""
    value: object = row
    for component in field.split("."):
        if not component or not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"row {row_index} lacks split field {field!r}")
        value = value[component]
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        raise ValueError(f"row {row_index} split field {field!r} must be a JSON scalar")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_index} split field {field!r} must be a JSON scalar") from error
    return value


def _canonical_split_key(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) for value in values
    )


def stratified_group_split(
    rows: Sequence[Mapping],
    *,
    selected_fraction: float,
    seed: int,
    stratify_fields: Sequence[str],
    group_fields: Sequence[str],
    balance_unit: str = "rows",
) -> dict[str, object]:
    """Split rows reproducibly while keeping provenance groups atomic.

    Every stratum must contain at least two groups so both outputs represent
    it. Within each stratum a named random stream shuffles groups. ``rows``
    chooses the shuffled prefix closest to the requested row fraction;
    ``groups`` chooses the closest group count. Returned indices retain input
    order so materialized JSONL files remain easy to audit.
    """
    if not 0.0 < selected_fraction < 1.0:
        raise ValueError("selected_fraction must be strictly between zero and one")
    if not rows:
        raise ValueError("cannot split an empty row sequence")
    stratify_fields = tuple(str(field) for field in stratify_fields)
    group_fields = tuple(str(field) for field in group_fields)
    if not stratify_fields or any(not field for field in stratify_fields):
        raise ValueError("stratify_fields must contain at least one nonempty field")
    if not group_fields or any(not field for field in group_fields):
        raise ValueError("group_fields must contain at least one nonempty field")
    if len(stratify_fields) != len(set(stratify_fields)):
        raise ValueError("stratify_fields must be unique")
    if len(group_fields) != len(set(group_fields)):
        raise ValueError("group_fields must be unique")
    if balance_unit not in {"rows", "groups"}:
        raise ValueError("balance_unit must be 'rows' or 'groups'")

    group_rows: dict[tuple[str, ...], list[int]] = defaultdict(list)
    group_strata: dict[tuple[str, ...], tuple[str, ...]] = {}
    stratum_values: dict[tuple[str, ...], tuple[object, ...]] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_index} is not a mapping")
        raw_stratum = tuple(_scalar_field(row, field, row_index) for field in stratify_fields)
        raw_group = tuple(_scalar_field(row, field, row_index) for field in group_fields)
        stratum = _canonical_split_key(raw_stratum)
        group = _canonical_split_key(raw_group)
        prior = group_strata.setdefault(group, stratum)
        if prior != stratum:
            raise ValueError(
                f"provenance group {dict(zip(group_fields, raw_group, strict=True))!r} "
                "crosses stratification cells"
            )
        stratum_values.setdefault(stratum, raw_stratum)
        group_rows[group].append(row_index)

    groups_by_stratum: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for group, stratum in group_strata.items():
        groups_by_stratum[stratum].append(group)

    selected_groups: set[tuple[str, ...]] = set()
    strata_report = []
    for stratum in sorted(groups_by_stratum):
        groups = groups_by_stratum[stratum]
        if len(groups) < 2:
            values = dict(zip(stratify_fields, stratum_values[stratum], strict=True))
            raise ValueError(f"stratum {values!r} has fewer than two provenance groups")
        rng = fork_rng(seed, "stratified-group-split:" + "\x1f".join(stratum))
        rng.shuffle(groups)
        if balance_unit == "groups":
            selected_count = round(len(groups) * selected_fraction)
            selected_count = max(1, min(len(groups) - 1, selected_count))
        else:
            target_rows = sum(len(group_rows[group]) for group in groups) * selected_fraction
            cumulative = 0
            candidates = []
            for count, group in enumerate(groups[:-1], 1):
                cumulative += len(group_rows[group])
                candidates.append(
                    (abs(cumulative - target_rows), abs(count / len(groups) - selected_fraction), count)
                )
            selected_count = min(candidates)[2]
        selected = groups[:selected_count]
        selected_groups.update(selected)
        selected_rows = sum(len(group_rows[group]) for group in selected)
        total_rows = sum(len(group_rows[group]) for group in groups)
        strata_report.append(
            {
                "values": dict(zip(stratify_fields, stratum_values[stratum], strict=True)),
                "rows": total_rows,
                "groups": len(groups),
                "selected_rows": selected_rows,
                "selected_groups": selected_count,
                "remainder_rows": total_rows - selected_rows,
                "remainder_groups": len(groups) - selected_count,
            }
        )

    selected_indices = tuple(
        index for group, indices in group_rows.items() if group in selected_groups for index in indices
    )
    selected_index_set = set(selected_indices)
    selected_indices = tuple(index for index in range(len(rows)) if index in selected_index_set)
    remainder_indices = tuple(index for index in range(len(rows)) if index not in selected_index_set)
    return {
        "selected_indices": selected_indices,
        "remainder_indices": remainder_indices,
        "selected_rows": len(selected_indices),
        "remainder_rows": len(remainder_indices),
        "selected_groups": len(selected_groups),
        "remainder_groups": len(group_rows) - len(selected_groups),
        "strata": strata_report,
    }


def fork_torch_generator(
    master_seed: int,
    name: str,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    """Return an independent torch generator forked from a master seed."""
    generator = torch.Generator(device=device)
    generator.manual_seed(fork_seed(master_seed, name))
    return generator


def label_weights_for_floor(
    labels: Sequence[str],
    floor: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Multipliers that lift every label to at least ``floor`` of one dimension's mass.

    Raising a starved label also shrinks everyone else, so multipliers guessed from the
    shortfall alone land under the target. The exact answer needs no solver while one
    dimension is being weighted: an item's weight is its label's weight, so a label's
    achieved share is its weight times its count over the total, and setting the weight
    to the target share divided by the natural share makes those cancel.

    ``weights`` supplies an existing per-item draw, making the floor a correction on
    top of it rather than a replacement for it.

    The floor policy is deliberately not equalization. Labels already above the floor
    keep their standing relative to each other; the mass to lift the starved ones is
    taken from them in proportion to what they hold.

    Exactness holds while this is the only weighted dimension. Weighting a second one
    moves these shares again, because the product no longer depends on this label alone.
    """
    counts: Counter[str] = Counter()
    if weights is None:
        counts.update(str(label) for label in labels)
    else:
        # Applied on top of an existing draw, the share to compare against the floor is
        # the mass a label already receives, not how many rows carry it. Using counts
        # here would fight whatever policy produced those weights.
        for label, weight in zip(labels, weights, strict=True):
            value = float(weight)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("weights must be finite and nonnegative")
            counts[str(label)] += value
    if not counts:
        raise ValueError("labels must be non-empty")
    floor = float(floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must be in [0, 1), got {floor}")
    if len(counts) * floor > 1.0 + 1e-12:
        raise ValueError(
            f"floor {floor} is infeasible for {len(counts)} labels; the maximum is {1.0 / len(counts):.6f}"
        )
    total = math.fsum(counts.values())
    if total <= 0.0:
        raise ValueError("labels carry no mass to redistribute")
    natural = {label: count / total for label, count in counts.items()}
    if not floor:
        return dict.fromkeys(counts, 1.0)
    raised = {label for label, share in natural.items() if share < floor}
    if not raised:
        return dict.fromkeys(counts, 1.0)
    if any(natural[label] == 0 for label in raised):
        raise ValueError("cannot give positive share to a label with zero sampling mass")
    while True:
        donors = math.fsum(share for label, share in natural.items() if label not in raised)
        scale = (1.0 - len(raised) * floor) / donors
        newly_starved = {
            label for label, share in natural.items() if label not in raised and share * scale < floor
        }
        if not newly_starved:
            break
        raised.update(newly_starved)
    target = {label: (floor if label in raised else share * scale) for label, share in natural.items()}
    return {label: target[label] / natural[label] for label in counts}


def dimension_weighted_items(
    item_labels: Sequence[Mapping[str, str]],
    label_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[list[float], dict]:
    """Per-item sampling weights as the product of each label's weight.

    An item carries one label per dimension: which file it came from, its language,
    later its domain. An unlisted label weighs 1, so a dimension nobody has an opinion
    about contributes nothing and weighting one language says nothing about the others.
    What the mix became is not left to inference: the receipt reports the share every
    label ended up with beside the share it would have had at equal weight, so a weight
    set by accident shows up as a share nobody intended.

    Multiplying is what keeps the dimensions independent. Doubling a language's weight
    doubles every item in that language wherever it came from, leaving the relative
    standing of the source files within that language untouched, and the same in
    reverse. The alternative, naming a weight per combination of source and language,
    grows as their product and buries provenance in compound bucket names.

    Weights are relative; the returned values are normalized to sum to one. The receipt
    records, per dimension, the share each label would have had at equal weight and the
    share it actually has, because a multiplier is a statement of intent and the share
    is the consequence, and only the second one is what the run trained on.
    """
    items = [dict(labels) for labels in item_labels]
    if not items:
        raise ValueError("item_labels must be non-empty")
    label_weights = {name: dict(values) for name, values in (label_weights or {}).items()}
    dimensions = sorted({name for labels in items for name in labels})
    for name in sorted(label_weights):
        if name not in dimensions:
            raise ValueError(f"no item carries a {name!r} label")
    for index, labels in enumerate(items):
        missing = [name for name in dimensions if name not in labels]
        if missing:
            raise ValueError(f"item {index} lacks labels for {missing}")
    observed = {name: sorted({labels[name] for labels in items}) for name in dimensions}
    for name, values in label_weights.items():
        unknown = sorted(set(values) - set(observed[name]))
        if unknown:
            raise ValueError(f"{name}: weights name labels with no items: {unknown}")
        for label, value in values.items():
            weight = float(value)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name}.{label}: weight must be finite and nonnegative")

    raw = []
    for labels in items:
        weight = 1.0
        for name, label in labels.items():
            weight *= float(label_weights.get(name, {}).get(label, 1.0))
        raw.append(weight)
    total = math.fsum(raw)
    if total <= 0.0:
        raise ValueError("every item weighs zero; at least one label weight must be positive")
    weights = [value / total for value in raw]

    receipt: dict = {"dimensions": {}}
    for name in dimensions:
        counts = Counter(labels[name] for labels in items)
        achieved: dict[str, float] = dict.fromkeys(observed[name], 0.0)
        for weight, labels in zip(weights, items, strict=True):
            achieved[labels[name]] += weight
        receipt["dimensions"][name] = {
            "label_weights": {
                label: float(label_weights.get(name, {}).get(label, 1.0)) for label in observed[name]
            },
            "equal_weight_share": {label: round(counts[label] / len(items), 8) for label in observed[name]},
            "achieved_share": {label: round(achieved[label], 8) for label in observed[name]},
        }
    return weights, receipt


def example_weights_from_pools(
    pool_labels: Sequence[str],
    pool_weights: Mapping[str, float],
    *,
    example_factors: Sequence[float] | None = None,
) -> list[float]:
    """Compile target pool masses into normalized per-example weights.

    Every example belongs to exactly one named pool. ``pool_weights`` assigns
    total probability mass to pools; examples divide their pool's mass equally
    unless ``example_factors`` supplies within-pool relative weights. This is
    the bridge from convenient file/source/language pools to the sampler's more
    general row-weight representation.
    """
    labels = [str(label) for label in pool_labels]
    if not labels:
        raise ValueError("pool_labels must be non-empty")
    observed = set(labels)
    declared = set(pool_weights)
    missing = sorted(observed - declared)
    unused = sorted(declared - observed)
    if missing:
        raise ValueError(f"missing weights for observed pools: {missing}")
    if unused:
        raise ValueError(f"pool weights name pools with no examples: {unused}")

    normalized_pool_weights: dict[str, float] = {}
    for name, raw_weight in pool_weights.items():
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"pool weight for {name!r} must be finite and nonnegative")
        normalized_pool_weights[str(name)] = weight
    pool_total = sum(normalized_pool_weights.values())
    if pool_total <= 0.0:
        raise ValueError("pool weights must sum to a positive value")
    normalized_pool_weights = {name: weight / pool_total for name, weight in normalized_pool_weights.items()}

    factors = [1.0] * len(labels) if example_factors is None else [float(value) for value in example_factors]
    if len(factors) != len(labels):
        raise ValueError("example_factors and pool_labels must have the same length")
    factor_totals: dict[str, float] = defaultdict(float)
    for index, (label, factor) in enumerate(zip(labels, factors)):
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError(f"example factor at index {index} must be finite and nonnegative")
        factor_totals[label] += factor
    empty = sorted(name for name, total in factor_totals.items() if total <= 0.0)
    if empty:
        raise ValueError(f"pool example factors must sum to a positive value: {empty}")
    return [
        normalized_pool_weights[label] * factor / factor_totals[label]
        for label, factor in zip(labels, factors)
    ]


def _split_length_sorted_physical(
    indices: list[int],
    *,
    lengths: Sequence[int],
    batch_size: int,
    gradient_accumulation_steps: int,
    logical_steps: int = 1,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Length-sort a sampled window without making optimizer steps monotone."""
    if not indices:
        return []
    indices.sort(key=lambda index: lengths[index], reverse=True)
    physical = [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]
    return _deal_over_steps(physical, gradient_accumulation_steps, logical_steps, rng)


def _deal_over_steps(
    physical: list[list[int]],
    gradient_accumulation_steps: int,
    logical_steps: int,
    rng: random.Random | None,
) -> list[list[int]]:
    """Deal length-ordered physical batches round-robin over optimizer steps.

    Each step receives one batch from each stretch of the length order, so its
    length mix follows the window's instead of being one length cluster.
    """
    if logical_steps <= 1 or len(physical) <= gradient_accumulation_steps:
        return physical

    steps = max(1, min(int(logical_steps), len(physical)))
    logical_groups: list[list[list[int]]] = [[] for _ in range(steps)]
    step_order = list(range(steps))
    if rng is not None:
        rng.shuffle(step_order)
        offset = rng.randrange(steps)
    else:
        offset = 0
    for rank, batch in enumerate(physical):
        step = step_order[(rank + offset) % steps]
        logical_groups[step].append(batch)
    return [batch for group in logical_groups for batch in group]


class _RemainingIndex:
    """Fenwick tree over sorted positions: count, select and remove remaining items."""

    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)
        for position in range(1, size + 1):
            self.tree[position] += 1
            parent = position + (position & -position)
            if parent <= size:
                self.tree[parent] += self.tree[position]
        self.remaining = [True] * size
        self.count = size
        self.top = 1 << size.bit_length()

    def remove(self, position: int) -> None:
        if not self.remaining[position]:
            raise ValueError(f"position {position} was already batched")
        self.remaining[position] = False
        self.count -= 1
        position += 1
        while position <= self.size:
            self.tree[position] -= 1
            position += position & -position

    def prefix(self, end: int) -> int:
        """Remaining items among sorted positions [0, end)."""
        total = 0
        while end > 0:
            total += self.tree[end]
            end -= end & -end
        return total

    def select(self, rank: int) -> int:
        """Sorted position of the remaining item with 0-based rank."""
        position, step = 0, self.top
        while step:
            nxt = position + step
            if nxt <= self.size and self.tree[nxt] <= rank:
                position = nxt
                rank -= self.tree[nxt]
            step >>= 1
        return position

    def nearest(self, anchor: int, count: int, key: Sequence[float]) -> list[int]:
        """Up to ``count`` remaining positions whose ``key`` is closest to ``anchor``'s.

        ``key`` must be nondecreasing over sorted positions. ``anchor`` itself
        is included when it remains.
        """
        below = self.prefix(anchor)
        left, right = below - 1, below
        chosen = []
        while len(chosen) < count and (left >= 0 or right < self.count):
            if left < 0:
                take_left = False
            elif right >= self.count:
                take_left = True
            else:
                left_position, right_position = self.select(left), self.select(right)
                take_left = key[anchor] - key[left_position] <= key[right_position] - key[anchor]
            if take_left:
                chosen.append(self.select(left))
                left -= 1
            else:
                chosen.append(self.select(right))
                right += 1
        return chosen


@dataclass
class BandBatchingStats:
    """How a window's draws became physical batches under band-random formation."""

    items: int = 0
    band_random_items: int = 0
    sparse_sorted_items: int = 0
    remainder_sorted_items: int = 0
    stretched_batches: int = 0
    over_budget_batches: int = 0
    duplicate_admissions: int = 0
    real_tokens: int = 0
    padded_tokens: int = 0
    spreads: list[float] | None = None

    def add(self, other: "BandBatchingStats") -> None:
        self.items += other.items
        self.band_random_items += other.band_random_items
        self.sparse_sorted_items += other.sparse_sorted_items
        self.remainder_sorted_items += other.remainder_sorted_items
        self.stretched_batches += other.stretched_batches
        self.over_budget_batches += other.over_budget_batches
        self.duplicate_admissions += other.duplicate_admissions
        self.real_tokens += other.real_tokens
        self.padded_tokens += other.padded_tokens
        self.spreads = (self.spreads or []) + (other.spreads or [])

    @property
    def padding_waste(self) -> float:
        """Share of padded token slots that are padding: 1 - real / padded."""
        return 1.0 - self.real_tokens / self.padded_tokens if self.padded_tokens else 0.0


def _band_random_physical(
    indices: list[int],
    *,
    lengths: Sequence[int],
    batch_size: int,
    rng: random.Random,
    gradient_accumulation_steps: int,
    logical_steps: int,
    padding_budget: float = 0.10,
    sparse_band_items: int | None = None,
    remainder_fraction: float = 0.01,
) -> tuple[list[list[int]], BandBatchingStats]:
    """Random physical batches whose occupants share a length band, dealt over steps.

    The window's draws are only partitioned, so every draw's marginal weight is
    untouched. A batch starts at a uniformly random remaining draw of length
    ``L`` and takes its other occupants uniformly at random from the remaining
    draws within ``L * (1 +- padding_budget / 2)``. Such a batch wastes at most
    ``padding_budget / (1 + padding_budget / 2)`` of its padded token slots, so
    ``padding_budget`` bounds padding waste; a smaller budget trades randomness
    of co-occupants for less padding. Draws whose
    band is sparse over the whole window (fewer than ``sparse_band_items``) are
    batched first with their sorted neighbours, and the last
    ``remainder_fraction`` of draws in sorted order, so neither the long tail nor
    the final stragglers form wide batches. A band that runs out of occupants
    stretches to the nearest remaining draws. Batches are then ordered by length
    and dealt round-robin over steps like sorted-window batches: a step of
    randomly assigned length clusters would look less like length-independent
    draws, not more.
    """
    stats = BandBatchingStats(items=len(indices), spreads=[])
    if not indices:
        return [], stats
    if not 0.0 < padding_budget < 2.0:
        raise ValueError("padding_budget must be in (0, 2)")
    half_width = padding_budget / 2
    sparse_band_items = 4 * batch_size if sparse_band_items is None else sparse_band_items
    order = sorted(indices, key=lambda index: lengths[index])
    sorted_lengths = [lengths[index] for index in order]
    size = len(order)

    def band(length: int) -> tuple[int, int]:
        width = half_width * length
        return bisect_left(sorted_lengths, length - width), bisect_left(sorted_lengths, length + width + 1e-9)

    remaining = _RemainingIndex(size)
    batches: list[list[int]] = []

    def emit(positions: list[int], stretched: bool = False) -> None:
        for position in positions:
            remaining.remove(position)
        members = [order[position] for position in positions]
        batch_lengths = [max(1, lengths[index]) for index in members]
        stats.spreads.append(max(batch_lengths) / min(batch_lengths))
        stats.stretched_batches += int(stretched)
        real, padded = sum(batch_lengths), max(batch_lengths) * len(batch_lengths)
        stats.real_tokens += real
        stats.padded_tokens += padded
        stats.over_budget_batches += int(1 - real / padded > padding_budget)
        batches.append(members)

    def pick_band_mates(seed: int, base: int, in_band: int) -> list[int]:
        """The seed plus random remaining band draws, avoiding repeat copies of one row.

        A heavily weighted row is drawn several times per epoch, and its copies
        share one length. Candidates are visited in random order and another
        copy of a row already in the batch is skipped while other candidates
        remain; only a band holding nothing else admits a duplicate.
        """
        chosen, rows, skipped, tried = [seed], {order[seed]}, [], set()
        seed_rank = remaining.prefix(seed) - base
        others = in_band - 1
        while len(tried) < others:
            rank = rng.randrange(others)
            if rank in tried:
                continue
            tried.add(rank)
            position = remaining.select(base + rank + (rank >= seed_rank))
            if order[position] in rows:
                skipped.append(position)
                continue
            chosen.append(position)
            rows.add(order[position])
            if len(chosen) == batch_size:
                return chosen
        stats.duplicate_admissions += batch_size - len(chosen)
        return chosen + skipped[: batch_size - len(chosen)]

    # 1. Sparse bands, in contiguous sorted runs, grouped with sorted neighbours.
    sparse = [hi - lo < sparse_band_items for lo, hi in (band(length) for length in sorted_lengths)]
    position = 0
    while position < size:
        if not sparse[position] or not remaining.remaining[position]:
            position += 1
            continue
        run = []
        while position < size and sparse[position] and len(run) < batch_size:
            if remaining.remaining[position]:
                run.append(position)
            position += 1
        if len(run) < batch_size:
            members = set(run)
            extra = [
                p
                for p in remaining.nearest(run[-1], batch_size + len(run), sorted_lengths)
                if p not in members
            ]
            run += extra[: batch_size - len(run)]
        stats.sparse_sorted_items += len(run)
        emit(run)
    # 2. The bulk: random seed, random band-mates.
    stop = max(batch_size, int(remainder_fraction * size))
    while remaining.count > stop:
        seed = remaining.select(rng.randrange(remaining.count))
        lo, hi = band(sorted_lengths[seed])
        base = remaining.prefix(lo)
        in_band = remaining.prefix(hi) - base
        if in_band >= batch_size:
            emit(pick_band_mates(seed, base, in_band))
        else:
            emit(remaining.nearest(seed, batch_size, sorted_lengths), stretched=True)
        stats.band_random_items += len(batches[-1])
    # 3. Final stragglers in sorted order.
    leftover = [remaining.select(rank) for rank in range(remaining.count)]
    for start in range(0, len(leftover), batch_size):
        chunk = leftover[start : start + batch_size]
        stats.remainder_sorted_items += len(chunk)
        emit(chunk)
    rng.shuffle(batches)  # ties in the length order below fall in random order
    batches.sort(key=lambda batch: sum(lengths[index] for index in batch) / len(batch), reverse=True)
    return _deal_over_steps(batches, gradient_accumulation_steps, logical_steps, rng), stats


DEFAULT_WEIGHTED_LENGTH_WINDOW_STEPS = 8
# band-random (default): see _band_random_physical. sorted-window: sort a
# window's draws by length and cut consecutive physical batches. It exists only
# to reproduce runs before 2026-09-27; its co-occupants are fixed nearest-length
# neighbours, and repeated draws of a heavy row share a batch (user: never use
# it for new runs). See topics/weighted-length-batch-sampling.md.
BATCH_FORMATIONS = ("sorted-window", "band-random", "random")
# How an epoch's weighted draws are made. carried (default): per-row credit
# carried across epochs, i.e. weighted draws without replacement over the run. systematic: exact per-epoch counts re-randomized every epoch (every run
# before 2026-09-27). independent: with replacement, each draw independent.
DRAW_POLICIES = ("carried", "systematic", "independent")
DEFAULT_RECURSIVE_LENGTH_BUCKET_WIDTH = 16


class WeightedLengthBatchSampler(Sampler[list[int]]):
    """Sample by per-example probability weight, then form short padded batches.

    One epoch contains ``epoch_examples`` draws (the dataset size by default).
    A randomized systematic draw gives every row its requested marginal weight
    with substantially less count variance than independent replacement draws;
    equal weights reduce exactly to one shuffled pass over the dataset. Wider
    windows are length-sorted into physical batches and round-robined back over
    optimizer steps, preserving random-like step composition.
    """

    def __init__(
        self,
        *,
        lengths: Sequence[int],
        weights: Sequence[float],
        batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
        epoch_examples: int | None = None,
        length_window_steps: int = DEFAULT_WEIGHTED_LENGTH_WINDOW_STEPS,
        batch_formation: str = "band-random",
        padding_budget: float = 0.10,
        draw_policy: str = "carried",
    ):
        if batch_formation not in BATCH_FORMATIONS:
            raise ValueError(f"batch_formation must be one of {BATCH_FORMATIONS}")
        if draw_policy not in DRAW_POLICIES:
            raise ValueError(f"draw_policy must be one of {DRAW_POLICIES}")
        self.draw_policy = draw_policy
        self._credit: list[float] | None = None
        self.batch_formation = batch_formation
        self.padding_budget = float(padding_budget)
        self.batching_stats = BandBatchingStats(spreads=[])
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if length_window_steps <= 0:
            raise ValueError("length_window_steps must be positive")
        self.lengths = [max(0, int(length)) for length in lengths]
        self.weights = [float(weight) for weight in weights]
        if not self.lengths or len(self.lengths) != len(self.weights):
            raise ValueError("lengths and weights must be non-empty and have the same length")
        for index, weight in enumerate(self.weights):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"example weight at index {index} must be finite and nonnegative")
        self.weight_total = sum(self.weights)
        if self.weight_total <= 0.0:
            raise ValueError("example weights must sum to a positive value")
        self.batch_size = int(batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.epoch_examples = len(self.weights) if epoch_examples is None else int(epoch_examples)
        if self.epoch_examples <= 0:
            raise ValueError("epoch_examples must be positive")
        self.length_window_steps = int(length_window_steps)
        self.seed = int(seed)
        self._rng = fork_rng(self.seed, "weighted-train-sampling")
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(self.epoch_examples / self.batch_size)

    def _sample_epoch_indices(self) -> list[int]:
        if self.draw_policy == "independent":
            return self._rng.choices(range(len(self.weights)), weights=self.weights, k=self.epoch_examples)
        if self.draw_policy == "carried":
            return self._carried_epoch_indices()
        # Randomize the cumulative-wheel order so adjacent corpus rows do not
        # share systematic-resampling inclusion correlations across epochs.
        order = list(range(len(self.weights)))
        self._rng.shuffle(order)
        interval = self.weight_total / self.epoch_examples
        target = self._rng.random() * interval
        position = 0
        cumulative = self.weights[order[0]]
        selected: list[int] = []
        for _ in range(self.epoch_examples):
            while target >= cumulative and position + 1 < len(order):
                position += 1
                cumulative += self.weights[order[position]]
            selected.append(order[position])
            target += interval
        self._rng.shuffle(selected)
        return selected

    def _carried_epoch_indices(self) -> list[int]:
        """Weighted draws without replacement over the run, via per-row credit.

        Each epoch adds every row's expected draw count to its credit. A row
        takes the whole part of a nonnegative credit; the remaining slots go by
        a systematic draw over the fractional parts, with a fresh random order
        and phase. Realized draws are subtracted, so credit stays in (-1, 1)
        and every row's count over any number of epochs is the floor or ceiling
        of its cumulative expected count. (Continuing one wheel pointer instead
        would repeat the same phase every epoch, since an epoch is exactly one
        turn of the wheel.)
        """
        if self._credit is None:
            self._credit = [0.0] * len(self.weights)
        scale = self.epoch_examples / self.weight_total
        credit = [c + weight * scale for c, weight in zip(self._credit, self.weights)]
        whole = [max(0, math.floor(c)) for c in credit]
        fraction = [c - k if c >= 0 else 0.0 for c, k in zip(credit, whole)]
        extra_slots = self.epoch_examples - sum(whole)
        counts = list(whole)
        if extra_slots < 0:
            # Rows overdrawn earlier hold negative credit, so the whole parts
            # owed to the others can exceed this epoch's draws. Withhold the
            # surplus by a systematic draw over those whole counts; withheld
            # rows keep their credit for later epochs.
            for row in self._systematic_draw([float(count) for count in whole], -extra_slots):
                counts[row] -= 1
            if min(counts) < 0:
                raise RuntimeError("carried credit withholding removed more draws than a row held")
            extra_slots = 0
        for row in self._systematic_draw(fraction, extra_slots) if extra_slots else []:
            counts[row] += 1
        self._credit = [c - n for c, n in zip(credit, counts)]
        selected = [row for row, n in enumerate(counts) for _ in range(n)]
        self._rng.shuffle(selected)
        return selected

    def _systematic_draw(self, masses: Sequence[float], draws: int) -> list[int]:
        """``draws`` rows by systematic sampling over ``masses`` in a fresh random order."""
        order = [row for row, mass in enumerate(masses) if mass > 0]
        self._rng.shuffle(order)
        interval = sum(masses[row] for row in order) / draws
        target = self._rng.random() * interval
        selected, cumulative, position = [], masses[order[0]], 0
        for _ in range(draws):
            while target >= cumulative and position + 1 < len(order):
                position += 1
                cumulative += masses[order[position]]
            selected.append(order[position])
            target += interval
        return selected

    def __iter__(self):
        sampled = self._sample_epoch_indices()
        if self.batch_formation == "random":
            for start in range(0, len(sampled), self.batch_size):
                yield sampled[start : start + self.batch_size]
            self.epoch += 1
            return
        logical = self.batch_size * self.gradient_accumulation_steps
        window = logical * self.length_window_steps
        for start in range(0, len(sampled), window):
            indices = sampled[start : start + window]
            if self.batch_formation == "band-random":
                batches, stats = _band_random_physical(
                    indices,
                    lengths=self.lengths,
                    batch_size=self.batch_size,
                    rng=self._rng,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    logical_steps=max(1, math.ceil(len(indices) / logical)),
                    padding_budget=self.padding_budget,
                )
                self.batching_stats.add(stats)
                yield from batches
                continue
            logical_steps = max(1, math.ceil(len(indices) / logical))
            yield from _split_length_sorted_physical(
                indices,
                lengths=self.lengths,
                batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                logical_steps=logical_steps,
                rng=self._rng,
            )
        self.epoch += 1

    def summary(self) -> str:
        nonzero = [weight for weight in self.weights if weight > 0.0]
        normalized = [weight / self.weight_total for weight in self.weights]
        effective_rows = 1.0 / sum(weight * weight for weight in normalized)
        return (
            f"examples={len(self.weights)} positive={len(nonzero)} epoch_examples={self.epoch_examples}"
            f" batch={self.batch_size}x{self.gradient_accumulation_steps}"
            f" length_window_steps={self.length_window_steps}"
            f" batch_formation={self.batch_formation}"
            f" draw_policy={self.draw_policy}"
            f" effective_weighted_rows={effective_rows:.1f}"
            f" min_positive_weight={min(nonzero):.3g} max_weight={max(self.weights):.3g}"
        )


def buffer_input_batches(iterator, dataset, config, *, stats_sink=None):
    from trainlib_input import BufferedIterator

    def batch_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sys.getsizeof(value) + sum(
                batch_bytes(key) + batch_bytes(item) for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return sys.getsizeof(value) + sum(batch_bytes(item) for item in value)
        if value is None or type(value) in (str, bytes, bool, int, float):
            return sys.getsizeof(value)
        raise TypeError(f"Unsupported buffered batch payload type: {type(value).__name__}")

    return BufferedIterator(
        iterator,
        size=batch_bytes,
        **config,
        snapshot=dataset.input_state,
        commit=dataset.commit_input_state,
        restore=dataset.restore_input_state,
        stats_sink=stats_sink,
    )


class InputTimedDataLoader(DataLoader):
    """Report host input wait as an upper bound on GPU idle time, excluding startup."""

    wait_threshold = 0.005
    wait_report_batches = 100
    pipeline_gap = "gaps/sketches/trainlib-augmentation-pipeline.md"

    def __init__(self, *args, input_pipeline=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_pipeline = input_pipeline
        self.buffer = None

    def close(self):
        if self.buffer is not None:
            self.buffer.close()
            self.pipeline_stats = self.buffer.stats()
            logging.getLogger(__name__).info("[input] %s", json.dumps(self.pipeline_stats))
            self.buffer = None

    def __iter__(self):
        self.close()
        iterator = super().__iter__()
        if self.input_pipeline:
            self.buffer = buffer_input_batches(iterator, self.dataset, self.input_pipeline)
            iterator = self.buffer
        start = None
        wait = 0.0
        count = 0
        reported = 0
        try:
            while True:
                before = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                ready = time.perf_counter()
                if start is None:
                    start = ready
                else:
                    wait += ready - before
                count += 1
                if count % self.wait_report_batches == 0:
                    self._report_input_wait(wait, ready - start, count)
                    reported = count
                yield batch
        finally:
            self.close()
            if start is not None and count != reported:
                self._report_input_wait(wait, time.perf_counter() - start, count)

    def _report_input_wait(self, wait: float, elapsed: float, batches: int) -> None:
        self.input_wait_stats = {
            "wait_seconds": wait,
            "elapsed_seconds": elapsed,
            "batches": batches,
            "wait_fraction": wait / elapsed if elapsed > 0 else 0.0,
        }
        if self.input_wait_stats["wait_fraction"] > self.wait_threshold:
            logging.getLogger(__name__).warning(
                "[train] host input wait %.3f%% exceeds 0.5%% (%d physical batches); GPU-idle upper bound; pipeline gap: %s",
                100 * self.input_wait_stats["wait_fraction"],
                batches,
                self.pipeline_gap,
            )


def capped_extra_weights(scores: list[float], budget: int, cap: int) -> list[float]:
    remaining = set(range(len(scores)))
    result = [0.0] * len(scores)
    mass = float(budget)
    while remaining:
        scale = mass / sum(scores[row] for row in remaining)
        saturated = {row for row in remaining if scores[row] * scale >= cap}
        if not saturated:
            for row in remaining:
                result[row] = scores[row] * scale
            break
        for row in saturated:
            result[row] = float(cap)
        mass -= cap * len(saturated)
        remaining -= saturated
    return result


class PrimaryExtraBatchSampler:
    """One primary per row plus a carried, continuously weighted extra-view budget."""

    def __init__(
        self,
        lengths: list[int],
        config: dict,
        *,
        seed: int,
        batch: int,
        grad_accum: int,
        formation: str,
        padding: float,
        window: int,
    ):
        self.config = config
        self.lengths = lengths
        self.seed = seed
        self.batch = batch
        self.grad_accum = grad_accum
        self.formation = formation
        self.padding = padding
        self.window = window
        self.width = config["max_extra"] + 1
        self.budget = round(len(lengths) * config["mean_extra"])
        if not 0 < self.budget <= len(lengths) * config["max_extra"]:
            raise ValueError("ASR extra-view budget rounds to zero or exceeds capacity")
        self.extra = WeightedLengthBatchSampler(
            lengths=lengths,
            weights=[1.0] * len(lengths),
            batch_size=1,
            gradient_accumulation_steps=1,
            seed=seed,
            epoch_examples=self.budget,
            length_window_steps=1,
            draw_policy="carried",
        )
        self.epoch = 0
        self.loss_ema = [0.0] * len(lengths)
        self.observed_epochs = [0] * len(lengths)
        self.pending = {}
        self.plan = None
        self.counts = None

    def __len__(self) -> int:
        return math.ceil((len(self.lengths) + self.budget) / self.batch)

    def prepare(self) -> None:
        if self.plan is not None:
            return
        active = self.config["policy"] == "loss" and self.epoch >= self.config["warm_epochs"]
        scores = [
            max(value, 1e-6) if active and count >= self.config["warm_epochs"] else 1.0
            for value, count in zip(self.loss_ema, self.observed_epochs)
        ]
        if active:
            mean = sum(scores) / len(scores)
            scores = [0.25 + 0.75 * min(value / mean, 4.0) for value in scores]
        self.extra.weights = capped_extra_weights(scores, self.budget, self.config["max_extra"])
        self.extra.weight_total = sum(self.extra.weights)
        draws = Counter(index for batch in self.extra for index in batch)
        self.counts = [1 + draws[row] for row in range(len(self.lengths))]
        if max(self.counts) > self.width:
            raise RuntimeError("ASR carried extra pulls exceeded per-row cap")
        indices = [row * self.width + view for row, count in enumerate(self.counts) for view in range(count)]
        weights = [
            float(view < self.counts[row]) for row in range(len(self.lengths)) for view in range(self.width)
        ]
        arrangement = WeightedLengthBatchSampler(
            lengths=[length for length in self.lengths for _ in range(self.width)],
            weights=weights,
            batch_size=self.batch,
            gradient_accumulation_steps=self.grad_accum,
            seed=self.seed + self.epoch,
            epoch_examples=len(indices),
            length_window_steps=self.window,
            batch_formation=self.formation,
            padding_budget=self.padding,
            draw_policy="carried",
        )
        self.plan = list(arrangement)

    def __iter__(self):
        self.prepare()
        yield from self.plan

    def observe(self, indices: list[int], losses: list[float], target_lengths: list[int]) -> None:
        for index, loss, length in zip(indices, losses, target_lengths):
            row, view = divmod(index, self.width)
            if view == 0:
                if row in self.pending:
                    raise RuntimeError("ASR primary row was observed twice in an epoch")
                self.pending[row] = loss / max(1, length)

    def finish_epoch(self) -> None:
        if len(self.pending) == len(self.lengths):
            for row, value in self.pending.items():
                count = self.observed_epochs[row]
                self.loss_ema[row] = (
                    self.config["ema"] * self.loss_ema[row] + (1 - self.config["ema"]) * value
                    if count
                    else value
                )
                self.observed_epochs[row] += 1
        self.pending = {}
        self.plan = self.counts = None
        self.epoch += 1

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "loss_ema": self.loss_ema,
            "observed_epochs": self.observed_epochs,
            "pending": self.pending,
            "plan": self.plan,
            "counts": self.counts,
            "extra_rng": self.extra._rng.getstate(),
            "extra_credit": self.extra._credit,
            "extra_epoch": self.extra.epoch,
        }

    def load_state_dict(self, state: dict, *, completed_epochs: int | None = None) -> None:
        for name in ("epoch", "loss_ema", "observed_epochs", "plan", "counts"):
            setattr(self, name, state[name])
        self.pending = {int(row): value for row, value in state["pending"].items()}
        version, values, gaussian = state["extra_rng"]
        self.extra._rng.setstate((version, tuple(values), gaussian))
        self.extra._credit = state["extra_credit"]
        self.extra.epoch = state["extra_epoch"]
        if completed_epochs is not None and self.epoch != completed_epochs:
            if self.epoch + 1 != completed_epochs or len(self.pending) != len(self.lengths):
                raise ValueError("Primary/extra sampling state disagrees with completed training epochs")
            self.finish_epoch()

    def summary(self) -> str:
        return f"rows={len(self.lengths)} primary=1 extra_budget={self.budget} max_extra={self.config['max_extra']} order={self.config['order']} policy={self.config['policy']}"


# Batch-schedule quality metrics. Every metric is "lower is better", so a
# configuration's score is a weighted sum. See
# topics/weighted-length-batch-sampling.md § Measuring a batch schedule.
BATCH_SCHEDULE_METRICS = (
    "padding_waste",
    "compute_inflation",
    "duplicate_row_batches",
    "repeated_mates",
    "step_length_tv",
    "step_mass_cv",
    "key_share_error",
    "step_key_share_sd",
    "steps_missing_a_key",
    "row_exposure_cv",
)


def step_length_representation_tv(
    lengths: Sequence[int], steps: Sequence[Sequence[int]], *, buckets: int = 4
) -> float:
    """Mean total variation between each step's length-bucket shares and the schedule's.

    Bucket edges are the schedule's length quantiles, so every bucket holds
    about ``1 / buckets`` of all draws; 0 means every step mirrors the whole
    schedule's length mix.
    """
    observed = sorted(lengths[index] for step in steps for index in step)
    edges = [observed[min(len(observed) - 1, len(observed) * i // buckets)] for i in range(1, buckets)]

    def bucket(value: int) -> int:
        return next((i for i, edge in enumerate(edges) if value <= edge), len(edges))

    totals = Counter(bucket(lengths[index]) for step in steps for index in step)
    total = sum(totals.values())
    expected = {b: totals[b] / total for b in range(buckets)}
    tvs = []
    for step in steps:
        if not step:
            continue
        counts = Counter(bucket(lengths[index]) for index in step)
        tvs.append(0.5 * sum(abs(counts[b] / len(step) - expected[b]) for b in range(buckets)))
    return sum(tvs) / len(tvs)


def batch_schedule_metrics(
    batches: Sequence[Sequence[int]],
    *,
    lengths: Sequence[int],
    gradient_accumulation_steps: int,
    weights: Sequence[float] | None = None,
    keys: Sequence[str] | None = None,
) -> dict[str, float]:
    """Measure a multi-epoch physical-batch schedule; every value is lower-is-better.

    - ``padding_waste``: share of padded token slots that are padding.
    - ``compute_inflation``: padded token slots per real token.
    - ``duplicate_row_batches``: share of physical batches holding one row twice.
    - ``repeated_mates``: for rows batched at least twice, the mean share of a
      later appearance's batch-mates already seen with that row (0 when
      co-residency is fresh every time).
    - ``step_length_tv``: ``step_length_representation_tv`` over quartiles.
    - ``step_mass_cv``: coefficient of variation of a step's total length.
    - ``key_share_error``: with ``keys`` and ``weights``, the largest gap
      between a key's share of draws and its share of weight.
    - ``step_key_share_sd``: with ``keys``, the mean over keys of the standard
      deviation of a step's share of that key.
    - ``steps_missing_a_key``: with ``keys``, the share of steps lacking at
      least one key (1 - stream diversity).
    - ``row_exposure_cv``: with ``weights``, the coefficient of variation of
      each row's draw count over its expected count, among rows expected at
      least once (0 when every row is drawn exactly as often as its weight).
    """
    batches = [list(batch) for batch in batches]
    if not batches:
        raise ValueError("an empty schedule has no metrics")
    real = padded = duplicates = 0
    seen_mates: dict[int, set[int]] = defaultdict(set)
    repeat_shares: dict[int, list[float]] = defaultdict(list)
    for batch in batches:
        batch_lengths = [max(1, int(lengths[index])) for index in batch]
        real += sum(batch_lengths)
        padded += max(batch_lengths) * len(batch_lengths)
        rows = Counter(batch)
        duplicates += any(count > 1 for count in rows.values())
        for row in rows:
            mates = set(rows) - {row}
            if row in seen_mates:
                repeat_shares[row].append(len(mates & seen_mates[row]) / max(1, len(mates)))
            seen_mates[row] |= mates
    steps = [
        [index for batch in batches[start : start + gradient_accumulation_steps] for index in batch]
        for start in range(0, len(batches), gradient_accumulation_steps)
    ]
    totals = [sum(int(lengths[index]) for index in step) for step in steps]
    mean_total = sum(totals) / len(totals)
    metrics = {
        "padding_waste": 1.0 - real / padded,
        "compute_inflation": padded / real,
        "duplicate_row_batches": duplicates / len(batches),
        "repeated_mates": (
            sum(sum(shares) / len(shares) for shares in repeat_shares.values()) / len(repeat_shares)
            if repeat_shares
            else 0.0
        ),
        "step_length_tv": step_length_representation_tv(lengths, steps),
        "step_mass_cv": (sum((t - mean_total) ** 2 for t in totals) / len(totals)) ** 0.5 / mean_total,
        "key_share_error": 0.0,
        "step_key_share_sd": 0.0,
        "steps_missing_a_key": 0.0,
        "row_exposure_cv": 0.0,
    }
    usage = Counter(index for batch in batches for index in batch)
    if weights is not None:
        weight_total = sum(weights)
        drawn_total = sum(usage.values())
        ratios = [
            usage[row] / (drawn_total * weight / weight_total)
            for row, weight in enumerate(weights)
            if drawn_total * weight / weight_total >= 1
        ]
        if ratios:
            mean = sum(ratios) / len(ratios)
            metrics["row_exposure_cv"] = (sum((r - mean) ** 2 for r in ratios) / len(ratios)) ** 0.5 / mean
    if keys is not None:
        names = sorted(set(keys))
        draws = Counter(keys[index] for step in steps for index in step)
        drawn = sum(draws.values())
        if weights is not None:
            mass = defaultdict(float)
            for key, weight in zip(keys, weights, strict=True):
                mass[key] += weight
            total = sum(mass.values())
            metrics["key_share_error"] = max(abs(draws[name] / drawn - mass[name] / total) for name in names)
        spreads = []
        for name in names:
            shares = [sum(keys[index] == name for index in step) / len(step) for step in steps]
            mean = sum(shares) / len(shares)
            spreads.append((sum((s - mean) ** 2 for s in shares) / len(shares)) ** 0.5)
        metrics["step_key_share_sd"] = sum(spreads) / len(spreads)
        metrics["steps_missing_a_key"] = sum(
            {keys[index] for index in step} != set(names) for step in steps
        ) / len(steps)
    return metrics


def exposure_gap_diagnostics(
    batches: Sequence[Sequence[int]],
    *,
    gradient_accumulation_steps: int,
    weights: Sequence[float],
) -> dict[str, float]:
    """How repeat draws of a row are spaced over optimizer steps, against independent draws.

    Reported, not scored: an exact systematic draw deliberately departs from
    independent with-replacement sampling. Under independent draws a row with
    per-step inclusion probability ``q`` (``1 - (1 - p)^draws_per_step`` for
    per-draw probability ``p``) has geometric step gaps with mean ``1/q`` and
    coefficient of variation ``sqrt(1 - q)``, and Poisson-like counts.

    - ``step_duplicate_rate``: share of steps holding some row more than once.
    - ``immediate_repeat_ratio``: observed gaps of exactly one step over their
      geometric expectation ``sum q`` (1 = independent; above 1 = clumped).
    - ``gap_cv_ratio``: mean over rows seen 3+ times of gap CV over
      ``sqrt(1 - q)`` (1 = independent; below 1 = more regular).
    - ``count_dispersion``: variance over mean of per-row counts divided by
      the binomial ``1 - p`` expectation, among rows expected at least once
      (1 = independent; below 1 = more even exposure).
    """
    steps = [
        [index for batch in batches[start : start + gradient_accumulation_steps] for index in batch]
        for start in range(0, len(batches), gradient_accumulation_steps)
    ]
    per_step = len(steps[0])
    total_weight = sum(weights)
    appearances: dict[int, list[int]] = defaultdict(list)
    duplicate_steps = 0
    for number, step in enumerate(steps):
        counts = Counter(step)
        duplicate_steps += any(count > 1 for count in counts.values())
        for row in counts:
            appearances[row].append(number)
    ones = expected_ones = 0.0
    cv_ratios = []
    for row, seen in appearances.items():
        p = weights[row] / total_weight
        q = 1 - (1 - p) ** per_step
        gaps = [b - a for a, b in zip(seen, seen[1:])]
        ones += sum(gap == 1 for gap in gaps)
        expected_ones += q * len(gaps)
        if len(gaps) >= 2 and q < 1:
            mean = sum(gaps) / len(gaps)
            sd = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
            cv_ratios.append((sd / mean) / (1 - q) ** 0.5)
    draws = sum(len(step) for step in steps)
    usage = Counter(index for step in steps for index in step)
    dispersion = []
    for row, weight in enumerate(weights):
        p = weight / total_weight
        expected = draws * p
        if expected >= 1:
            dispersion.append(((usage[row] - expected) ** 2, expected * (1 - p)))
    return {
        "step_duplicate_rate": duplicate_steps / len(steps),
        "immediate_repeat_ratio": ones / expected_ones if expected_ones else 0.0,
        "gap_cv_ratio": sum(cv_ratios) / len(cv_ratios) if cv_ratios else 0.0,
        "count_dispersion": sum(sq for sq, _ in dispersion) / sum(var for _, var in dispersion)
        if dispersion
        else 0.0,
    }


def schedule_score(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Weighted sum of lower-is-better schedule metrics; unknown metric names are an error."""
    unknown = set(weights) - set(BATCH_SCHEDULE_METRICS)
    if unknown:
        raise ValueError(f"unknown schedule metrics: {sorted(unknown)}")
    return sum(weight * metrics[name] for name, weight in weights.items())


def sweep_batch_schedules(
    make_sampler: Callable[[Mapping[str, object]], Sampler],
    configurations: Sequence[Mapping[str, object]],
    *,
    epochs: int,
    lengths: Sequence[int],
    gradient_accumulation_steps: int,
    weights: Sequence[float] | None = None,
    keys: Sequence[str] | None = None,
    score_weights: Mapping[str, float] | None = None,
) -> list[tuple[Mapping[str, object], dict[str, float], float | None]]:
    """Run each configuration's sampler for ``epochs`` and measure its schedule.

    Returns ``(configuration, metrics, score)`` per configuration, best score
    first when ``score_weights`` is given, otherwise in input order.
    """
    results = []
    for configuration in configurations:
        sampler = make_sampler(configuration)
        batches = [batch for _ in range(epochs) for batch in sampler]
        metrics = batch_schedule_metrics(
            batches,
            lengths=lengths,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weights=weights,
            keys=keys,
        )
        score = None if score_weights is None else schedule_score(metrics, score_weights)
        results.append((configuration, metrics, score))
    if score_weights is not None:
        results.sort(key=lambda result: result[2])
    return results


@dataclass(frozen=True)
class _WeightedLengthBucket:
    indices: tuple[int, ...]
    weights: tuple[float, ...]
    cumulative_weights: tuple[float, ...]
    total_weight: float


class RecursiveWeightedBatchSampler(Sampler[list[tuple[int, bool]]]):
    """Recursively sample pool, batch variant, length bucket, then rows.

    Pools and rows are sampled with replacement across physical batches.  A
    physical batch first selects one pool by its aggregate row weight, then a
    boolean variant from that pool's configured probability.  Rows are drawn
    without duplication *within* that batch from one length bucket and the
    nearest nonempty buckets as needed.  Every emitted logical step is full;
    ``epoch_examples`` is rounded up to the next physical-batch x gradient-
    accumulation boundary.

    The tuple index is deliberately generic: the dataset decides what the
    boolean variant means.  PII training uses it for an MLM rather than tagging
    view of the same stored row.
    """

    def __init__(
        self,
        *,
        lengths: Sequence[int],
        weights: Sequence[float],
        pool_keys: Sequence[str],
        batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
        epoch_examples: int | None = None,
        length_bucket_width: int = DEFAULT_RECURSIVE_LENGTH_BUCKET_WIDTH,
        default_variant_probability: float = 0.0,
        variant_probability_by_pool: Mapping[str, float] | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if length_bucket_width < 0:
            raise ValueError("length_bucket_width must be nonnegative")
        self.lengths = [max(0, int(length)) for length in lengths]
        self.weights = [float(weight) for weight in weights]
        self.pool_keys = [str(key) for key in pool_keys]
        if not self.lengths or not (len(self.lengths) == len(self.weights) == len(self.pool_keys)):
            raise ValueError("lengths, weights, and pool_keys must be non-empty and aligned")
        for index, weight in enumerate(self.weights):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"example weight at index {index} must be finite and nonnegative")
        if sum(self.weights) <= 0.0:
            raise ValueError("example weights must sum to a positive value")
        self.batch_size = int(batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        requested_examples = len(self.weights) if epoch_examples is None else int(epoch_examples)
        if requested_examples <= 0:
            raise ValueError("epoch_examples must be positive")
        self.requested_epoch_examples = requested_examples
        logical_batch_size = self.batch_size * self.gradient_accumulation_steps
        self.logical_steps = math.ceil(requested_examples / logical_batch_size)
        self.epoch_examples = self.logical_steps * logical_batch_size
        self.length_bucket_width = int(length_bucket_width)
        self.default_variant_probability = self._probability(
            default_variant_probability,
            "default_variant_probability",
        )
        overrides = {
            str(pool): self._probability(probability, f"variant probability for {pool!r}")
            for pool, probability in (variant_probability_by_pool or {}).items()
        }
        self.seed = int(seed)
        self._rng = fork_rng(self.seed, "recursive-weighted-train-sampling")
        self.epoch = 0
        self.last_epoch_statistics: dict[str, object] | None = None

        pool_rows: dict[str, list[int]] = defaultdict(list)
        for index, (pool, weight) in enumerate(zip(self.pool_keys, self.weights, strict=True)):
            if weight > 0.0:
                pool_rows[pool].append(index)
        if not pool_rows:
            raise ValueError("recursive sampler has no positive-weight rows")
        unknown_overrides = sorted(set(overrides) - set(pool_rows))
        if unknown_overrides:
            raise ValueError(f"variant probabilities name pools with no positive rows: {unknown_overrides}")
        self.variant_probability_by_pool = {
            pool: overrides.get(pool, self.default_variant_probability) for pool in pool_rows
        }

        self._pool_names = tuple(sorted(pool_rows))
        pool_masses = [sum(self.weights[index] for index in pool_rows[pool]) for pool in self._pool_names]
        self._pool_cumulative = self._cumulative(pool_masses)
        self._pool_total = self._pool_cumulative[-1]
        self.pool_probabilities = {
            pool: mass / self._pool_total for pool, mass in zip(self._pool_names, pool_masses, strict=True)
        }
        self._pool_buckets = {pool: self._build_pool_buckets(indices) for pool, indices in pool_rows.items()}
        self.target_bucket_shares_by_pool = {
            pool: {
                key: bucket.total_weight / sum(item.total_weight for item in buckets.values())
                for key, bucket in sorted(buckets.items())
            }
            for pool, buckets in self._pool_buckets.items()
        }

    @staticmethod
    def _probability(value: float, name: str) -> float:
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return probability

    @staticmethod
    def _cumulative(values: Sequence[float]) -> tuple[float, ...]:
        cumulative = []
        total = 0.0
        for value in values:
            total += float(value)
            cumulative.append(total)
        return tuple(cumulative)

    def _bucket_for_length(self, length: int) -> int:
        if not self.length_bucket_width:
            return 0
        return (max(0, int(length)) + self.length_bucket_width // 2) // self.length_bucket_width

    def _build_pool_buckets(self, indices: Sequence[int]) -> dict[int, _WeightedLengthBucket]:
        members: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            members[self._bucket_for_length(self.lengths[index])].append(index)
        buckets = {}
        for key, bucket_indices in members.items():
            bucket_weights = tuple(self.weights[index] for index in bucket_indices)
            cumulative = self._cumulative(bucket_weights)
            buckets[key] = _WeightedLengthBucket(
                indices=tuple(bucket_indices),
                weights=bucket_weights,
                cumulative_weights=cumulative,
                total_weight=cumulative[-1],
            )
        return buckets

    def __len__(self) -> int:
        return self.logical_steps * self.gradient_accumulation_steps

    def _weighted_choice(self, values: Sequence, cumulative: Sequence[float], total: float):
        position = bisect_left(cumulative, self._rng.random() * total)
        return values[min(position, len(values) - 1)]

    def _sample_pool(self) -> str:
        return self._weighted_choice(self._pool_names, self._pool_cumulative, self._pool_total)

    def _sample_unique_bucket_rows(
        self,
        bucket: _WeightedLengthBucket,
        count: int,
        excluded: set[int],
    ) -> list[int]:
        available = len(bucket.indices) - sum(index in excluded for index in bucket.indices)
        count = min(int(count), available)
        if count <= 0:
            return []
        if count == available:
            selected = [index for index in bucket.indices if index not in excluded]
            self._rng.shuffle(selected)
            return selected

        selected: list[int] = []
        selected_set: set[int] = set()
        attempts = 0
        attempt_limit = max(32, count * 16)
        while len(selected) < count and attempts < attempt_limit:
            index = self._weighted_choice(
                bucket.indices,
                bucket.cumulative_weights,
                bucket.total_weight,
            )
            attempts += 1
            if index in excluded or index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
        if len(selected) == count:
            return selected

        # Highly concentrated weights can make rejection sampling stall.
        # Exponential keys provide a weighted-without-replacement fallback for
        # the still-unselected rows.
        remaining = [
            (-math.log(max(self._rng.random(), 1e-300)) / weight, index)
            for index, weight in zip(bucket.indices, bucket.weights, strict=True)
            if index not in excluded and index not in selected_set
        ]
        remaining.sort()
        selected.extend(index for _key, index in remaining[: count - len(selected)])
        return selected

    def _sample_length_grouped_batch(self, pool: str) -> tuple[list[int], int]:
        buckets = self._pool_buckets[pool]
        keys = tuple(sorted(buckets))
        masses = [buckets[key].total_weight for key in keys]
        cumulative = self._cumulative(masses)
        anchor = self._weighted_choice(keys, cumulative, cumulative[-1])
        by_distance: dict[int, list[int]] = defaultdict(list)
        for key in keys:
            by_distance[abs(key - anchor)].append(key)
        ordered_keys = []
        for distance in sorted(by_distance):
            tied = by_distance[distance]
            self._rng.shuffle(tied)
            ordered_keys.extend(tied)

        selected: list[int] = []
        selected_set: set[int] = set()
        for key in ordered_keys:
            additions = self._sample_unique_bucket_rows(
                buckets[key],
                self.batch_size - len(selected),
                selected_set,
            )
            selected.extend(additions)
            selected_set.update(additions)
            if len(selected) == self.batch_size:
                break
        if len(selected) != self.batch_size:
            raise ValueError(
                f"pool {pool!r} has only {len(selected)} unique positive-weight rows; "
                f"physical batch size is {self.batch_size}"
            )
        return selected, anchor

    def _bucket_exposure_statistics(
        self,
        emitted_bucket_entries: Mapping[str, Counter[int]],
    ) -> dict[str, object]:
        """Compare packing-time bucket exposure with ideal weighted row draws."""
        target_by_pool = {}
        realized_by_pool = {}
        error_by_pool = {}
        total_variation_by_pool = {}
        maximum_absolute_error = 0.0
        for pool in self._pool_names:
            target = self.target_bucket_shares_by_pool[pool]
            emitted = emitted_bucket_entries.get(pool, Counter())
            emitted_total = sum(emitted.values())
            realized = {key: emitted.get(key, 0) / emitted_total if emitted_total else 0.0 for key in target}
            error = {key: realized[key] - target[key] for key in target}
            target_by_pool[pool] = {str(key): value for key, value in target.items()}
            realized_by_pool[pool] = {str(key): value for key, value in realized.items()}
            error_by_pool[pool] = {str(key): value for key, value in error.items()}
            if emitted_total:
                total_variation_by_pool[pool] = 0.5 * sum(abs(value) for value in error.values())
                maximum_absolute_error = max(
                    maximum_absolute_error,
                    *(abs(value) for value in error.values()),
                )
            else:
                total_variation_by_pool[pool] = None
        return {
            "reference": "independent_weighted_rows_within_selected_pool",
            "target_bucket_shares_by_pool": target_by_pool,
            "realized_bucket_shares_by_pool": realized_by_pool,
            "realized_minus_target_by_pool": error_by_pool,
            "total_variation_by_pool": total_variation_by_pool,
            "maximum_absolute_bucket_share_error": maximum_absolute_error,
        }

    def __iter__(self):
        pool_batches: Counter[str] = Counter()
        variant_batches: Counter[str] = Counter()
        anchor_buckets: Counter[str] = Counter()
        emitted_bucket_entries: dict[str, Counter[int]] = defaultdict(Counter)
        for _ in range(len(self)):
            pool = self._sample_pool()
            variant = self._rng.random() < self.variant_probability_by_pool[pool]
            indices, anchor = self._sample_length_grouped_batch(pool)
            pool_batches[pool] += 1
            if variant:
                variant_batches[pool] += 1
            anchor_buckets[f"{pool}:{anchor}"] += 1
            for index in indices:
                emitted_bucket_entries[pool][self._bucket_for_length(self.lengths[index])] += 1
            yield [(index, variant) for index in indices]
        self.last_epoch_statistics = {
            "epoch": self.epoch,
            "physical_batches": len(self),
            "sampled_examples": self.epoch_examples,
            "pool_batches": dict(sorted(pool_batches.items())),
            "variant_batches": dict(sorted(variant_batches.items())),
            "anchor_buckets": dict(sorted(anchor_buckets.items())),
            "emitted_bucket_entries_by_pool": {
                pool: {str(key): value for key, value in sorted(counts.items())}
                for pool, counts in sorted(emitted_bucket_entries.items())
            },
            "packing_exposure": self._bucket_exposure_statistics(emitted_bucket_entries),
        }
        info("RECURSIVE-SAMPLER-EPOCH: " + json.dumps(self.last_epoch_statistics, sort_keys=True))
        self.epoch += 1

    def summary(self) -> str:
        expected_variant = sum(
            self.pool_probabilities[pool] * self.variant_probability_by_pool[pool]
            for pool in self._pool_names
        )
        return (
            f"examples={len(self.weights)} requested_epoch_examples={self.requested_epoch_examples}"
            f" sampled_epoch_examples={self.epoch_examples} batch={self.batch_size}x"
            f"{self.gradient_accumulation_steps} pools={dict(self.pool_probabilities)}"
            f" variant_probabilities={self.variant_probability_by_pool}"
            f" expected_variant_batch_fraction={expected_variant:.6f}"
            f" length_binning={'disabled' if not self.length_bucket_width else self.length_bucket_width}"
        )


class SophiaG(torch.optim.Optimizer):
    """Sophia-G: Second-Order Clipped Stochastic Optimization (Liu et al. 2023).

    Uses Gauss-Newton-Bartlett (GNB) diagonal Hessian estimator with EMA.
    The Hessian estimate ĥ is updated every `k` optimizer steps via a separate
    forward+backward pass with sampled pseudo-labels.  Between Hessian updates
    the stale ĥ is reused (fine in practice for k≤10).

    Update rule (per parameter p):
        m_t  = β1·m_{t-1} + (1-β1)·g_t          # first moment
        ĥ_t  = β2·ĥ_{t-1} + (1-β2)·g_hat²        # diagonal Hessian EMA
                                                    # (g_hat from GNB pass)
        ratio = clip(m_t / (ρ·ĥ_t + ε), -1, 1)
        p    ← p - lr·ratio                        # clipped update

    Recommended for LoRA fine-tuning:
        lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=0.1, k=10
    (Same lr as AdamW; rho smaller than pretraining default of 0.04.)
    """

    OPTIM_NAME = "sophia_g"

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.01,
        weight_decay: float = 0.1,
        k: int = 10,
        eps: float = 1e-12,
    ):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
        self.k = k
        # Incremented in training_step; hessian update fires when this hits k
        self._optimizer_steps = 0

    @torch.no_grad()
    def update_hessian(self) -> None:
        """Update diagonal Hessian estimates from current p.grad (GNB pass gradients).

        Must be called after backward() on the GNB pseudo-loss, BEFORE zero_grad().
        """
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p.data)
                # EMA of squared gradient: ĥ ← β2·ĥ + (1-β2)·g²
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, _ = group["betas"]
            rho = group["rho"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["step"] = 0
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p.data)

                state["step"] += 1
                g = p.grad.data

                # First moment
                state["exp_avg"].mul_(beta1).add_(g, alpha=1.0 - beta1)

                # Decoupled weight decay
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                # Sophia clipped update
                h = state["hessian"].clamp(min=eps)
                ratio = (state["exp_avg"] / (rho * h)).clamp_(-1.0, 1.0)
                p.data.add_(ratio, alpha=-lr)

        self._optimizer_steps += 1
        return loss

    @property
    def needs_hessian_update(self) -> bool:
        """True on the step when a GNB hessian update should be computed."""
        return self._optimizer_steps % self.k == 0


class LengthBucketedBatchSampler(Sampler[list[int]]):
    """Random bucket-anchor order with locally homogeneous physical batches.

    Dataset-agnostic: construct with a precomputed per-example `lengths` list
    (index i is the token length used to bucket example i). The caller owns
    how lengths are derived from its dataset.
    """

    def __init__(
        self,
        *,
        lengths: Sequence[int],
        batch_size: int,
        bucket_width: int,
        fill: str,
        seed: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if bucket_width <= 0:
            raise ValueError("bucket_width must be positive")
        if fill not in {"partial", "nearest"}:
            raise ValueError(f"unsupported length-bucket fill mode {fill!r}")
        self.batch_size = int(batch_size)
        self.bucket_width = int(bucket_width)
        self.fill = fill
        self.seed = int(seed)
        self._rng = fork_rng(self.seed, "train-shuffle")
        self.lengths = list(lengths)
        self.epoch = 0
        self._cached_epoch: int | None = None
        self._cached_batches: list[list[int]] | None = None

    def __len__(self) -> int:
        return len(self._batches_for_epoch(self.epoch))

    def _bucket_for_length(self, length: int) -> int:
        # Width=16 gives buckets centered roughly every 16 tokens, i.e. +/-8.
        return (max(0, int(length)) + self.bucket_width // 2) // self.bucket_width

    def _take_from_bucket(self, buckets: dict[int, list[int]], key: int, batch: list[int]) -> None:
        while len(batch) < self.batch_size and buckets.get(key):
            batch.append(buckets[key].pop())

    def _take_immediate_neighbors(
        self,
        buckets: dict[int, list[int]],
        anchor: int,
        batch: list[int],
        rng: random.Random,
    ) -> None:
        keys = [key for key in (anchor - 1, anchor + 1) if buckets.get(key)]
        # If both neighbors exist, randomize the side so partial fill does not
        # always prefer shorter examples.
        rng.shuffle(keys)
        for key in keys:
            self._take_from_bucket(buckets, key, batch)
            if len(batch) >= self.batch_size:
                return

    def _nearest_nonempty_keys(self, buckets: dict[int, list[int]], anchor: int, rng: random.Random):
        nonempty = [key for key, values in buckets.items() if values and key != anchor]
        by_distance: dict[int, list[int]] = defaultdict(list)
        for key in nonempty:
            by_distance[abs(key - anchor)].append(key)
        for distance in sorted(by_distance):
            keys = by_distance[distance]
            rng.shuffle(keys)
            yield from keys

    def _make_batches(self, epoch: int) -> list[list[int]]:
        rng = self._rng
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, length in enumerate(self.lengths):
            buckets[self._bucket_for_length(length)].append(idx)
        for values in buckets.values():
            rng.shuffle(values)

        batches: list[list[int]] = []
        remaining = len(self.lengths)
        while remaining > 0:
            keys = [key for key, values in buckets.items() if values]
            draw = rng.randrange(remaining)
            cumulative = 0
            anchor = keys[-1]
            for key in keys:
                cumulative += len(buckets[key])
                if draw < cumulative:
                    anchor = key
                    break

            batch: list[int] = []
            self._take_from_bucket(buckets, anchor, batch)
            if self.fill == "partial" and len(batch) < self.batch_size:
                self._take_immediate_neighbors(buckets, anchor, batch, rng)
            if self.fill == "nearest" and len(batch) < self.batch_size:
                for key in self._nearest_nonempty_keys(buckets, anchor, rng):
                    self._take_from_bucket(buckets, key, batch)
                    if len(batch) >= self.batch_size:
                        break
            remaining -= len(batch)
            batches.append(batch)
        return batches

    def _batches_for_epoch(self, epoch: int) -> list[list[int]]:
        if self._cached_epoch != epoch or self._cached_batches is None:
            self._cached_batches = self._make_batches(epoch)
            self._cached_epoch = epoch
        return self._cached_batches

    def __iter__(self):
        batches = self._batches_for_epoch(self.epoch)
        self.epoch += 1
        self._cached_epoch = None
        self._cached_batches = None
        yield from batches

    def summary(self) -> str:
        bucket_counts = Counter(self._bucket_for_length(length) for length in self.lengths)
        return (
            f"examples={len(self.lengths)} batch={self.batch_size}"
            f" bucket_width={self.bucket_width} fill={self.fill} buckets={len(bucket_counts)}"
            f" largest_bucket={max(bucket_counts.values()) if bucket_counts else 0}"
            f" sampler_batches={len(self)}"
        )


@dataclass(frozen=True)
class MixStream:
    """One data stream to mix: examples [start, start+count) at a target weight.

    Weights are relative (the sampler normalizes). The largest-count stream is the
    epoch anchor, consumed without replacement (it defines epoch length); the rest
    are drawn with replacement (replay pools). See
    topics/logical-batch-stream-mixing.md."""

    start: int
    count: int
    weight: float
    name: str = ""


DEFAULT_STREAM_STRICT_EVERY = 8


class LogicalMixBatchSampler(Sampler[list[int]]):
    """Emit physical batches whose gradient (logical) windows hit a target mix.

    Two constructions:
      - 2-stream back-compat: primary_count / generic_count / generic_fraction.
      - N-stream: streams=[MixStream(...), ...].

    Dataset-agnostic: a precomputed per-example `lengths` list plus per-stream
    index ranges. Data-stream representation is the priority; length-sorting the
    logical window (`_split_physical`) is a soft efficiency pass that may mix
    streams within a physical batch. See topics/logical-batch-stream-mixing.md.
    """

    def __init__(
        self,
        *,
        lengths: Sequence[int],
        batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
        primary_count: int | None = None,
        generic_count: int | None = None,
        generic_fraction: float | None = None,
        streams: Sequence[MixStream] | None = None,
        stream_strict_every: int = DEFAULT_STREAM_STRICT_EVERY,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.batch_size = int(batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.seed = int(seed)
        self._rng = fork_rng(self.seed, "train-shuffle")
        self.lengths = list(lengths)
        self.epoch = 0
        self._cached_epoch: int | None = None
        self._cached_batches: list[list[int]] | None = None
        logical = self.batch_size * self.gradient_accumulation_steps
        if streams is not None:
            self._nstream = True
            if stream_strict_every <= 0:
                raise ValueError("stream_strict_every must be positive")
            self.stream_strict_every = int(stream_strict_every)
            self._init_streams(list(streams))
            return
        self._nstream = False
        if primary_count is None or generic_count is None or generic_fraction is None:
            raise ValueError("LogicalMixBatchSampler needs streams=... or the 2-stream kwargs")
        if primary_count <= 0:
            raise ValueError("--generic-mix-frac requires at least one primary example")
        if generic_count <= 0:
            raise ValueError("--generic-mix-frac requires at least one generic example")
        if not (0.0 < generic_fraction < 1.0):
            raise ValueError("--generic-mix-frac must be between 0 and 1")
        self.primary_count = int(primary_count)
        self.generic_count = int(generic_count)
        self.generic_fraction = float(generic_fraction)
        self.generic_per_logical = max(1, int(round(logical * self.generic_fraction)))
        if self.generic_per_logical >= logical:
            raise ValueError(
                "--generic-mix-frac leaves no primary examples in a logical batch; lower the fraction"
            )
        self.primary_per_logical = logical - self.generic_per_logical

    def _init_streams(self, streams: list[MixStream]) -> None:
        if not streams:
            raise ValueError("streams must be non-empty")
        weight_total = sum(max(0.0, s.weight) for s in streams)
        if weight_total <= 0.0:
            raise ValueError("stream weights must sum to > 0")
        norm: list[MixStream] = []
        for i, s in enumerate(streams):
            if s.count <= 0:
                raise ValueError(f"stream {s.name or i} has no examples")
            norm.append(
                MixStream(int(s.start), int(s.count), max(0.0, s.weight) / weight_total, s.name or f"s{i}")
            )
        self.streams = norm
        # anchor = largest stream, consumed without replacement -> defines epoch length
        self.anchor = max(range(len(norm)), key=lambda i: norm[i].count)
        # Persist replay decks + deficit ACROSS epochs so with-replacement coverage
        # is continuous: a replay stream drawn a fraction of its size per epoch would
        # otherwise re-sample a fresh subset each epoch (binomial coverage variance).
        # A persistent deck cycles every item before repeating, near-uniform coverage.
        self._replay_decks: dict[int, list[int]] = {}
        for i, sm in enumerate(norm):
            if i != self.anchor:
                deck = list(range(sm.start, sm.start + sm.count))
                self._rng.shuffle(deck)
                self._replay_decks[i] = deck
        self._owed = [0.0] * len(norm)

    def __len__(self) -> int:
        if self._nstream:
            batches, _examples = self._simulate_nstream_epoch_size()
            return batches
        return len(self._batches_for_epoch(self.epoch))

    def _draw_generic(self, deck: list[int], rng: random.Random, n: int) -> list[int]:
        out: list[int] = []
        generic_indices = list(range(self.primary_count, self.primary_count + self.generic_count))
        while len(out) < n:
            if not deck:
                deck.extend(generic_indices)
                rng.shuffle(deck)
            out.append(deck.pop())
        return out

    def _split_physical(
        self,
        logical_indices: list[int],
        *,
        logical_steps: int = 1,
        rng: random.Random | None = None,
    ) -> list[list[int]]:
        return _split_length_sorted_physical(
            logical_indices,
            lengths=self.lengths,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logical_steps=logical_steps,
            rng=rng,
        )

    def _make_batches(self, epoch: int) -> list[list[int]]:
        if self._nstream:
            return self._make_batches_nstream()
        rng = self._rng
        primary = list(range(self.primary_count))
        rng.shuffle(primary)
        generic_deck = list(range(self.primary_count, self.primary_count + self.generic_count))
        rng.shuffle(generic_deck)
        batches: list[list[int]] = []
        pos = 0
        while pos < len(primary):
            primary_take = min(self.primary_per_logical, len(primary) - pos)
            primary_window = primary[pos : pos + primary_take]
            pos += primary_take
            if primary_take == self.primary_per_logical:
                generic_take = self.generic_per_logical
            else:
                generic_take = int(
                    round(primary_take * self.generic_fraction / (1.0 - self.generic_fraction))
                )
            logical_indices = primary_window + self._draw_generic(generic_deck, rng, generic_take)
            rng.shuffle(logical_indices)
            batches.extend(self._split_physical(logical_indices))
        return batches

    def _draw_with_replacement(
        self, deck: list[int], stream: MixStream, rng: random.Random, n: int
    ) -> list[int]:
        out: list[int] = []
        full = range(stream.start, stream.start + stream.count)
        while len(out) < n:
            if not deck:
                deck.extend(full)
                rng.shuffle(deck)
            out.append(deck.pop())
        return out

    def _nstream_group_counts(
        self,
        *,
        logical: int,
        anchor_remaining: int,
        owed: list[float],
        rng: random.Random,
    ) -> tuple[int, int, list[int]]:
        n = len(self.streams)
        anchor = self.anchor
        expected_anchor_per_logical = max(1e-9, self.streams[anchor].weight * logical)
        group_steps = min(
            self.stream_strict_every,
            max(1, math.ceil(anchor_remaining / expected_anchor_per_logical)),
        )
        slots = logical * group_steps
        for i in range(n):
            owed[i] += self.streams[i].weight * slots
        counts = [0] * n
        for _ in range(slots):
            order = list(range(n))
            rng.shuffle(order)  # random tie-break keeps candidate sets non-degenerate
            best = -1
            best_owed = float("-inf")
            for i in order:
                if i == anchor and counts[anchor] >= anchor_remaining:
                    continue  # anchor is without-replacement; do not exceed its remainder
                if owed[i] > best_owed:
                    best_owed = owed[i]
                    best = i
            if best < 0:
                break  # only the exhausted anchor was eligible (n==1 edge)
            counts[best] += 1
            owed[best] -= 1.0
        # priority-1 diversity: a stream that should appear in this strictness
        # window (weight*slots >= 1) but drew 0 borrows one slot from the
        # fullest stream; debit deficit so proportions stay on track. Streams
        # below one-per-window are left to amortize.
        if slots >= n:
            for i in range(n):
                if counts[i] > 0 or self.streams[i].weight * slots < 1.0:
                    continue
                if i == anchor and counts[anchor] >= anchor_remaining:
                    continue  # anchor genuinely exhausted this epoch; cannot force
                donor = max(range(n), key=lambda j: counts[j] if j != i else -1)
                if counts[donor] > 1:
                    counts[i] += 1
                    counts[donor] -= 1
                    owed[i] -= 1.0
                    owed[donor] += 1.0
        return group_steps, slots, counts

    def _simulate_nstream_epoch_size(self) -> tuple[int, int]:
        rng = random.Random()
        rng.setstate(self._rng.getstate())
        logical = self.batch_size * self.gradient_accumulation_steps
        owed = list(self._owed)
        anchor = self.anchor
        anchor_deck = list(
            range(self.streams[anchor].start, self.streams[anchor].start + self.streams[anchor].count)
        )
        rng.shuffle(anchor_deck)
        replay_decks = {i: list(deck) for i, deck in self._replay_decks.items()}
        anchor_remaining = self.streams[anchor].count
        batches = 0
        examples = 0
        while anchor_remaining > 0:
            group_steps, _slots, counts = self._nstream_group_counts(
                logical=logical,
                anchor_remaining=anchor_remaining,
                owed=owed,
                rng=rng,
            )
            group_examples = sum(counts)
            examples += group_examples
            for i, take in enumerate(counts):
                if not take:
                    continue
                if i == anchor:
                    del anchor_deck[:take]
                else:
                    deck = replay_decks[i]
                    full = range(self.streams[i].start, self.streams[i].start + self.streams[i].count)
                    for _ in range(take):
                        if not deck:
                            deck.extend(full)
                            rng.shuffle(deck)
                        deck.pop()
            anchor_remaining -= counts[anchor]
            dummy = list(range(group_examples))
            rng.shuffle(dummy)
            batches += len(self._split_physical(dummy, logical_steps=group_steps, rng=rng))
        return batches, examples

    def _iter_batches_nstream(self):
        # Greedy largest-deficit fill: each logical strictness window owes every
        # stream weight*L*strict_every more slots. Physical batches are yielded
        # group-by-group, so training can start as soon as the first window is
        # planned; no full epoch schedule has to be materialized before the GPU
        # gets work.
        rng = self._rng
        logical = self.batch_size * self.gradient_accumulation_steps
        anchor = self.anchor
        # anchor deck is fresh each epoch (consumed once, exactly-N coverage); replay
        # decks + deficit persist on self across epochs (continuous coverage/proportions).
        anchor_deck = list(
            range(self.streams[anchor].start, self.streams[anchor].start + self.streams[anchor].count)
        )
        rng.shuffle(anchor_deck)
        owed = self._owed
        anchor_remaining = self.streams[anchor].count
        while anchor_remaining > 0:
            group_steps, _slots, counts = self._nstream_group_counts(
                logical=logical,
                anchor_remaining=anchor_remaining,
                owed=owed,
                rng=rng,
            )
            logical_indices: list[int] = []
            for i in range(len(self.streams)):
                take = counts[i]
                if not take:
                    continue
                if i == anchor:
                    logical_indices.extend(anchor_deck[:take])  # without replacement
                    del anchor_deck[:take]
                else:
                    logical_indices.extend(
                        self._draw_with_replacement(self._replay_decks[i], self.streams[i], rng, take)
                    )
            anchor_remaining -= counts[anchor]
            rng.shuffle(logical_indices)
            yield from self._split_physical(logical_indices, logical_steps=group_steps, rng=rng)

    def _make_batches_nstream(self) -> list[list[int]]:
        return list(self._iter_batches_nstream())

    def _batches_for_epoch(self, epoch: int) -> list[list[int]]:
        if self._cached_epoch != epoch or self._cached_batches is None:
            self._cached_batches = self._make_batches(epoch)
            self._cached_epoch = epoch
        return self._cached_batches

    def __iter__(self):
        if self._nstream:
            self.epoch += 1
            yield from self._iter_batches_nstream()
            return
        batches = self._batches_for_epoch(self.epoch)
        self.epoch += 1
        self._cached_epoch = None
        self._cached_batches = None
        yield from batches

    def summary(self) -> str:
        logical = self.batch_size * self.gradient_accumulation_steps
        if self._nstream:
            sampler_batches, total = self._simulate_nstream_epoch_size()
            streams = " ".join(f"{s.name}:{s.count}@{s.weight:.3g}" for s in self.streams)
            return (
                f"streams=[{streams}] anchor={self.streams[self.anchor].name}"
                f" strict_every={self.stream_strict_every}"
                f" batch={self.batch_size}x{self.gradient_accumulation_steps}"
                f" logical_examples={logical} epoch_examples={total} sampler_batches={sampler_batches}"
            )
        total = sum(len(batch) for batch in self._batches_for_epoch(self.epoch))
        generic_examples = sum(
            1 for batch in self._batches_for_epoch(self.epoch) for idx in batch if idx >= self.primary_count
        )
        return (
            f"primary={self.primary_count} generic_pool={self.generic_count}"
            f" generic_mix_frac={self.generic_fraction:g}"
            f" batch={self.batch_size}x{self.gradient_accumulation_steps}"
            f" logical_examples={logical}"
            f" per_logical=primary:{self.primary_per_logical}/generic:{self.generic_per_logical}"
            f" epoch_examples={total} epoch_generic={generic_examples}"
            f" sampler_batches={len(self)}"
        )


def add_patience_trajectory_args(parser) -> None:
    """Validation-patience stopping and patience-LR anneal/rebound flags.

    Shared by train-lora.py and small plain-PyTorch loops (via EpochTrajectory) so
    one flag vocabulary and one set of defaults drive ValidationController and
    PatienceLrController everywhere.
    """
    parser.add_argument(
        "--val-patience",
        type=int,
        default=3,
        help=(
            "Number of consecutive full val cycles with no improvement by "
            "--val-min-delta before stopping. [legacy: 0 — val stopping did not exist]"
        ),
    )
    parser.add_argument(
        "--val-min-delta",
        type=float,
        default=0.001,
        help="Minimum decrease in val loss to count as improvement. [legacy: 0.0]",
    )
    parser.add_argument(
        "--patience-lr-factor",
        type=float,
        default=1.0,
        help=(
            "Opt-in LR anneal factor applied when validation patience would otherwise stop. "
            "Values in (0,1) multiply optimizer group LRs and scheduler base LRs; 1 disables."
        ),
    )
    parser.add_argument(
        "--patience-lr-anneal-stages",
        type=int,
        default=1,
        help=(
            "Number of patience-exhaustion events to convert into LR anneal stages before "
            "allowing validation patience to stop. Each stage resets patience/cadence and "
            "continues training at the lower LR. Only active when --patience-lr-factor < 1."
        ),
    )
    parser.add_argument(
        "--patience-lr-min-factor",
        type=float,
        default=0.0,
        help=(
            "Optional lower bound on cumulative patience LR scale relative to the current "
            "scheduler base LRs. 0 disables the floor."
        ),
    )
    parser.add_argument(
        "--patience-lr-floor",
        type=float,
        default=0.0,
        help=(
            "Optional absolute floor for current optimizer group LRs when applying a "
            "patience anneal stage. The stage uses max(lr * factor, floor). 0 disables."
        ),
    )
    parser.add_argument(
        "--patience-lr-max-increase-factor",
        type=float,
        default=1.0,
        help=(
            "Default-off noisy-stopping recovery knob. Values >1 allow a patience LR "
            "stage to increase a current group LR by at most this factor when a floor "
            "or future policy would otherwise raise it, still capped by the original "
            "starting LR and the nominal unratcheted schedule LR at the current step. "
            "Rebound increases are already capped by their geometric target; values >1 "
            "add an extra per-adjustment cap there."
        ),
    )
    parser.add_argument(
        "--patience-lr-rebound-ratio",
        type=float,
        default=1.25,
        help=(
            "Damped-hysteresis probe after a patience LR anneal stage. If the recent "
            "validation-loss improvement is at least this multiple of the previous "
            "reference improvement, geometrically rebound partway toward the pre-anneal "
            "LR; weaker follow-up progress damps back down. Values <=1 disable."
        ),
    )
    parser.add_argument(
        "--patience-lr-rebound-ema",
        type=float,
        default=2.0,
        help=(
            "EMA horizon, in full validation cycles, used for the rebound progress "
            "signal. 1 uses the raw cycle-to-cycle loss decrease."
        ),
    )
    parser.add_argument(
        "--patience-lr-rebound-window",
        type=int,
        default=3,
        help=(
            "Number of full validation cycles after each patience LR anneal stage during "
            "which rebound/damping adjustments may be made. 0 disables rebound."
        ),
    )
    parser.add_argument(
        "--patience-lr-rebound-up-power",
        type=float,
        default=0.5,
        help=(
            "Log-space fraction to move from the current LR scale toward the pre-anneal "
            "upper bound when rebound progress is strong. 0=no move, 0.5=geometric "
            "halfway, 1=jump to bound."
        ),
    )
    parser.add_argument(
        "--patience-lr-rebound-down-power",
        type=float,
        default=0.5,
        help=(
            "Log-space fraction to move from the current LR scale back toward the last "
            "lower bound when rebound progress is weak."
        ),
    )
    parser.add_argument(
        "--patience-lr-rebound-down-overshoot",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the lower rebound bound when damping weak progress. 1 means "
            "do not probe below the proven lower LR; values <1 cautiously overshoot lower."
        ),
    )


class ValidationController:
    """Validation-driven early stopping with optional exact-val recheck.

    Owns the stop decision and its state (running-val best/patience and the exact-val
    recheck bookkeeping). It *acts on* validation results — it does NOT run validation
    or touch the model, optimizer, checkpoint pool, or disk; the host computes a
    (smeared) running val loss and calls observe(), supplying those effects as hooks.

    observe() updates best/patience, optionally triggers an exact-val recheck through
    the hooks, consults the host's reactive LR on patience exhaustion
    (_try_patience_lr_anneal), and returns whether training should stop. Reactive LR
    (anneal/rebound) is a separate concern (PatienceLrController); with no LR behavior
    attached those hooks are inert and observe() reduces to plain patience
    early-stopping (+ exact recheck) — the simple/debuggable mode.

    Hook protocol (the host, e.g. RegionScaleTrainer, provides these by their existing
    method names, so call sites and the characterization harness stay unchanged):
      _update_val_cycle_progress(running_loss)         per-cycle LR progress signal
      _maybe_adjust_patience_lr_rebound(running_loss)  per-cycle LR rebound watch
      _resume_recent_checkpoint_capture(*, reason)
      _save_val_best()
      _write_val_decode_snapshot(event, *, running_loss)
      _exact_eval_candidate() -> (source, state, source_steps)
      _compute_exact_val_loss(state) -> float
      _save_exact_best_state(*, loss, source, source_steps, state)
      _pause_recent_checkpoint_capture(*, rewind_to_step, reason)
      _try_patience_lr_anneal(running_loss) -> bool    anneal+continue instead of stop?
      _reset_val_cadence(step)
    """

    def __init__(
        self,
        *,
        val_patience: int,
        val_min_delta: float,
        exact_early_stopping: float = float("inf"),
        exact_early_stopping_cooldown: int = 0,
    ):
        self.val_patience = int(val_patience)
        self.val_min_delta = float(val_min_delta)
        self.exact_early_stopping = float(exact_early_stopping)
        self.exact_early_stopping_cooldown = int(exact_early_stopping_cooldown)
        self.val_best_loss: float = float("inf")
        self.patience_count: int = 0
        self.exact_best_loss: float | None = None
        self.exact_best_source: str | None = None
        self.exact_best_source_steps: tuple[int, ...] = ()
        self.exact_last_loss: float | None = None
        self.exact_checks: int = 0
        self.exact_last_check_step: int = 0
        self.exact_last_candidate_signature: tuple[str, tuple[int, ...]] | None = None

    def exact_enabled(self) -> bool:
        return self.exact_early_stopping >= 0 and math.isfinite(self.exact_early_stopping)

    def observe(self, running_loss: float, *, step: int, cycles_complete: int, hooks) -> bool:
        """Process one completed val cycle; return True iff training should stop."""
        hooks._update_val_cycle_progress(running_loss)
        hooks._maybe_adjust_patience_lr_rebound(running_loss)
        if running_loss < self.val_best_loss - self.val_min_delta:
            self.val_best_loss = running_loss
            self.patience_count = 0
            hooks._resume_recent_checkpoint_capture(reason="smeared val improved")
            hooks._save_val_best()
            info(f"[Val] NEW BEST loss={running_loss:.4f} at cycle {cycles_complete}")
            hooks._write_val_decode_snapshot("best", running_loss=running_loss)
            return False
        self.patience_count += 1
        info(
            f"[Val] No improvement (patience {self.patience_count}/{self.val_patience})"
            f" running_loss={running_loss:.4f} best={self.val_best_loss:.4f}"
        )
        hooks._write_val_decode_snapshot("patience", running_loss=running_loss)
        if self.exact_enabled() and self._exact_recheck(running_loss, step=step, hooks=hooks):
            return False
        if self.patience_count >= self.val_patience:
            if hooks._try_patience_lr_anneal(running_loss):
                self.patience_count = 0
                hooks._reset_val_cadence(step)
                hooks._resume_recent_checkpoint_capture(reason="patience LR anneal")
                return False
            info(f"\n[VAL-STOP] Val loss no improvement for {self.val_patience} cycles; stopping.")
            hooks._write_val_decode_snapshot("stop", running_loss=running_loss)
            return True
        return False

    def _exact_recheck(self, running_loss: float, *, step: int, hooks) -> bool:
        """Run the exact-val recheck; return True iff it reset patience (skip the stop check)."""
        ref_loss = self.exact_best_loss if self.exact_best_loss is not None else self.val_best_loss
        if running_loss < ref_loss + self.exact_early_stopping:
            return False
        source, state, source_steps = hooks._exact_eval_candidate()
        signature = (source, tuple(source_steps))
        exact_loss: float | None = None
        reused = False
        if signature == self.exact_last_candidate_signature and self.exact_last_loss is not None:
            exact_loss = self.exact_last_loss
            reused = True
            info(
                f"[ExactVal] reusing last exact score for unchanged candidate {source}"
                f" exact_val_loss={exact_loss:.4f}"
            )
        else:
            cooldown_active = (
                self.exact_early_stopping_cooldown > 0
                and self.exact_last_check_step > 0
                and step - self.exact_last_check_step < self.exact_early_stopping_cooldown
                and self.patience_count < self.val_patience
            )
            if cooldown_active:
                remaining = self.exact_early_stopping_cooldown - (step - self.exact_last_check_step)
                info(
                    f"[ExactVal] trigger active but cooldown blocks new whole-val check"
                    f" for {remaining} more optimizer steps"
                )
            else:
                exact_loss = hooks._compute_exact_val_loss(state)
                self.exact_checks += 1
                self.exact_last_loss = exact_loss
                self.exact_last_check_step = step
                self.exact_last_candidate_signature = signature
                info(
                    f"[ExactVal] check {self.exact_checks}: source={source}"
                    f" exact_val_loss={exact_loss:.4f} trigger_ref={ref_loss:.4f}"
                )
        if exact_loss is not None:
            if self.exact_best_loss is None:
                self._set_exact_best(exact_loss, source, source_steps, state, hooks)
                self.patience_count = 0
                info("[ExactVal] Established first exact-val baseline; not stopping on first exact check.")
                return True
            if exact_loss < self.exact_best_loss - self.val_min_delta:
                self._set_exact_best(exact_loss, source, source_steps, state, hooks)
                self.patience_count = 0
                info(f"[ExactVal] NEW BEST exact loss={exact_loss:.4f}; resetting patience.")
                return True
            if exact_loss <= self.exact_best_loss + self.val_min_delta:
                if not reused and self.exact_best_source_steps:
                    rewind_to_step = max(self.exact_best_source_steps)
                    hooks._pause_recent_checkpoint_capture(
                        rewind_to_step=rewind_to_step,
                        reason="smeared-regression-not-confirmed-by-exact-val",
                    )
                    self.patience_count = 0
                    info(
                        f"[ExactVal] Exact loss {exact_loss:.4f} stayed within tolerance of best exact"
                        f" {self.exact_best_loss:.4f}; rewound recent averaging horizon"
                        f" to step {rewind_to_step} and reset patience."
                    )
                    return True
                info(
                    f"[ExactVal] Exact loss {exact_loss:.4f} remained within tolerance of best exact"
                    f" {self.exact_best_loss:.4f}; unchanged candidate, keeping current patience."
                )
            info(
                f"[ExactVal] No improvement vs best exact {self.exact_best_loss:.4f}"
                f" (patience {self.patience_count}/{self.val_patience})"
            )
        return False

    def _set_exact_best(self, loss: float, source: str, source_steps, state, hooks) -> None:
        self.exact_best_loss = loss
        self.exact_best_source = source
        self.exact_best_source_steps = tuple(int(s) for s in source_steps)
        hooks._save_exact_best_state(loss=loss, source=source, source_steps=list(source_steps), state=state)


class PatienceLrController:
    """Reactive learning-rate control driven by validation patience/progress.

    Owns the patience-LR anneal/rebound decisions and their state (anneal event count,
    rebound watch, validation-progress EMA). It reads the host's currently-applied LR
    scale and requests optimizer changes through hooks, but does NOT touch the optimizer
    itself — the LR mutation is inherently host-coupled. A separate concern from
    ValidationController (early stopping), which consults this via the host on patience
    exhaustion; attach it or not independently.

    The cumulative applied LR scale stays host-side (it is maintained by the optimizer
    mutation and read back here), so this controller is the single source of the *policy*
    state, not the applied scale.

    Hook protocol (the host provides):
      _patience_lr_scale                 attr: cumulative LR scale currently applied
      _val_cycles_complete               attr: completed validation cycles (rebound window)
      _apply_patience_lr_scale_change(next_scale, running_loss, *, reason, allow_increase)
                                         -> bool: mutate optimizer/scheduler toward next_scale
    """

    def __init__(
        self,
        *,
        patience_lr_factor: float,
        patience_lr_min_factor: float,
        patience_lr_floor: float,
        patience_lr_max_increase_factor: float,
        patience_lr_anneal_stages: int,
        patience_lr_rebound_ratio: float,
        patience_lr_rebound_ema: float,
        patience_lr_rebound_window: int,
        patience_lr_rebound_up_power: float,
        patience_lr_rebound_down_power: float,
        patience_lr_rebound_down_overshoot: float,
        val_min_delta: float,
    ):
        self.patience_lr_factor = float(patience_lr_factor)
        self.patience_lr_min_factor = float(patience_lr_min_factor)
        self.patience_lr_floor = float(patience_lr_floor)
        self.patience_lr_max_increase_factor = max(1.0, float(patience_lr_max_increase_factor))
        self.patience_lr_anneal_stages = int(patience_lr_anneal_stages)
        self.patience_lr_rebound_ratio = float(patience_lr_rebound_ratio)
        self.patience_lr_rebound_ema = max(1.0, float(patience_lr_rebound_ema))
        self.patience_lr_rebound_window = max(0, int(patience_lr_rebound_window))
        self.patience_lr_rebound_up_power = min(1.0, max(0.0, float(patience_lr_rebound_up_power)))
        self.patience_lr_rebound_down_power = min(1.0, max(0.0, float(patience_lr_rebound_down_power)))
        self.patience_lr_rebound_down_overshoot = max(1e-6, float(patience_lr_rebound_down_overshoot))
        self.val_min_delta = float(val_min_delta)
        self._patience_lr_events: int = 0
        self._patience_lr_rebound_adjustments: int = 0
        self._patience_lr_rebound_state: dict[str, float | int | str] | None = None
        self._val_last_cycle_loss: float | None = None
        self._val_last_cycle_progress: float | None = None
        self._val_progress_ema: float | None = None

    def _patience_lr_rebound_enabled(self) -> bool:
        return (
            self.patience_lr_factor < 1.0
            and self.patience_lr_anneal_stages > 0
            and self.patience_lr_rebound_ratio > 1.0
            and self.patience_lr_rebound_window > 0
        )

    def _update_val_cycle_progress(self, running_loss: float) -> None:
        if self._val_last_cycle_loss is None:
            self._val_last_cycle_loss = running_loss
            self._val_last_cycle_progress = None
            return
        progress = self._val_last_cycle_loss - running_loss
        self._val_last_cycle_loss = running_loss
        self._val_last_cycle_progress = progress
        if self._val_progress_ema is None or self.patience_lr_rebound_ema <= 1.0:
            self._val_progress_ema = progress
            return
        alpha = 2.0 / (self.patience_lr_rebound_ema + 1.0)
        self._val_progress_ema = alpha * progress + (1.0 - alpha) * self._val_progress_ema

    def _patience_lr_progress_floor(self) -> float:
        return max(float(self.val_min_delta), 1e-12)

    def _patience_lr_progress_signal(self) -> float | None:
        if self._val_progress_ema is None:
            return None
        return max(0.0, float(self._val_progress_ema))

    def _patience_lr_geom_toward(self, current: float, target: float, power: float) -> float:
        current = max(1e-12, float(current))
        target = max(1e-12, float(target))
        if abs(target - current) <= max(1e-12, abs(current) * 1e-9):
            return current
        return current * ((target / current) ** min(1.0, max(0.0, power)))

    def _begin_patience_lr_rebound(self, previous_scale: float, running_loss: float, *, hooks) -> None:
        if not self._patience_lr_rebound_enabled():
            self._patience_lr_rebound_state = None
            return
        low_scale = float(hooks._patience_lr_scale)
        high_scale = float(previous_scale)
        if high_scale <= low_scale + max(1e-12, abs(low_scale) * 1e-9):
            self._patience_lr_rebound_state = None
            return
        signal = self._patience_lr_progress_signal()
        floor = self._patience_lr_progress_floor()
        reference = max(signal if signal is not None else 0.0, floor)
        cycle = int(hooks._val_cycles_complete)
        self._patience_lr_rebound_state = {
            "start_cycle": cycle,
            "expires_cycle": cycle + self.patience_lr_rebound_window,
            "lower_scale": low_scale,
            "upper_scale": high_scale,
            "reference_progress": reference,
            "direction": "awaiting",
        }
        info(
            "[Val] Patience LR rebound watch started:"
            f" cycle={cycle} expires_cycle={cycle + self.patience_lr_rebound_window}"
            f" lower_scale={low_scale:g} upper_scale={high_scale:g}"
            f" reference_progress={reference:.6g} running_loss={running_loss:.4f}"
        )

    def _maybe_adjust_patience_lr_rebound(self, running_loss: float, *, hooks) -> None:
        state = self._patience_lr_rebound_state
        if not state or not self._patience_lr_rebound_enabled():
            return
        cycle = int(hooks._val_cycles_complete)
        expires_cycle = int(state.get("expires_cycle", cycle))
        if cycle > expires_cycle:
            info(
                "[Val] Patience LR rebound watch expired:"
                f" cycle={cycle} expires_cycle={expires_cycle}"
                f" scale={hooks._patience_lr_scale:g}"
            )
            self._patience_lr_rebound_state = None
            return
        signal = self._patience_lr_progress_signal()
        if signal is None:
            return
        floor = self._patience_lr_progress_floor()
        reference = max(float(state.get("reference_progress", floor)), floor)
        threshold = reference * self.patience_lr_rebound_ratio
        current = float(hooks._patience_lr_scale)
        direction = str(state.get("direction", "awaiting"))
        raw = self._val_last_cycle_progress
        raw_s = "none" if raw is None else f"{raw:.6g}"
        if signal >= threshold:
            upper = max(current, float(state.get("upper_scale", current)))
            target = self._patience_lr_geom_toward(current, upper, self.patience_lr_rebound_up_power)
            if target <= current + max(1e-12, abs(current) * 1e-9):
                return
            old_current = current
            if hooks._apply_patience_lr_scale_change(
                target,
                running_loss,
                reason=f"Patience LR rebound up adjustment {self._patience_lr_rebound_adjustments + 1}",
                allow_increase=True,
            ):
                self._patience_lr_rebound_adjustments += 1
                state["lower_scale"] = old_current
                state["reference_progress"] = max(signal, floor)
                state["direction"] = "up"
                info(
                    "[Val] Patience LR rebound accepted stronger progress:"
                    f" raw_progress={raw_s} ema_progress={signal:.6g}"
                    f" threshold={threshold:.6g} next_scale={hooks._patience_lr_scale:g}"
                )
            return
        if direction == "awaiting":
            info(
                "[Val] Patience LR rebound awaiting stronger post-anneal progress:"
                f" raw_progress={raw_s} ema_progress={signal:.6g}"
                f" threshold={threshold:.6g} scale={hooks._patience_lr_scale:g}"
            )
            return
        lower = min(current, float(state.get("lower_scale", current)))
        target_lower = lower * self.patience_lr_rebound_down_overshoot
        min_factor = max(0.0, float(self.patience_lr_min_factor))
        if min_factor > 0.0:
            target_lower = max(min_factor, target_lower)
        target = self._patience_lr_geom_toward(current, target_lower, self.patience_lr_rebound_down_power)
        if target >= current - max(1e-12, abs(current) * 1e-9):
            state["reference_progress"] = max(signal, floor)
            return
        old_current = current
        if hooks._apply_patience_lr_scale_change(
            target,
            running_loss,
            reason=f"Patience LR rebound down adjustment {self._patience_lr_rebound_adjustments + 1}",
            allow_increase=False,
        ):
            self._patience_lr_rebound_adjustments += 1
            state["upper_scale"] = old_current
            state["reference_progress"] = max(signal, floor)
            state["direction"] = "down"
            info(
                "[Val] Patience LR rebound damped weaker progress:"
                f" raw_progress={raw_s} ema_progress={signal:.6g}"
                f" threshold={threshold:.6g} next_scale={hooks._patience_lr_scale:g}"
            )

    def _apply_patience_lr_anneal(self, running_loss: float, *, hooks) -> bool:
        factor = float(self.patience_lr_factor)
        if not (0.0 < factor < 1.0):
            return False
        min_factor = max(0.0, float(self.patience_lr_min_factor))
        next_scale = hooks._patience_lr_scale * factor
        if min_factor > 0.0:
            next_scale = max(min_factor, next_scale)
        if next_scale >= hooks._patience_lr_scale - 1e-12:
            return False
        previous_scale = float(hooks._patience_lr_scale)
        event = self._patience_lr_events + 1
        if not hooks._apply_patience_lr_scale_change(
            next_scale,
            running_loss,
            reason=f"Patience LR anneal event {event}",
            allow_increase=False,
        ):
            return False
        self._patience_lr_events = event
        self._begin_patience_lr_rebound(previous_scale=previous_scale, running_loss=running_loss, hooks=hooks)
        return True

    def _try_patience_lr_anneal(self, running_loss: float, *, hooks) -> bool:
        # Called (via the host) by ValidationController on patience exhaustion: anneal LR and
        # report "continue instead of stop". The host's caller resets val patience/cadence.
        if self._patience_lr_events >= max(0, self.patience_lr_anneal_stages):
            return False
        if not self._apply_patience_lr_anneal(running_loss, hooks=hooks):
            return False
        info(
            f"[VAL-ANNEAL] Val patience exhausted; applied LR anneal"
            f" {self._patience_lr_events}/{self.patience_lr_anneal_stages},"
            " reset patience/cadence, and continuing."
        )
        return True


class RefEmbedAux:
    """Reference-embedding auxiliary loss.

    Pulls a pooled decoder hidden state toward a cached target embedding via a small
    projection head installed on the model (model.ref_embed_aux_head). Owns the pooling
    mode + loss config and the pool/loss computation; the host owns the target cache, the
    collator that supplies (ref_targets, ref_mask), and head construction/save.

    Two attachment regimes, selected by ``pool``:

    - Pooled (``prompt-end`` / ``source-end`` / ``target-last`` / ``target-mean``):
      one pooled decoder hidden state per example toward one cached target embedding
      (the v1 reference-embedding aux; blunt, many-to-one).
    - Alignment-conditioned (``aligned-span``): each supervised target token's decoder
      hidden state toward the cached encoder rep of *its aligned source span*. Finer,
      local signal -- the [[depth-aligned-embedding-aux]] / [[alignment-for-mt]] Hook 3
      mechanism change. The collator supplies per-token targets ``ref_targets`` of shape
      ``[B, L, E]`` and a per-token validity mask ``ref_mask`` ``[B, L]`` (True only where
      a supervised target token has a valid aligned-source-span target).
    """

    def __init__(self, *, weight: float, loss: str, pool: str, head_attr: str = "ref_embed_aux_head"):
        self.weight = float(weight)
        self.loss = str(loss)
        self.pool = str(pool)
        self.head_attr = str(head_attr)

    def head_module(self, model):
        head_module = getattr(model, self.head_attr, None)
        if head_module is not None:
            return head_module
        base_model = getattr(model, "base_model", None)
        return getattr(base_model, self.head_attr, None)

    def pool_hidden(
        self,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = target_mask.to(device=hidden_states.device, dtype=torch.bool)
        counts = mask.sum(dim=1)
        if self.pool in {"prompt-end", "source-end"}:
            prompt_positions = counts - 1
            valid = prompt_positions >= 0
            if not bool(valid.any()):
                return hidden_states.new_zeros((0, hidden_states.shape[-1])), valid
            rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)[valid]
            return hidden_states[rows, prompt_positions[valid]], valid
        valid = counts > 0
        if not bool(valid.any()):
            return hidden_states.new_zeros((0, hidden_states.shape[-1])), valid
        hidden = hidden_states[valid]
        mask = mask[valid]
        if self.pool == "target-last":
            positions = mask.long().sum(dim=1) - 1
            cumsum = mask.long().cumsum(dim=1)
            last_positions = (cumsum == positions[:, None] + 1).long().argmax(dim=1)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[rows, last_positions], valid
        if self.pool != "target-mean":
            raise ValueError(f"unsupported --ref-embed-aux-pool {self.pool!r}")
        pooled = hidden.to(torch.float32).masked_fill(~mask[..., None], 0.0).sum(dim=1)
        pooled = pooled / counts[valid].to(device=hidden.device, dtype=torch.float32).clamp(min=1)[:, None]
        return pooled, valid

    def loss_value(
        self,
        model,
        outputs,
        ref_targets: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.weight <= 0 or ref_targets is None or ref_mask is None:
            return None
        head_module = self.head_module(model)
        if head_module is None:
            raise RuntimeError("--ref-embed-aux-weight > 0 but no ref_embed_aux_head is installed")
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("reference-embedding aux loss requires model outputs.hidden_states")
        if self.pool == "aligned-span":
            return self._aligned_span_loss(head_module, hidden_states[-1], ref_targets, ref_mask)
        pooled, valid = self.pool_hidden(hidden_states[-1], ref_mask)
        if pooled.numel() == 0:
            return hidden_states[-1].new_zeros(())
        head_param = next(head_module.parameters())
        pred = head_module(pooled.to(dtype=head_param.dtype))
        pred = pred.to(torch.float32)
        target = ref_targets.to(device=pred.device, dtype=torch.float32)
        target = target[valid.to(pred.device)]
        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        if self.loss == "cosine":
            return 1.0 - (pred * target).sum(dim=-1).mean()
        if self.loss == "mse":
            return F.mse_loss(pred, target)
        raise ValueError(f"unsupported --ref-embed-aux-loss {self.loss!r}")

    def _aligned_span_loss(
        self,
        head_module,
        hidden: torch.Tensor,
        ref_token_targets: torch.Tensor,
        ref_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Alignment-conditioned per-token loss.

        Each supervised target token (where ``ref_token_mask`` is True) has its
        last-layer decoder hidden state projected through the aux head and pulled
        toward ``ref_token_targets`` at that position -- the cached encoder rep of the
        token's aligned source span. Unmasked positions (prompt, EOS, target tokens
        with no alignment) contribute nothing. Shapes: ``hidden`` [B, L, H],
        ``ref_token_targets`` [B, L, E], ``ref_token_mask`` [B, L].
        """
        mask = ref_token_mask.to(device=hidden.device, dtype=torch.bool)
        if not bool(mask.any()):
            return hidden.new_zeros(())
        sel_hidden = hidden[mask]  # [P, H]
        head_param = next(head_module.parameters())
        pred = head_module(sel_hidden.to(dtype=head_param.dtype)).to(torch.float32)
        target = ref_token_targets.to(device=pred.device, dtype=torch.float32)[mask]  # [P, E]
        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        if self.loss == "cosine":
            return 1.0 - (pred * target).sum(dim=-1).mean()
        if self.loss == "mse":
            return F.mse_loss(pred, target)
        raise ValueError(f"unsupported --ref-embed-aux-loss {self.loss!r}")


def build_aux_head(in_dim: int, out_dim: int, *, kind: str = "linear", dropout: float = 0.0):
    """Build a RefEmbedAux projection head (decoder-hidden -> target-embed space).

    `linear` is one bias-free `nn.Linear` (kept identical to the original head when
    dropout=0, so pinned golden losses are unaffected). `mlp2` adds a hidden GELU layer:
    higher capacity to fit the target, which tends to shrink the residual gradient
    reaching its input (the decoder hidden / adapter) -- a swept axis, not a default. The
    head is a learned cross-space map (the two spaces share no basis), so it is always
    trainable; dynamics (LR schedule, dropout, capacity) are the levers, never freezing.
    """
    from torch import nn

    if kind == "linear":
        head = nn.Linear(in_dim, out_dim, bias=False)
        return nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head
    if kind == "mlp2":
        return nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, out_dim, bias=False)
        )
    raise ValueError(f"unknown aux head kind {kind!r} (expected linear|mlp2)")


class RefEmbedAuxSet:
    """A weighted set of RefEmbedAux objectives -> total aux = Σ_i weight_i · aux_i.

    The multi-embedder aux: each objective is one RefEmbedAux with its own weight, its
    own projection head (``head_attr``, default ``ref_embed_aux_head_{i}``), and its own
    per-batch ``(target, mask)``. A single scalar-flag objective is just a 1-element set,
    so one code path serves one or many embedders. The host (train-lora) builds the
    per-objective targets and heads and passes the per-batch target/mask lists; this owns
    the loss aggregation and head-attr bookkeeping. ``head_attr`` differs per objective so
    each head is a distinct module on the model.
    """

    def __init__(self, specs: list[dict]):
        self.objectives = [
            RefEmbedAux(
                weight=spec["weight"],
                loss=spec.get("loss", "cosine"),
                pool=spec["pool"],
                head_attr=spec.get("head_attr") or f"ref_embed_aux_head_{i}",
            )
            for i, spec in enumerate(specs)
        ]

    @property
    def enabled(self) -> bool:
        return any(aux.weight > 0 for aux in self.objectives)

    def head_attrs(self) -> list[str]:
        return [aux.head_attr for aux in self.objectives]

    def loss_value(self, model, outputs, targets, masks):
        """Aggregate the objectives. ``targets``/``masks`` are lists aligned with the
        objectives (entries may be None). Returns ``(total_weighted_or_None, per_raw)``
        where ``per_raw[i]`` is objective i's unweighted loss (float) or None."""
        total = None
        per: list = []
        for i, aux in enumerate(self.objectives):
            target = targets[i] if targets is not None and i < len(targets) else None
            mask = masks[i] if masks is not None and i < len(masks) else None
            value = aux.loss_value(model, outputs, target, mask)
            if value is None:
                per.append(None)
                continue
            per.append(float(value.detach()))
            contrib = aux.weight * value
            total = contrib if total is None else total + contrib
        return total, per


if __name__ == "__main__":
    # CPU-only self-test of the logical-batch controllers moved out of
    # train-lora.py: every example index appears exactly once per epoch, no
    # physical batch exceeds batch_size, and a fixed seed is deterministic.
    _lengths = [1 + (i % 7) for i in range(64)]

    _lb = LengthBucketedBatchSampler(lengths=_lengths, batch_size=4, bucket_width=2, fill="partial", seed=0)
    _batches = list(iter(_lb))
    assert sorted(i for b in _batches for i in b) == list(range(len(_lengths)))
    assert all(len(b) <= 4 for b in _batches)
    _lb2 = LengthBucketedBatchSampler(lengths=_lengths, batch_size=4, bucket_width=2, fill="partial", seed=0)
    assert list(iter(_lb2)) == _batches, "length-bucket must be deterministic for a fixed seed"

    _mix = LogicalMixBatchSampler(
        lengths=_lengths,
        primary_count=12,
        generic_count=52,
        generic_fraction=0.8,
        batch_size=4,
        gradient_accumulation_steps=4,
        seed=42,
    )
    assert (_mix.primary_per_logical, _mix.generic_per_logical) == (3, 13)
    _mix_batches = list(iter(_mix))
    assert sorted(i for b in _mix_batches for i in b if i < 12) == list(range(12))

    print(f"trainlib self-test OK: {len(_batches)} bucketed batches, {len(_mix_batches)} mix batches")
