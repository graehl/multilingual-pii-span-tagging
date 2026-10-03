"""Atomic rsync publication of a quiescent training output directory."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


def receive(action: str, destination: str, generation: str, previous: str = "") -> dict:
    root = Path(destination)
    if not root.is_absolute() or root.name in {"", ".", ".."}:
        raise ValueError("checkpoint mirror destination must be an absolute run path")
    if not re.fullmatch(r"snapshot-[0-9a-f]{32}", generation):
        raise ValueError("invalid checkpoint mirror generation")
    store = root.parent / f".{root.name}.mirror"
    if root.exists() and not root.is_symlink():
        raise ValueError("checkpoint mirror refuses to replace an existing directory")
    current = os.readlink(root) if root.is_symlink() else ""
    if current and (Path(current).parent != store or not Path(current).name.startswith("snapshot-")):
        raise ValueError("checkpoint mirror destination is owned by another writer")
    stage = store / (generation + ".partial")
    if action == "prepare":
        store.mkdir(parents=True, exist_ok=True)
        stage.mkdir(exist_ok=True)
        return {"stage": str(stage), "previous": current}
    if action != "publish":
        raise ValueError(f"unknown checkpoint mirror action: {action}")
    if current != previous:
        raise RuntimeError("checkpoint mirror destination changed during transfer")
    completed = store / generation
    stage.rename(completed)
    pointer = store / (generation + ".link")
    pointer.symlink_to(completed)
    os.replace(pointer, root)
    # Only generations published by this receiver are eligible for rotation.
    (completed / ".mirror-complete.json").write_text(json.dumps({"destination": str(root)}))
    for older in store.glob("snapshot-*"):
        marker = older / ".mirror-complete.json"
        if older == completed or not marker.is_file():
            continue
        if json.loads(marker.read_text()).get("destination") == str(root):
            shutil.rmtree(older)
    return {"published": str(root), "generation": str(completed)}


class CheckpointMirror:
    """Publish after saving, synchronously so checkpoint rotation cannot race rsync."""

    def __init__(
        self,
        destination: str,
        *,
        interval_seconds: float = 3600,
        timeout_seconds: float = 600,
        ssh_command: str = "ssh -o BatchMode=yes -o ConnectTimeout=10",
    ):
        for value in (interval_seconds, timeout_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("checkpoint mirror timing must be finite and nonnegative")
        if timeout_seconds == 0:
            raise ValueError("checkpoint mirror timeout must be positive")
        self.host = None
        if ":" in destination and not destination.startswith("/"):
            self.host, destination = destination.split(":", 1)
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", self.host):
                raise ValueError("checkpoint mirror requires an SSH host alias or user@host")
        if not Path(destination).is_absolute():
            raise ValueError("checkpoint mirror requires an absolute destination path")
        self.destination = destination
        self.ssh_command = shlex.split(ssh_command)
        if not self.ssh_command:
            raise ValueError("checkpoint mirror SSH command cannot be empty")
        self.interval = interval_seconds
        self.timeout = timeout_seconds
        self.last_success = None

    def _receive(self, action: str, generation: str, previous: str = "") -> dict:
        if self.host is None:
            return receive(action, self.destination, generation, previous)
        command = shlex.join(["python3", "-", action, self.destination, generation, previous])
        result = subprocess.run(
            [*self.ssh_command, self.host, command],
            input=Path(__file__).read_text(),
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=True,
        )
        return json.loads(result.stdout)

    def save(self, source: Path, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and self.last_success is not None and now - self.last_success < self.interval:
            return False
        source = source.resolve(strict=True)
        if self.host is None and Path(self.destination).resolve().is_relative_to(source):
            raise ValueError("checkpoint mirror destination cannot be inside training output")
        generation = "snapshot-" + uuid.uuid4().hex
        prepared = self._receive("prepare", generation)
        stage = prepared["stage"]
        target = f"{self.host}:{stage}/" if self.host else stage + "/"
        command = [
            "rsync",
            "-a",
            "--checksum",
            "--protect-args",
            "--copy-unsafe-links",
            "--exclude=/.checkpoint-mirror/",
            "--exclude=/.mirror-complete.json",
        ]
        if self.host:
            command += ["-e", shlex.join(self.ssh_command)]
        if prepared["previous"]:
            command += ["--link-dest=" + prepared["previous"]]
        command += [str(source) + "/", target]
        logs = source / ".checkpoint-mirror"
        logs.mkdir(exist_ok=True)
        for attempt in range(2):
            try:
                with (
                    (logs / f"{generation}-{attempt}.out").open("w") as stdout,
                    (logs / f"{generation}-{attempt}.err").open("w") as stderr,
                ):
                    subprocess.run(command, stdout=stdout, stderr=stderr, timeout=self.timeout, check=True)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt:
                    print(f"[checkpoint-mirror] FAILED; previous copy preserved; logs={logs}", flush=True)
                    raise
        receipt = self._receive("publish", generation, prepared["previous"])
        receipt.update(source=str(source), completed_at=time.time())
        (logs / "last-success.json").write_text(json.dumps(receipt, indent=2) + "\n")
        self.last_success = time.monotonic()
        print(f"[checkpoint-mirror] published {self.host or 'local'}:{self.destination}", flush=True)
        return True


if __name__ == "__main__":
    print(json.dumps(receive(*sys.argv[1:])))
