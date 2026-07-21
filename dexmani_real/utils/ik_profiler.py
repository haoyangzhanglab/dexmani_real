"""Opt-in profiler trigger for IK timing anomaly diagnosis.

Triggered when per-iteration DLS time exceeds a threshold — spawns a
short-duration background ``py-spy record`` to capture native call stacks.

Enable with: ``DEXMANI_IK_PROFILE=1``.
Output: ``/tmp/ik_profiles/ik_dls<X>ms_<timestamp>.svg`` (flame graphs).

**Prerequisite:** ptrace must be un-restricted for py-spy to attach::

    sudo sysctl -w kernel.yama.ptrace_scope=0

When py-spy is unavailable (ptrace blocked, not installed, …) the profiler
falls back to ``faulthandler.dump_traceback()`` — Python-level stacks only,
but enough to spot GC sweeps, lock contention, or unexpected re-entrancy on
the IK thread.
"""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import time
from pathlib import Path


def _ptrace_scope() -> int:
    try:
        return int(Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip())
    except Exception:
        return -1  # can't read — assume restricted


class IKProfiler:
    """Background profiler trigger for anomalous DLS timing.

    Primary: spawns ``py-spy record --duration 3`` (non-blocking, native stacks).
    Fallback: calls ``faulthandler.dump_traceback()`` (Python-level stacks only).

    Both paths are safe to call from the IK hot path — they never raise.
    """

    def __init__(
        self,
        output_dir: str = "/tmp/ik_profiles",
        dls_threshold_ms: float = 5.0,
        cooldown_s: float = 30.0,
    ) -> None:
        self._enabled = os.environ.get("DEXMANI_IK_PROFILE") == "1"
        self._output_dir = Path(output_dir)
        self._dls_threshold_ms = dls_threshold_ms
        self._cooldown_s = cooldown_s
        self._last_trigger_s = 0.0
        self._trigger_count = 0
        self._fallback_count = 0
        self._running_procs: list[subprocess.Popen] = []

        if not self._enabled:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Check whether py-spy can actually attach.
        scope = _ptrace_scope()
        self._pyspy_ok = scope == 0

        if not self._pyspy_ok:
            # One-shot warning: py-spy won't work; faulthandler fallback
            # will capture Python stacks only.
            msg = (
                f"IKProfiler: ptrace_scope={scope} — py-spy needs scope=0 "
                f"(run: sudo sysctl -w kernel.yama.ptrace_scope=0). "
                f"Falling back to faulthandler (Python stacks only)."
            )
            # Print to stderr rather than logger so it's visible even if
            # the logging system isn't fully up yet at import time.
            print(msg, file=sys.stderr, flush=True)

    # ── public API ──

    def maybe_profile(self, dls_per_iter_ms: float, fk_per_iter_ms: float = 0.0) -> None:
        """Trigger a profiler snapshot if DLS/iter exceeds threshold.

        Called from the IK hot path — must never raise.
        """
        if not self._enabled:
            return
        if dls_per_iter_ms < self._dls_threshold_ms:
            return

        now = time.monotonic()
        if now - self._last_trigger_s < self._cooldown_s:
            return

        self._last_trigger_s = now
        self._trigger_count += 1

        if self._pyspy_ok:
            self._trigger_pyspy(dls_per_iter_ms, fk_per_iter_ms)
        else:
            self._trigger_fallback()

    # ── internal ──

    def _trigger_pyspy(self, dls_per_iter_ms: float, fk_per_iter_ms: float) -> None:
        """Spawn background py-spy record (non-blocking, 3s, 100Hz, native)."""
        # Reap completed background profilers.
        self._running_procs = [p for p in self._running_procs if p.poll() is None]

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"ik_dls{dls_per_iter_ms:.0f}ms_fk{fk_per_iter_ms:.0f}ms_{ts}.svg"
        out = self._output_dir / fname
        err = self._output_dir / fname.replace(".svg", ".err")
        pid = os.getpid()

        try:
            with open(err, "w") as err_fh:
                proc = subprocess.Popen(
                    [
                        "py-spy",
                        "record",
                        "-o",
                        str(out),
                        "--duration",
                        "3",
                        "--pid",
                        str(pid),
                        "--rate",
                        "500",
                        "--native",
                        "--idle",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=err_fh,
                )
            self._running_procs.append(proc)
        except Exception:
            # Unexpected (py-spy missing, OOM, …) — don't break IK.
            pass

    def _trigger_fallback(self) -> None:
        """Dump Python-level stacks of all threads via faulthandler.

        Writes to a timestamped file so multi-trigger sessions are preserved.
        No C-level frames — but GC sweeps, lock contention, and unexpected
        re-entrancy are visible.
        """
        self._fallback_count += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = self._output_dir / f"ik_fallback_{ts}_#{self._fallback_count}.txt"
        try:
            with open(out, "w") as fh:
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except Exception:
            pass
