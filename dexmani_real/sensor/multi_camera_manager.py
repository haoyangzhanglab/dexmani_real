"""Multi-camera manager — N camera processes with N shared memory buffers.

Manages multiple RealSense cameras simultaneously, each in its own
crash-isolated process with a dedicated SharedMemoryRingBuffer.

Ref: ManiUniCon multi-camera design (configurable N camera daemons).

HDF5 layout extension:
    Recording with N cameras produces:
      /camera_0/rgb, depth, timestamps
      /camera_1/rgb, depth, timestamps
      ...
    (Backward compatible: single camera uses /camera/ as before.)

Usage:
    mgr = MultiCameraManager([
        CameraProcessConfig(serial="123456", shm_name="dexmani_cam_0"),
        CameraProcessConfig(serial="789012", shm_name="dexmani_cam_1"),
    ])
    mgr.start_all()

    # In controller loop:
    frames = mgr.read_all_latest()
    for name, frame in frames.items():
        if frame is not None:
            recorder.add_frame(..., camera_frame=frame, camera_name=name)

    mgr.stop_all()
"""

from __future__ import annotations

from dataclasses import dataclass

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class MultiCameraConfig:
    """Configuration for MultiCameraManager.

    Note: auto_restart is accepted but not yet wired — CameraProcess has its
    own in-process reconnection logic. (See camera_process.py:216-249.)
    """

    auto_restart: bool = True


class MultiCameraManager:
    """Manages multiple camera processes with shared memory.

    Each camera has:
      - A CameraProcess (with shared memory)
      - A named SharedMemoryRingBuffer
      - A health status
    """

    def __init__(self, configs: list, cfg: MultiCameraConfig | None = None) -> None:
        """
        Args:
            configs: List of CameraProcessConfig objects (one per camera).
            cfg: Optional MultiCameraConfig for global settings.
        """
        # Import here to avoid circular dependencies
        from dexmani_real.sensor.camera_process import CameraProcess, CameraProcessConfig

        self._configs = configs
        self._cfg = cfg or MultiCameraConfig()

        self._processes: list[CameraProcess] = []
        self._names: list[str] = []

        # Build camera processes
        for i, cam_config in enumerate(configs):
            # Ensure shared memory mode is enabled and names are unique
            if isinstance(cam_config, CameraProcessConfig):
                c = cam_config
                if not c.shm_name or c.shm_name == "dexmani_cam_0":
                    c.shm_name = f"dexmani_cam_{i}"
            else:
                # Assume it's a dict-like object
                c = CameraProcessConfig(
                    camera_name=getattr(cam_config, "camera_name", f"camera_{i}"),
                    serial=getattr(cam_config, "serial", None),
                    hz=getattr(cam_config, "hz", 30.0),
                    warmup_frames=getattr(cam_config, "warmup_frames", 10),
                    timeout_ms=getattr(cam_config, "timeout_ms", 1000),
                    shm_name=f"dexmani_cam_{i}",
                )

            proc = CameraProcess(c)
            name = c.camera_name or f"camera_{i}"
            self._processes.append(proc)
            self._names.append(name)

        logger.info(
            "MultiCameraManager created: %d camera(s) — %s",
            len(self._processes),
            ", ".join(self._names),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self) -> list[bool]:
        """Start all camera processes.

        Returns a list of booleans indicating success for each camera.
        """
        results = []
        for name, proc in zip(self._names, self._processes):
            ok = proc.start()
            results.append(ok)
            if ok:
                logger.info("  Camera '%s' started.", name)
            else:
                logger.warning("  Camera '%s' failed to start.", name)
        return results

    def stop_all(self, timeout: float = 3.0) -> None:
        """Stop all camera processes."""
        for name, proc in zip(self._names, self._processes):
            try:
                proc.stop(timeout=timeout)
                logger.info("  Camera '%s' stopped.", name)
            except (ValueError, RuntimeError) as e:
                logger.warning("  Camera '%s' stop error: %s", name, e)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_all_latest(self) -> dict[str, dict | None]:
        """Read the latest frame from all cameras.

        Returns a dict mapping camera name → camera frame dict (or None).
        """
        frames: dict[str, dict | None] = {}
        for name, proc in zip(self._names, self._processes):
            try:
                frames[name] = proc.poll_latest_frame()
            except (ValueError, RuntimeError, OSError):
                logger.debug("Camera '%s' poll failed.", name)
                frames[name] = None
        return frames
