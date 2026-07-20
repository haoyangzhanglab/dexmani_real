"""SharedMemoryFrameManager — centralized access to VR shared memory ring buffers.

Provides typed read/write access to VR ring buffers across process boundaries.
The main process uses this as the central hub for VR sensor data; the VR
process writes into the same named blocks.

Camera frames are handled directly by CameraProcess/CameraRingBuffer — this
manager only manages the camera SHM lifecycle (create/close/unlink), not
per-frame read/write.

Usage:
    # Main process (creator)
    mgr = SharedMemoryFrameManager(camera_hw=(480, 640), n_cameras=1)
    vr_frame = mgr.read_latest_vr()

    # VR process (attacher)
    mgr = SharedMemoryFrameManager(create=False)
    mgr.write_vr_frame(vr_array)
"""

from __future__ import annotations

from dexmani_real.shm.layouts import VR_FRAME_DTYPE, array_to_vr_frame, vr_frame_to_array
from dexmani_real.shm.ring_buffer import CameraRingBuffer, SharedMemoryRingBuffer
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Default shared memory names
_VR_SHM_NAME = "dexmani_vr_frames"
_CAM_SHM_PREFIX = "dexmani_cam_{}"


class SharedMemoryFrameManager:
    """Centralized manager for all shared memory sensor buffers.

    Creates (or attaches to) named SharedMemoryRingBuffer instances for
    VR frames and N camera frames. All buffer access is zero-copy at the
    numpy level; copies are only made when returning dicts to callers.
    """

    def __init__(
        self,
        camera_hw: tuple[int, int] = (480, 640),
        n_cameras: int = 1,
        vr_maxlen: int = 3,
        cam_maxlen: int = 5,
        create: bool = True,
        vr_name: str = _VR_SHM_NAME,
        cam_prefix: str = _CAM_SHM_PREFIX,
    ) -> None:
        self._camera_hw = camera_hw
        self._n_cameras = n_cameras
        self._create = create

        # VR ring buffer (small, ~600B per slot)
        self._vr_buf: SharedMemoryRingBuffer | None = None
        if create:
            try:
                self._vr_buf = SharedMemoryRingBuffer(vr_name, VR_FRAME_DTYPE, maxlen=vr_maxlen, create=True)
            except FileExistsError:
                # Already created by another process — attach
                self._vr_buf = SharedMemoryRingBuffer(vr_name, VR_FRAME_DTYPE, maxlen=vr_maxlen, create=False)
        else:
            self._vr_buf = SharedMemoryRingBuffer(vr_name, VR_FRAME_DTYPE, maxlen=vr_maxlen, create=False)

        # Camera ring buffers (large, ~1.5MB per slot)
        h, w = camera_hw
        self._cam_bufs: list[CameraRingBuffer | None] = []
        for i in range(n_cameras):
            name = cam_prefix.format(i)
            if create:
                try:
                    buf = CameraRingBuffer(
                        name,
                        rgb_shape=(h, w, 3),
                        depth_shape=(h, w),
                        maxlen=cam_maxlen,
                        create=True,
                    )
                except FileExistsError:
                    buf = CameraRingBuffer(
                        name,
                        rgb_shape=(h, w, 3),
                        depth_shape=(h, w),
                        maxlen=cam_maxlen,
                        create=False,
                    )
            else:
                buf = CameraRingBuffer(
                    name,
                    rgb_shape=(h, w, 3),
                    depth_shape=(h, w),
                    maxlen=cam_maxlen,
                    create=False,
                )
            self._cam_bufs.append(buf)

    # ------------------------------------------------------------------
    # VR frame access
    # ------------------------------------------------------------------

    def write_vr_frame(self, vr_dict: dict) -> int:
        """Write a VR frame dict to shared memory (producer-side).

        Returns the sequence number assigned to this frame.
        """
        if self._vr_buf is None:
            raise RuntimeError("VR buffer not initialized")
        arr = vr_frame_to_array(vr_dict)
        return self._vr_buf.write(arr)

    def read_latest_vr(self) -> dict | None:
        """Read the latest VR frame as a dict (consumer-side).

        Returns a dict matching QuestHandTracker.get_latest() output,
        or None if no frame is available.
        """
        if self._vr_buf is None:
            return None
        arr = self._vr_buf.read_latest()
        if arr is None:
            return None
        return array_to_vr_frame(arr)

    def read_latest_vr_with_age(self) -> tuple[dict | None, float]:
        """Read the latest VR frame with age in seconds.

        Returns (frame_dict_or_None, age_s).
        """
        if self._vr_buf is None:
            return None, float("inf")
        age_ns = self._vr_buf.frame_age_ns()
        if age_ns < 0:
            return None, float("inf")
        frame = self.read_latest_vr()
        return frame, age_ns * 1e-9

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def vr_age_s(self) -> float:
        if self._vr_buf is None:
            return float("inf")
        age_ns = self._vr_buf.frame_age_ns()
        if age_ns < 0:
            return float("inf")
        return age_ns * 1e-9

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._vr_buf is not None:
            self._vr_buf.close()
        for buf in self._cam_bufs:
            if buf is not None:
                buf.close()

    def unlink(self) -> None:
        """Destroy all shared memory blocks (only call from creator)."""
        if self._vr_buf is not None:
            try:
                self._vr_buf.unlink()
            except FileNotFoundError:
                pass
        for buf in self._cam_bufs:
            if buf is not None:
                try:
                    buf.unlink()
                except FileNotFoundError:
                    pass
