"""Synchronous camera sidecars, owned by the recorder process."""
from dataclasses import dataclass, field
from pathlib import Path
import h5py
import numpy as np
from dexmani_real.recording.storage.video import VideoEncoder, VideoEncoderConfig


@dataclass(frozen=True)
class CameraStreamWriterConfig:
    rgb_shape: tuple[int, int, int]
    depth_shape: tuple[int, int]
    fps: float
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)

    def __post_init__(self):
        if len(self.rgb_shape) != 3 or self.rgb_shape[2] != 3 or self.depth_shape != self.rgb_shape[:2]:
            raise ValueError("camera writer requires matching RGB and depth image planes")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("camera writer fps must be finite and positive")


class CameraStreamWriter:
    def __init__(self, directory, config):
        self.config = config
        self.frame_count = 0
        self.encoder = self.depth_file = None
        self.resources_released = False
        directory = Path(directory)
        try:
            h,w,_ = config.rgb_shape
            self.encoder = VideoEncoder(directory / "rgb.mp4", config=config.video,
                fps=config.fps, width=w, height=h)
            self.depth_file = h5py.File(directory / "depth.h5", "w")
            self.depth = self.depth_file.create_dataset("depth", shape=(0,*config.depth_shape),
                maxshape=(None,*config.depth_shape), chunks=(1,*config.depth_shape),
                dtype=np.uint16, compression="gzip", compression_opts=1)
        except Exception:
            self.close()
            raise

    def write(self, rgb, depth):
        if self.resources_released:
            raise RuntimeError("camera writer is closed")
        if (rgb.shape != self.config.rgb_shape or rgb.dtype != np.uint8
                or depth.shape != self.config.depth_shape or depth.dtype != np.uint16):
            raise ValueError("RGB/depth shape or dtype mismatch")
        self.encoder.write_frame(rgb)
        self.depth.resize(self.frame_count + 1, axis=0)
        self.depth[self.frame_count] = depth
        self.frame_count += 1

    def close(self):
        # The parent process enforces the finalization deadline if disk/codec IO blocks.
        failures = []
        for name in ("encoder", "depth_file"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close()
                    setattr(self, name, None)
                except Exception as exc:
                    failures.append(str(exc))
        self.resources_released = self.encoder is None and self.depth_file is None
        if failures:
            raise RuntimeError("; ".join(failures))
