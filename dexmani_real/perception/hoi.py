"""统一在线与离线的 HOI 跟踪帧结果。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.perception import HOITrackingConfig

if TYPE_CHECKING:
    from cordvip_geometry import RGBDFrame, SceneTracker, TrackingConfig


@dataclass(frozen=True)
class HOIFrameResult:
    """各对象按模板顺序排列，位姿从物体局部坐标系变换到相机系。"""

    object_pose_camera: np.ndarray
    object_valid: np.ndarray
    quality: tuple[dict[str, float | int | None], ...]
    failures: dict[str, str]

    @property
    def valid(self) -> bool:
        return bool(self.object_valid.all())


def track_hoi_frame(
    tracker: SceneTracker,
    frame: RGBDFrame,
    template: Mapping[str, np.ndarray],
    prompts: dict[str, str],
    *,
    initialize: bool = False,
) -> HOIFrameResult:
    """按模板顺序返回当前结果，初始化证据不足时返回整帧无效。"""
    names = template["object_names"]
    poses = np.full((len(names), 4, 4), np.nan, dtype=np.float32)
    if initialize:
        from cordvip_geometry import TrackingFailure

        try:
            scene = tracker.initialize(frame, prompts)
        except TrackingFailure as error:
            return HOIFrameResult(
                poses, np.zeros(len(names), dtype=bool),
                tuple({} for _ in names), dict.fromkeys(names, str(error)),
            )
    else:
        scene = tracker.track(frame)
    valid = np.array([scene.valid[name] for name in names], dtype=bool)
    for index, name in enumerate(names):
        if valid[index]:
            poses[index] = scene.poses[name]
    return HOIFrameResult(poses, valid, tuple(scene.quality[name] for name in names), scene.failures)


def build_tracking_config(config: HOITrackingConfig) -> TrackingConfig:
    """将实验参数传入后端，其余阈值沿用后端默认值。"""
    if not isinstance(config, HOITrackingConfig):
        raise TypeError("tracking_config must be HOITrackingConfig")

    from cordvip_geometry import TrackingConfig

    return TrackingConfig(**asdict(config))
