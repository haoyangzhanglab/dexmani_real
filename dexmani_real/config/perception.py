"""点云处理、手表面模板与 HOI 跟踪的实验参数。"""

from dataclasses import dataclass

import numpy as np

HOI_NUM_POINTS = 1024
HOI_MIN_POINTS_PER_OBJECT = 64
HOI_MAX_OBJECTS = HOI_NUM_POINTS // HOI_MIN_POINTS_PER_OBJECT


@dataclass(frozen=True)
class HOITrackingConfig:
    """实验参数在 HOI 入口转换为后端的有效配置。"""

    segmentation_interval_frames: int = 10
    recovery_enabled: bool = True
    ostrack_checkpoint: str = "third_party/weights/ostrack/OSTrack_ep0300.pth.tar"
    lost_timeout_s: float = 0.1
    recovery_retry_interval_frames: int = 2
    xy_prior_gain: float = 0.2
    kalman_xy_prior: bool = False
    recovery_crop: bool = True


@dataclass(frozen=True)
class PointCloudConfig:
    remove_table: bool = True
    num_points: int = 1024
    table_margin_m: float = 0.010
    workspace: tuple[float, ...] = (0.15, -0.50, 0.0, 0.85, 0.50, 0.80)

    def workspace_bounds(self) -> np.ndarray:
        workspace = np.asarray(self.workspace, dtype=np.float64)
        if (
            workspace.shape != (6,)
            or not np.isfinite(workspace).all()
            or np.any(workspace[:3] >= workspace[3:])
        ):
            raise ValueError("workspace must contain finite ordered xyz lower/upper bounds")
        return workspace.reshape(2, 3)

    def validate(self) -> np.ndarray:
        """校验配置，返回工作空间的 XYZ 下界和上界。"""
        if type(self.remove_table) is not bool:
            raise TypeError("remove_table must be bool")
        if type(self.num_points) is not int or self.num_points < 1:
            raise ValueError("num_points must be a positive integer")
        bounds = self.workspace_bounds()
        if self.remove_table and (not np.isfinite(self.table_margin_m) or self.table_margin_m < 0):
            raise ValueError("table_margin_m must be finite and nonnegative")
        return bounds


@dataclass(frozen=True)
class HandCloudConfig:
    num_points: int = 1024

    def validate(self) -> None:
        if type(self.num_points) is not int or self.num_points < 1:
            raise ValueError("hand_pointcloud.num_points must be a positive integer")
