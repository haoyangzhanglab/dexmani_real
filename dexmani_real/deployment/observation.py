"""从已校验的传感快照构造策略观测。"""

import math
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.geometry.camera import camera_parameters_cv
from dexmani_real.geometry.kinematics import (
    ArmFK,
    HandFK,
    eef_poses,
)
from dexmani_real.geometry.pose import encode_eef_pose
from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.runtime.observation import ObservationNeeds

FIELD_DEPENDENCIES = {
    "joint_state": ("arm", "hand"),
    "eef_pose": ("arm", "arm_fk"),
    "fingertip_xyz": ("arm", "hand", "arm_fk", "hand_fk", "fingertips"),
    "hand_pointcloud": ("arm", "hand", "arm_fk", "hand_fk", "hand_cloud"),
    "finger_tactile_force": ("aggregate",),
    "taxel_tactile_force": ("dense",),
    "point_cloud": ("cloud",),
    "rgb": ("camera", "rgb"),
    "depth": ("camera", "depth"),
    "camera_intrinsic": ("camera_static",),
    "camera_extrinsic": ("camera_static",),
}


@dataclass(frozen=True)
class PolicyObservationRow:
    values: dict[str, np.ndarray]
    features: dict[str, np.ndarray] = field(default_factory=dict, compare=False)


class ObservationBuilder:
    def __init__(self, info, runtime):
        self.info = info
        self.fields = tuple(info.observation_fields)
        unknown = set(self.fields) - FIELD_DEPENDENCIES.keys()
        if unknown:
            raise ValueError(f"Real has no live producer for requested modalities: {sorted(unknown)}")
        self.dependencies = frozenset(
            source for name in self.fields for source in FIELD_DEPENDENCIES[name]
        )
        self.needs = ObservationNeeds(
            camera_fields=tuple(name for name in ("rgb", "depth") if name in self.dependencies),
            point_cloud="cloud" in self.dependencies,
            required_tactile=tuple(c for c in ("aggregate", "dense") if c in self.dependencies),
            camera_static="camera_static" in self.dependencies,
        )
        self.camera_parameters = {}
        self.arm_fk = ArmFK() if "arm_fk" in self.dependencies else None
        self.hand_fk = self.hand_cloud = None
        if "hand_fk" in self.dependencies:
            self.hand_fk = HandFK(
                info.fingertip_link_names
                if "fingertips" in self.dependencies else XHAND_FINGERTIP_LINK_NAMES,
            )
            if "hand_cloud" in self.dependencies:
                self.hand_cloud = HandPointCloud(self.hand_fk, runtime.hand_pointcloud.num_points)

    def project(self, row):
        fields, dependencies = self.fields, self.dependencies
        values = {}
        if "arm" in dependencies:
            values["arm_qpos"] = row.arm["qpos"][0].copy()
        if "hand" in dependencies:
            values["hand_qpos"] = row.hand["qpos"][0].copy()
        for name in fields:
            if name == "finger_tactile_force":
                # 独立复制，避免聚合触觉视图持有仅供录制的稠密载荷。
                values[name] = row.hand["tactile_aggregate"][0].copy()
            elif name == "taxel_tactile_force":
                values[name] = row.hand["tactile_dense"][0]
            elif name == "point_cloud":
                values[name] = row.point_cloud
            elif name in {"rgb", "depth"}:
                values[name] = row.camera[name]
        return PolicyObservationRow(values)

    def configure_camera(self, camera):
        self.camera_parameters = {}
        requested = set(self.fields) & {"depth", "camera_intrinsic", "camera_extrinsic"}
        if not requested:
            return
        if "depth" in requested and not math.isclose(
            camera.depth_scale_m, self.info.depth_config["depth_scale_m"], rel_tol=1e-9, abs_tol=0.0,
        ):
            raise ValueError("Live depth scale differs from the saved raw-count input recipe")
        requested.discard("depth")
        if not requested:
            return
        parameters = camera_parameters_cv(
            camera.color,
            camera.T_xarm_base_from_color if "camera_extrinsic" in requested else None,
        )
        self.camera_parameters = {name: parameters[name] for name in requested}

    def build(self, rows):
        fields, dependencies = self.fields, self.dependencies
        arrays = {}
        if "joint_state" in fields:
            arrays["joint_state"] = np.stack([
                np.concatenate((r.values["arm_qpos"], r.values["hand_qpos"])) for r in rows
            ]).astype(np.float32)
        for name in fields:
            sources = FIELD_DEPENDENCIES[name]
            if "camera_static" in sources:
                values = self.camera_parameters[name]
                arrays[name] = np.broadcast_to(values, (len(rows), values.size)).copy()
            elif name != "joint_state" and "arm_fk" not in sources:
                arrays[name] = np.stack([r.values[name] for r in rows])

        if self.arm_fk is not None:
            new_rows = [row for row in rows if not row.features]
            features = {}
            if new_rows:
                arm = np.stack([r.values["arm_qpos"] for r in new_rows])
                poses = eef_poses(arm, arm_fk=self.arm_fk)
                if "eef_pose" in fields:
                    features["eef_pose"] = encode_eef_pose(poses).astype(np.float32)
                if self.hand_fk is not None:
                    features.update(hand_geometry(
                        poses, np.stack([r.values["hand_qpos"] for r in new_rows]),
                        arm_fk=self.arm_fk, hand_fk=self.hand_fk,
                        hand_cloud=self.hand_cloud,
                        include_fingertips="fingertips" in dependencies,
                    ))
            for index, row in enumerate(new_rows):
                row.features.update({name: value[index].copy() for name, value in features.items()})
            for name in fields:
                if "arm_fk" in FIELD_DEPENDENCIES[name]:
                    arrays[name] = np.stack([r.features[name] for r in rows])

        return {name: arrays[name] for name in fields}
