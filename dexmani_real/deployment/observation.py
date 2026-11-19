"""从已校验的传感快照构造策略观测。"""

import cv2
import numpy as np

from dexmani_real.geometry.camera import (
    camera_extrinsic_cv,
    init_undistort_maps,
    remap_image,
)
from dexmani_real.geometry.kinematics import (
    ArmFK,
    HandFK,
    eef_poses,
)
from dexmani_real.geometry.pose import encode_eef_pose
from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
from dexmani_real.runtime.observation import ObservationNeeds

FIELD_DEPENDENCIES = {
    "object_pointcloud": ("hoi",),
    "joint_state": ("arm", "hand"),
    "eef_pose": ("arm", "arm_fk"),
    "fingertip_xyz": ("arm", "hand", "arm_fk", "hand_fk", "fingertips"),
    "hand_pointcloud": ("arm", "hand", "arm_fk", "hand_fk", "hand_cloud"),
    "finger_tactile_force": ("aggregate",),
    "taxel_tactile_force": ("dense",),
    "point_cloud": ("cloud",),
    "rgb": ("rgb",),
    "camera_intrinsic": ("camera_static",),
    "camera_extrinsic": ("camera_static",),
}

JOINT_OBSERVATION_FIELDS = frozenset(
    field for field, sources in FIELD_DEPENDENCIES.items() if {"arm", "hand"}.intersection(sources)
)


class ObservationBuilder:
    last_rgb_sequence: int | None
    last_rgb: np.ndarray | None

    def __init__(self, info, runtime):
        self.fields = tuple(info.observation_fields)
        unknown = set(self.fields) - FIELD_DEPENDENCIES.keys()
        if unknown:
            raise ValueError(f"Real has no live producer for requested modalities: {sorted(unknown)}")
        self.dependencies = frozenset(
            source for name in self.fields for source in FIELD_DEPENDENCIES[name]
        )
        self.joint_devices = tuple(name for name in ("arm", "hand") if name in self.dependencies)
        self.needs = ObservationNeeds(
            hoi="hoi" in self.dependencies,
            hoi_joint_devices=self.joint_devices if "hoi" in self.dependencies else (),
            camera_fields=("rgb",) if "rgb" in self.dependencies else (),
            point_cloud="cloud" in self.dependencies,
            required_tactile=tuple(c for c in ("aggregate", "dense") if c in self.dependencies),
        )
        self.camera_static = "camera_static" in self.dependencies
        self.camera_parameters = {}
        self.arm_fk = ArmFK() if "arm_fk" in self.dependencies else None
        self.hand_fk = self.hand_cloud = None
        if "hand_fk" in self.dependencies:
            self.hand_fk = HandFK()
            if "hand_cloud" in self.dependencies:
                self.hand_cloud = HandPointCloud(self.hand_fk, runtime.hand_pointcloud.num_points)

    def project(self, row):
        values = {}
        for device in self.joint_devices:
            name = f"{device}_qpos"
            qpos = (
                row.aligned_joints[name]
                if self.needs.hoi_joint_devices
                else getattr(row, device)["qpos"][0]
            )
            values[name] = qpos.copy()
        # Ring 快照独立持有原始大数组，RGB 矫正结果单独进入模型历史。
        for name in self.fields:
            if name == "finger_tactile_force":
                # 两种触觉已为 SDK / 255；独立复制避免聚合视图持有稠密载荷。
                values[name] = row.hand["tactile_aggregate"][0].copy()
            elif name == "taxel_tactile_force":
                values[name] = row.hand["tactile_dense"][0].copy()
            elif name == "object_pointcloud":
                values[name] = row.hoi["object_pointcloud"]
            elif name == "point_cloud":
                values[name] = row.point_cloud
            elif name == "rgb":
                sequence = row.camera["ring_sequence"]
                if sequence != self.last_rgb_sequence:
                    self.last_rgb = remap_image(
                        row.camera["rgb"], self.rgb_maps, interpolation=cv2.INTER_LINEAR,
                    )
                    self.last_rgb_sequence = sequence
                values[name] = self.last_rgb
        for value in values.values():
            value.flags.writeable = False
        return values

    def configure_camera(self, camera) -> None:
        self.camera_parameters.clear()
        self.rgb_maps = init_undistort_maps(camera.color) if "rgb" in self.fields else None
        self.last_rgb_sequence = None
        self.last_rgb = None
        for name in self.fields:
            if name == "camera_intrinsic":
                values = camera.color.matrix(dtype=np.float32).ravel()
            elif name == "camera_extrinsic":
                values = camera_extrinsic_cv(camera.T_xarm_base_from_color)
            else:
                continue
            values.setflags(write=False)
            self.camera_parameters[name] = values

    def build(self, rows):
        fields, dependencies = self.fields, self.dependencies
        # 同一批关节同时服务状态与 FK，不为每种派生模态重复拼接历史。
        joints = {
            device: np.array([row[f"{device}_qpos"] for row in rows])
            for device in self.joint_devices
        }
        arrays = {}
        if "joint_state" in fields:
            arrays["joint_state"] = np.concatenate((joints["arm"], joints["hand"]), axis=1, dtype=np.float32)
        for name in fields:
            sources = FIELD_DEPENDENCIES[name]
            if "camera_static" in sources:
                values = self.camera_parameters[name]
                arrays[name] = np.broadcast_to(values, (len(rows), values.size)).copy()
            elif name != "joint_state" and "arm_fk" not in sources:
                arrays[name] = np.stack([r[name] for r in rows])

        if self.arm_fk is not None:
            poses = eef_poses(joints["arm"], arm_fk=self.arm_fk)
            if "eef_pose" in fields:
                arrays["eef_pose"] = encode_eef_pose(poses).astype(np.float32)
            if self.hand_fk is not None:
                arrays.update(hand_geometry(
                    poses, joints["hand"],
                    arm_fk=self.arm_fk, hand_fk=self.hand_fk,
                    hand_cloud=self.hand_cloud,
                    include_fingertips="fingertips" in dependencies,
                ))

        return {name: arrays[name] for name in fields}
