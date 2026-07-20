"""XArm7 + XHand simulated robot model for SAPIEN."""

import numpy as np
import sapien.core as sapien
from transforms3d import euler

from dexmani_real import ASSET_DIR

__all__ = ["XArm7XHand"]


class XArm7XHand:
    """XArm7 + XHand simulated robot model for SAPIEN.

    Formerly XArm7_XHand (PEP 8 compliance — underscores in class names
    are reserved for leading/trailing double underscores).
    """

    def __init__(
        self,
        scene: sapien.Scene,
        disable_self_collision: bool = True,
        root_pose=sapien.Pose(p=[0.0, 0.0, 0.0], q=euler.euler2quat(0, 0, np.pi / 6)),
        arm_home_qpos=np.deg2rad([-30.0, -1.9, 0.0, 13.5, -180.0, 74.7, 0.0]),
        arm_pd_gains: dict | None = None,
        hand_pd_gains: dict | None = None,
    ):
        self.scene = scene
        self.arm_dof = 7
        self.hand_dof = 12
        self.dof = self.arm_dof + self.hand_dof
        self.load_model(disable_self_collision, root_pose)
        self.pin_model = self.model.create_pinocchio_model()  # link_names和joint_names都与self.model一致

        self.register_link_names()
        self.register_joint_names()
        self.register_home_qpos(arm_home_qpos)
        self.set_joint_pd_controller(arm_pd_gains, hand_pd_gains)
        self.set_physx_and_render_properties()

    # --------------------------------------------------------------
    # Model loading and initialization
    # --------------------------------------------------------------
    def load_model(self, disable_self_collision, root_pose: sapien.Pose):
        loader = self.scene.create_urdf_loader()
        loader.multiple_collisions_decomposition = "coacd"
        loader.multiple_collisions_decomposition_params = dict(
            threshold=0.05,
            preprocess_mode="auto",
            resolution=2000,
            mcts_nodes=20,
            mcts_iterations=150,
            mcts_max_depth=3,
            merge=True,
            seed=0,
            verbose=False,
        )

        urdf_path = ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_right.urdf"
        robot_builder = loader.load_file_as_articulation_builder(str(urdf_path))
        if disable_self_collision:
            for lb in robot_builder.link_builders:
                lb.collision_groups = [1, 1, 1, 0x1F1F]
        self.model = robot_builder.build(fix_root_link=True)
        self.model.set_name("xarm7_xhand_right")
        self.model.set_root_pose(root_pose)

    def register_link_names(self):
        names = [link.get_name() for link in self.model.get_links()]
        idx = names.index("custom_eef_link")
        self.link_names = names
        self.arm_link_names = names[: idx + 1]
        self.hand_link_names = names[idx + 1 :]
        self.fingertip_link_names = [
            "right_hand_" + n + "_tip" for n in ["thumb_rota", "index_rota", "mid", "ring", "pinky"]
        ]
        self.contact_link_names = [
            "right_hand_" + n + "_link2" for n in ["thumb_rota", "index_rota", "mid", "ring", "pinky"]
        ]
        self.imagine_pcd_link_names = ["right_hand_link"] + [
            "right_hand_" + n + j
            for n in ["thumb_rota", "index_rota", "mid", "ring", "pinky"]
            for j in ["_link1", "_link2"]
        ]
        self.imagine_pcd_link_num_points = [192] + [32] * 10

    def register_joint_names(self):
        self.arm_joint_names = [f"joint{i + 1}" for i in range(self.arm_dof)]
        self.hand_joint_names = [
            "thumb_bend_joint",
            "thumb_rota_joint1",
            "thumb_rota_joint2",
            "index_bend_joint",
            "index_joint1",
            "index_joint2",
            "mid_joint1",
            "mid_joint2",
            "ring_joint1",
            "ring_joint2",
            "pinky_joint1",
            "pinky_joint2",
        ]
        self.hand_joint_names = ["right_hand_" + n for n in self.hand_joint_names]
        custom_joint_names = self.arm_joint_names + self.hand_joint_names
        sapien_joint_names = [j.get_name() for j in self.model.get_active_joints()]
        # mapping: user->sapien     inv_mapping: sapien->user
        index_map = {name: i for i, name in enumerate(custom_joint_names)}
        self.mapping = [index_map[n] for n in sapien_joint_names]
        self.inv_mapping = np.argsort(self.mapping)

    def register_home_qpos(self, arm_home_qpos: np.ndarray):
        hand_home_qpos = np.zeros(12)
        self.home_qpos = np.concatenate([arm_home_qpos, hand_home_qpos])
        self.model.set_qpos(self.home_qpos[self.mapping])
        eef_home_pose = self.forward_kinematics(self.home_qpos, target_link_names=["custom_eef_link"])[0]
        self.eef_home_pose = sapien.Pose(p=eef_home_pose[:3], q=eef_home_pose[3:])

    def set_joint_pd_controller(self, arm_pd_gains: dict | None = None, hand_pd_gains: dict | None = None):
        arm_pd_gains = arm_pd_gains or {}
        hand_pd_gains = hand_pd_gains or {}
        active_joints = self.model.get_active_joints()
        for joint in active_joints:
            name = joint.get_name()
            if name in self.arm_joint_names:
                joint.set_drive_property(
                    # stiffness=1000, damping=120 → default PD for bare xArm7.
                    stiffness=arm_pd_gains.get("stiffness", 1000),
                    damping=arm_pd_gains.get("damping", 120),
                    force_limit=arm_pd_gains.get("force_limit", 200),
                )
            elif name in self.hand_joint_names:
                joint.set_drive_property(
                    stiffness=hand_pd_gains.get("stiffness", 500),
                    damping=hand_pd_gains.get("damping", 100),
                    force_limit=hand_pd_gains.get("force_limit", 80),
                )

    def set_physx_and_render_properties(self):
        for link in self.model.get_links():
            name = link.get_name()
            if name in self.hand_link_names:
                for geom in link.collision_shapes:
                    geom.set_patch_radius(0.05)
                    geom.set_min_patch_radius(0.04)
                    geom.set_physical_material(sapien.physx.PhysxMaterial(2.0, 1.5, 0.0))

            for component in link.get_entity().get_components():
                if isinstance(component, sapien.render.RenderBodyComponent):
                    for shape in component.render_shapes:
                        for part in shape.parts:
                            part.material.set_specular(0.6)
                            part.material.set_metallic(0.3)
                            part.material.set_roughness(0.5)

    # --------------------------------------------------------------
    # Proprioceptive state retrieval
    # --------------------------------------------------------------
    @property
    def qlimits(self):
        return self.model.get_qlimits()[self.inv_mapping]

    def get_qpos(self):
        return self.model.get_qpos()[self.inv_mapping]

    def get_eef_pose(self):
        return self.model.find_link_by_name("custom_eef_link").get_entity_pose()

    def get_palm_pose(self):
        return self.model.find_link_by_name("right_hand_ee_link").get_entity_pose()

    def get_palm_pose_from_qpos(self, qpos: np.ndarray):
        result = self.forward_kinematics(qpos, ["right_hand_ee_link"])[0]
        return sapien.Pose(p=result[:3], q=result[3:])

    def get_palm2eef_transform(self):
        palm_pose = self.get_palm_pose()
        eef_pose = self.get_eef_pose()
        palm2eef_transform = palm_pose.inv() * eef_pose
        return palm2eef_transform

    def get_link_poses(self, link_names: list[str]):
        link_poses = []
        for name in link_names:
            link_pose = self.model.find_link_by_name(name).get_entity_pose()
            link_poses.append(np.concatenate([link_pose.p, link_pose.q]))
        return np.asarray(link_poses)

    def forward_kinematics(self, qpos: np.ndarray, target_link_names: list[str]):
        self.pin_model.compute_forward_kinematics(qpos[self.mapping])
        target_link_poses = []
        for name in target_link_names:
            link_idx = self.link_names.index(name)
            link_pose = self.pin_model.get_link_pose(link_idx)
            # Transform from robot-local frame to world frame for comparison with get_link_poses
            link_pose = self.model.get_root_pose() * link_pose
            target_link_poses.append(np.concatenate([link_pose.p, link_pose.q]))
        return np.asarray(target_link_poses)

    def _try_inverse_kinematics(
        self, eef_pose: sapien.Pose, full_qpos_init: np.ndarray = None
    ) -> tuple[np.ndarray | None, bool, float]:
        eef_local_pose = self.model.get_root_pose().inv() * eef_pose
        eef_link_idx = self.link_names.index("custom_eef_link")

        if full_qpos_init is None:
            qpos_init_user = self.get_qpos()
        else:
            qpos_init_user = np.asarray(full_qpos_init, dtype=np.float64)

        qpos_init_sapien = qpos_init_user[self.mapping]

        active_qmask_user = np.concatenate(
            [
                np.ones(self.arm_dof, dtype=np.int32),
                np.zeros(self.hand_dof, dtype=np.int32),
            ]
        )
        active_qmask_sapien = active_qmask_user[self.mapping]

        ik_qpos_sapien, flag, error = self.pin_model.compute_inverse_kinematics(
            link_index=eef_link_idx,
            pose=eef_local_pose,
            initial_qpos=qpos_init_sapien,
            active_qmask=active_qmask_sapien,
            max_iterations=1000,
        )

        if not flag:
            return None, False, float(error)
        return ik_qpos_sapien[self.inv_mapping], True, float(error)

    def inverse_kinematics(self, eef_pose: sapien.Pose, full_qpos_init: np.ndarray = None):
        qpos, converged, error = self._try_inverse_kinematics(eef_pose, full_qpos_init)
        if not converged:
            raise RuntimeError(f"IK did not converge, final error: {error}")
        return qpos

    # --------------------------------------------------------------
    # Control interface
    # --------------------------------------------------------------
    def set_qpos(self, qpos: np.ndarray):
        self.model.set_qpos(qpos[self.mapping])

    def balance_passive_force(self):
        qf = self.model.compute_passive_force()
        self.model.set_qf(qf)

    def clip_action(self, action: np.ndarray):
        qlimits = self.qlimits
        action = np.clip(action, qlimits[:, 0], qlimits[:, 1])
        return action

    def apply_action(self, target_qpos: np.ndarray):
        target_qpos = self.clip_action(np.asarray(target_qpos, dtype=np.float64))
        for joint, q in zip(self.model.get_active_joints(), target_qpos[self.mapping]):
            joint.set_drive_target(q)
            joint.set_drive_velocity_target(0.0)

    def reset(self, random_init: bool = False, offset_pos: np.ndarray = np.zeros(3), offset_rpy=np.zeros(3)):
        if not random_init:
            reset_qpos = self.home_qpos
        else:
            reset_eef_pos = self.eef_home_pose.p + offset_pos
            reset_quat = euler.euler2quat(*offset_rpy)
            reset_eef_pose = sapien.Pose(p=reset_eef_pos, q=reset_quat)
            reset_qpos = self.inverse_kinematics(reset_eef_pose, full_qpos_init=self.home_qpos)

        self.set_qpos(reset_qpos)
        self.model.set_qvel(np.zeros(self.dof))
        self.balance_passive_force()
        self.apply_action(reset_qpos)
