import os
import math
import numpy as np
import mplib as mp


JOINT_VEL_LIMITS = np.deg2rad([45.0, 45.0, 45.0, 45.0, 60.0, 60.0, 90.0])
JOINT_ACC_LIMITS = JOINT_VEL_LIMITS * 1.5
IK_COST_WEIGHTS = np.array([3.0, 1.0, 1.0, 1.0, 2.5, 1.0, 2.5], dtype=float)
PREFERRED_LIMITS_DEG = np.array(
    [
        [-180.0, 180.0],
        [-117.0, 116.0],
        [-180.0, 180.0],
        [-6.0, 225.0],
        [-180.0, 180.0],
        [-97.0, 180.0],
        [-180.0, 180.0],
    ],
    dtype=float,
)
STEP_LIMITS_DEG = np.array([45.0, 35.0, 45.0, 35.0, 45.0, 35.0, 45.0], dtype=float)
EXCURSION_LIMITS_DEG = np.array([270.0, 140.0, 270.0, 220.0, 270.0, 180.0, 270.0], dtype=float)
TOTAL_MOTION_LIMITS_DEG = np.array([330.0, 180.0, 330.0, 260.0, 330.0, 220.0, 330.0], dtype=float)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    qv = np.concatenate(([0.0], np.asarray(v, dtype=float)))
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quat((1.0 - t) * q0 + t * q1)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    return normalize_quat(w0 * q0 + w1 * q1)


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def pose_mul(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p1, dtype=float) + quat_rotate(np.asarray(q1, dtype=float), np.asarray(p2, dtype=float))
    q = normalize_quat(quat_mul(np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)))
    return p, q


def pose_inv(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_inv = quat_conj(normalize_quat(np.asarray(q, dtype=float)))
    p_inv = -quat_rotate(q_inv, np.asarray(p, dtype=float))
    return p_inv, q_inv


class XArm7Planner:
    def __init__(
        self,
        urdf_path: str,
        srdf_path: str,
        move_group: str = "custom_link_eef",
        joint_vel_limits: np.ndarray | None = None,
        joint_acc_limits: np.ndarray | None = None,
        mp_dt: float = 0.005,
        root_position: list[float] | np.ndarray | None = None,
        root_orientation_wxyz: list[float] | np.ndarray | None = None,
        ik_cost_weights: np.ndarray | None = None,
        preferred_limits_deg: np.ndarray | None = None,
        step_limits_deg: np.ndarray | None = None,
        excursion_limits_deg: np.ndarray | None = None,
        total_motion_limits_deg: np.ndarray | None = None,
    ):
        if mp is None:
            raise ImportError("mplib is required. Install with: pip install mplib")
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(urdf_path)
        if not os.path.isfile(srdf_path):
            raise FileNotFoundError(srdf_path)

        self.mp_dt = float(mp_dt)
        self.vel_limits = np.asarray(joint_vel_limits if joint_vel_limits is not None else JOINT_VEL_LIMITS, dtype=float)
        self.acc_limits = np.asarray(joint_acc_limits if joint_acc_limits is not None else JOINT_ACC_LIMITS, dtype=float)
        self.ik_cost_weights = np.asarray(ik_cost_weights if ik_cost_weights is not None else IK_COST_WEIGHTS, dtype=float)
        self.preferred_limits = np.deg2rad(np.asarray(preferred_limits_deg if preferred_limits_deg is not None else PREFERRED_LIMITS_DEG, dtype=float))
        self.step_limits = np.deg2rad(np.asarray(step_limits_deg if step_limits_deg is not None else STEP_LIMITS_DEG, dtype=float))
        self.excursion_limits = np.deg2rad(np.asarray(excursion_limits_deg if excursion_limits_deg is not None else EXCURSION_LIMITS_DEG, dtype=float))
        self.total_motion_limits = np.deg2rad(np.asarray(total_motion_limits_deg if total_motion_limits_deg is not None else TOTAL_MOTION_LIMITS_DEG, dtype=float))

        self.mp_planner = mp.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            move_group=move_group,
            use_convex=False,
            joint_vel_limits=self.vel_limits.tolist(),
            joint_acc_limits=self.acc_limits.tolist(),
        )
        self.joint_limits = self.mp_planner.joint_limits.copy()
        self.root_position = np.zeros(3, dtype=float)
        self.root_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.set_root_pose(
            [0.0, 0.0, 0.0] if root_position is None else root_position,
            [1.0, 0.0, 0.0, 0.0] if root_orientation_wxyz is None else root_orientation_wxyz,
        )

    def print_plan_debug(self, verbose: bool, title: str, **kwargs):
        if not verbose:
            return
        print(f"[planner] {title}")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    def make_pose(self, position: np.ndarray, orientation_wxyz: np.ndarray):
        return mp.Pose(p=np.asarray(position, dtype=float), q=normalize_quat(np.asarray(orientation_wxyz, dtype=float)))

    def set_root_pose(self, position: list[float] | np.ndarray, orientation_wxyz: list[float] | np.ndarray):
        self.root_position = np.asarray(position, dtype=float)
        self.root_orientation_wxyz = normalize_quat(np.asarray(orientation_wxyz, dtype=float))
        self.mp_planner.set_base_pose(self.make_pose(self.root_position, self.root_orientation_wxyz))

    def world_to_root(self, position: list[float] | np.ndarray, orientation_wxyz: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        root_inv_p, root_inv_q = pose_inv(self.root_position, self.root_orientation_wxyz)
        return pose_mul(root_inv_p, root_inv_q, np.asarray(position, dtype=float), np.asarray(orientation_wxyz, dtype=float))

    def root_to_world(self, position: list[float] | np.ndarray, orientation_wxyz: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pose_mul(self.root_position, self.root_orientation_wxyz, np.asarray(position, dtype=float), np.asarray(orientation_wxyz, dtype=float))

    def set_vel_limits(self, limits: list[float] | np.ndarray):
        self.vel_limits = np.asarray(limits, dtype=float)
        self.mp_planner.joint_vel_limits = self.vel_limits

    def set_acc_limits(self, limits: list[float] | np.ndarray):
        self.acc_limits = np.asarray(limits, dtype=float)
        self.mp_planner.joint_acc_limits = self.acc_limits

    def set_ik_cost_weights(self, weights: list[float] | np.ndarray):
        self.ik_cost_weights = np.asarray(weights, dtype=float)

    def set_motion_preferences(
        self,
        preferred_limits_deg: list[list[float]] | np.ndarray | None = None,
        step_limits_deg: list[float] | np.ndarray | None = None,
        excursion_limits_deg: list[float] | np.ndarray | None = None,
        total_motion_limits_deg: list[float] | np.ndarray | None = None,
    ):
        if preferred_limits_deg is not None:
            self.preferred_limits = np.deg2rad(np.asarray(preferred_limits_deg, dtype=float))
        if step_limits_deg is not None:
            self.step_limits = np.deg2rad(np.asarray(step_limits_deg, dtype=float))
        if excursion_limits_deg is not None:
            self.excursion_limits = np.deg2rad(np.asarray(excursion_limits_deg, dtype=float))
        if total_motion_limits_deg is not None:
            self.total_motion_limits = np.deg2rad(np.asarray(total_motion_limits_deg, dtype=float))

    def wrap_angles(self, qpos: np.ndarray, ref_qpos: np.ndarray) -> np.ndarray:
        period = 2.0 * np.pi
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        k = np.round((ref_qpos - qpos) / period)
        k_min = np.ceil((low - qpos) / period)
        k_max = np.floor((high - qpos) / period)
        k = np.clip(k, k_min, k_max)
        return qpos + k * period

    def forward_kinematics(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pm = self.mp_planner.pinocchio_model
        pm.compute_forward_kinematics(np.asarray(qpos, dtype=float))
        pose = pm.get_link_pose(self.mp_planner.move_group_link_id)
        return self.root_to_world(np.asarray(pose.p, dtype=float), np.asarray(pose.q, dtype=float))

    def ik_cost(self, qpos: np.ndarray, ref_qpos: np.ndarray) -> float:
        delta = np.asarray(qpos, dtype=float) - np.asarray(ref_qpos, dtype=float)
        cost = float(np.sum(self.ik_cost_weights * delta * delta))
        lo = self.preferred_limits[:, 0]
        hi = self.preferred_limits[:, 1]
        outside = np.maximum(lo - qpos, 0.0) + np.maximum(qpos - hi, 0.0)
        cost += float(20.0 * np.sum(outside * outside))
        return cost

    def unwrap_path(self, path: np.ndarray, ref_qpos: np.ndarray) -> np.ndarray:
        path = np.asarray(path, dtype=float)
        if len(path) == 0:
            return path.reshape(0, len(ref_qpos))
        out = np.empty_like(path)
        out[0] = self.wrap_angles(path[0], ref_qpos)
        for i in range(1, len(path)):
            out[i] = self.wrap_angles(path[i], out[i - 1])
        return out

    def path_total_motion(self, path: np.ndarray) -> np.ndarray:
        if len(path) < 2:
            return np.zeros(path.shape[1], dtype=float)
        return np.sum(np.abs(np.diff(path, axis=0)), axis=0)

    def path_excursion(self, path: np.ndarray, start_qpos: np.ndarray) -> np.ndarray:
        if len(path) == 0:
            return np.zeros_like(start_qpos)
        return np.max(np.abs(path - start_qpos[None, :]), axis=0)

    def path_step_motion(self, path: np.ndarray) -> np.ndarray:
        if len(path) < 2:
            return np.zeros(path.shape[1], dtype=float)
        return np.max(np.abs(np.diff(path, axis=0)), axis=0)

    def path_debug_dict(self, path: np.ndarray, start_qpos: np.ndarray) -> dict:
        return {
            "n_waypoints": int(len(path)),
            "max_step_deg": np.rad2deg(self.path_step_motion(path)).round(2).tolist(),
            "max_excursion_deg": np.rad2deg(self.path_excursion(path, start_qpos)).round(2).tolist(),
            "total_motion_deg": np.rad2deg(self.path_total_motion(path)).round(2).tolist(),
        }

    def path_reject_reason(self, path: np.ndarray, start_qpos: np.ndarray) -> str | None:
        if len(path) == 0:
            return "empty_path"
        if len(path) >= 2:
            step = self.path_step_motion(path)
            if np.any(step > self.step_limits):
                idx = int(np.argmax(step / self.step_limits))
                return f"step_limit_j{idx + 1}"
        excursion = self.path_excursion(path, start_qpos)
        if np.any(excursion > self.excursion_limits):
            idx = int(np.argmax(excursion / self.excursion_limits))
            return f"excursion_limit_j{idx + 1}"
        total_motion = self.path_total_motion(path)
        if np.any(total_motion > self.total_motion_limits):
            idx = int(np.argmax(total_motion / self.total_motion_limits))
            return f"total_motion_limit_j{idx + 1}"
        return None

    def path_is_valid(self, path: np.ndarray, start_qpos: np.ndarray) -> bool:
        return self.path_reject_reason(path, start_qpos) is None

    def inverse_kinematics(
        self,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        current_qpos: np.ndarray,
        n_seeds: int = 25,
        threshold: float = 1e-3,
    ) -> np.ndarray | None:
        goal_pos_root, goal_quat_root = self.world_to_root(goal_pos, goal_quat)
        status, qpos = self.mp_planner.IK(
            goal_pose=self.make_pose(goal_pos_root, goal_quat_root),
            start_qpos=current_qpos,
            mask=None,
            n_init_qpos=n_seeds,
            threshold=threshold,
            return_closest=True,
        )
        if status != "Success":
            return None
        return self.wrap_angles(qpos, current_qpos)

    def inverse_kinematics_multi(
        self,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        current_qpos: np.ndarray,
        n_perturb: int = 15,
        perturb_scale: float = 0.1,
        verbose: bool = False,
    ) -> list[np.ndarray]:
        goal_pos_root, goal_quat_root = self.world_to_root(goal_pos, goal_quat)
        candidates: list[np.ndarray] = []

        def collect(start_qpos: np.ndarray):
            status, qpos = self.mp_planner.IK(
                goal_pose=self.make_pose(goal_pos_root, goal_quat_root),
                start_qpos=start_qpos,
                n_init_qpos=1,
                threshold=1e-3,
                return_closest=True,
            )
            if status == "Success":
                qpos = self.wrap_angles(qpos, current_qpos)
                if not any(np.allclose(qpos, c, atol=1e-6) for c in candidates):
                    candidates.append(qpos)

        collect(current_qpos)
        for _ in range(n_perturb):
            noise = np.random.uniform(-perturb_scale, perturb_scale, size=current_qpos.shape)
            collect(np.clip(current_qpos + noise, self.joint_limits[:, 0], self.joint_limits[:, 1]))

        candidates.sort(key=lambda q: self.ik_cost(q, current_qpos))
        if verbose:
            self.print_plan_debug(verbose, "ik candidates", count=len(candidates))
            for idx, q in enumerate(candidates):
                self.print_plan_debug(
                    verbose,
                    f"candidate {idx}",
                    cost=round(self.ik_cost(q, current_qpos), 6),
                    qpos_deg=np.rad2deg(q).round(2).tolist(),
                )
        return candidates

    def plan_qpos_path(self, goal_qpos: np.ndarray, current_qpos: np.ndarray) -> np.ndarray | None:
        result = self.mp_planner.plan_qpos(
            goal_qposes=[goal_qpos],
            current_qpos=current_qpos,
            time_step=self.mp_dt,
        )
        if result.get("status", "") != "Success":
            return None
        path = np.asarray(result.get("position", np.array([])), dtype=float)
        if len(path) == 0:
            return current_qpos.reshape(1, -1)
        return self.unwrap_path(path, current_qpos)

    def plan_joint_first(self, goal_pos: np.ndarray, goal_quat: np.ndarray, current_qpos: np.ndarray, verbose: bool = False) -> np.ndarray | None:
        candidates = self.inverse_kinematics_multi(goal_pos, goal_quat, current_qpos, verbose=verbose)
        if len(candidates) == 0:
            self.print_plan_debug(verbose, "joint_first", status="no_ik_candidates")
            return None

        best_path = None
        best_score = None
        best_idx = -1
        for idx, q_goal in enumerate(candidates):
            path = self.plan_qpos_path(q_goal, current_qpos)
            if path is None:
                self.print_plan_debug(verbose, f"joint_first candidate {idx}", status="plan_qpos_failed")
                continue
            reject_reason = self.path_reject_reason(path, current_qpos)
            debug = self.path_debug_dict(path, current_qpos)
            score = self.ik_cost(q_goal, current_qpos) + float(np.sum(self.path_total_motion(path)))
            self.print_plan_debug(
                verbose,
                f"joint_first candidate {idx}",
                score=round(score, 6),
                reject_reason=reject_reason,
                **debug,
            )
            if reject_reason is not None:
                continue
            if best_score is None or score < best_score:
                best_path = path
                best_score = score
                best_idx = idx

        if best_path is None:
            self.print_plan_debug(verbose, "joint_first", status="all_candidates_rejected")
            return None
        self.print_plan_debug(verbose, "joint_first selected", idx=best_idx, score=round(float(best_score), 6))
        return best_path

    def cartesian_waypoints(
        self,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        current_qpos: np.ndarray,
        pos_step: float = 0.01,
        rot_step_deg: float = 5.0,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        start_pos, start_quat = self.forward_kinematics(current_qpos)
        dist = float(np.linalg.norm(np.asarray(goal_pos, dtype=float) - start_pos))
        rot = quat_angle(start_quat, np.asarray(goal_quat, dtype=float))
        steps_pos = max(1, int(math.ceil(dist / pos_step)))
        steps_rot = max(1, int(math.ceil(rot / math.radians(rot_step_deg))))
        steps = max(steps_pos, steps_rot)
        waypoints = []
        for i in range(1, steps + 1):
            t = i / steps
            pos = (1.0 - t) * start_pos + t * np.asarray(goal_pos, dtype=float)
            quat = quat_slerp(start_quat, np.asarray(goal_quat, dtype=float), t)
            waypoints.append((pos, quat))
        return waypoints

    def plan_cartesian_tracking(self, goal_pos: np.ndarray, goal_quat: np.ndarray, current_qpos: np.ndarray, verbose: bool = False) -> np.ndarray | None:
        q_prev = np.asarray(current_qpos, dtype=float).copy()
        path = [q_prev.copy()]
        waypoints = self.cartesian_waypoints(goal_pos, goal_quat, q_prev)
        self.print_plan_debug(verbose, "cartesian_tracking", n_waypoints=len(waypoints))
        for idx, (pos, quat) in enumerate(waypoints):
            candidates = self.inverse_kinematics_multi(pos, quat, q_prev, n_perturb=8, perturb_scale=0.05, verbose=False)
            if len(candidates) == 0:
                self.print_plan_debug(verbose, "cartesian_tracking failed", waypoint=idx, reason="no_ik_candidates")
                return None
            q_next = candidates[0]
            if np.any(np.abs(q_next - q_prev) > self.step_limits):
                self.print_plan_debug(
                    verbose,
                    "cartesian_tracking failed",
                    waypoint=idx,
                    reason="step_limit",
                    step_deg=np.rad2deg(np.abs(q_next - q_prev)).round(2).tolist(),
                )
                return None
            path.append(q_next.copy())
            q_prev = q_next
        path_array = np.asarray(path, dtype=float)
        reject_reason = self.path_reject_reason(path_array, current_qpos)
        self.print_plan_debug(verbose, "cartesian_tracking raw_path", reject_reason=reject_reason, **self.path_debug_dict(path_array, current_qpos))
        if reject_reason is not None:
            return None
        return path_array

    def apply_topp(self, path: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return path, np.zeros_like(path), 0.0
        result = self.mp_planner.TOPP(path, step=self.mp_dt)
        if len(result) < 5:
            return None
        _, positions, velocities, _, duration = result
        positions = self.unwrap_path(np.asarray(positions, dtype=float), path[0])
        return positions, np.asarray(velocities, dtype=float), float(duration)

    def plan_path(
        self,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        current_qpos: np.ndarray,
        plan_mode: str = "joint_first",
        verbose: bool = False,
    ) -> np.ndarray | None:
        goal_pos = np.asarray(goal_pos, dtype=float)
        goal_quat = normalize_quat(np.asarray(goal_quat, dtype=float))
        current_qpos = np.asarray(current_qpos, dtype=float)

        if plan_mode == "cartesian_tracking":
            path = self.plan_cartesian_tracking(goal_pos, goal_quat, current_qpos, verbose=verbose)
            if path is None:
                self.print_plan_debug(verbose, "plan_path fallback", from_mode="cartesian_tracking", to_mode="joint_first")
                path = self.plan_joint_first(goal_pos, goal_quat, current_qpos, verbose=verbose)
        else:
            path = self.plan_joint_first(goal_pos, goal_quat, current_qpos, verbose=verbose)

        if path is None:
            self.print_plan_debug(verbose, "plan_path", status="failed")
            return None

        topp = self.apply_topp(path)
        if topp is None:
            self.print_plan_debug(verbose, "plan_path", status="topp_failed")
            return None
        positions, _, duration = topp
        reject_reason = self.path_reject_reason(positions, current_qpos)
        self.print_plan_debug(verbose, "plan_path topp", duration=round(duration, 4), reject_reason=reject_reason, **self.path_debug_dict(positions, current_qpos))
        if reject_reason is not None:
            return None
        return positions

    def check_collision(self, qpos: np.ndarray) -> bool:
        return len(self.mp_planner.check_for_self_collision(np.asarray(qpos, dtype=float))) > 0

    def check_env_collision(self, qpos: np.ndarray) -> bool:
        return len(self.mp_planner.check_for_env_collision(np.asarray(qpos, dtype=float))) > 0