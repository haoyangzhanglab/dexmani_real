import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from robot.xhand import XHand, XHandConfig
from teleop.hand_retarget import XHandRetargeter
from teleop.quest_hand_tracker import QuestHandTracker


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    keypoint_3d_array = np.asarray(keypoint_3d_array, dtype=np.float64)
    assert keypoint_3d_array.shape == (21, 3)

    eps = 1e-8

    wrist = keypoint_3d_array[0]
    index_mcp = keypoint_3d_array[5]
    middle_mcp = keypoint_3d_array[9]
    ring_mcp = keypoint_3d_array[13]
    pinky_mcp = keypoint_3d_array[17]

    # 1. Use more palm points for plane fitting.
    # Original version only used [0, 5, 9].
    # This version uses wrist + four MCP joints for a more stable palm plane.
    palm_points = np.stack(
        [wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp],
        axis=0,
    )

    # Remove invalid landmarks early.
    if not np.all(np.isfinite(palm_points)):
        return np.eye(3, dtype=np.float64)

    palm_center = palm_points.mean(axis=0, keepdims=True)
    centered = palm_points - palm_center

    # Degenerate case: all palm points collapse or are nearly identical.
    if np.linalg.norm(centered) < eps:
        return np.eye(3, dtype=np.float64)

    # 2. Fit palm plane using SVD.
    # The last right-singular vector is the normal of the best-fit plane.
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)

    normal = vh[-1, :]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < eps:
        return np.eye(3, dtype=np.float64)
    normal = normal / normal_norm

    # 3. Define x axis.
    # Keep original behavior: x points from middle MCP to wrist.
    x_vector = wrist - middle_mcp

    # Project x_vector onto palm plane to make it orthogonal to normal.
    x = x_vector - np.dot(x_vector, normal) * normal
    x_norm = np.linalg.norm(x)

    # Fallback: use palm center to wrist if middle_mcp is unreliable.
    if x_norm < eps:
        x_vector = wrist - palm_center.reshape(3)
        x = x_vector - np.dot(x_vector, normal) * normal
        x_norm = np.linalg.norm(x)

    if x_norm < eps:
        return np.eye(3, dtype=np.float64)

    x = x / x_norm

    # 4. Construct lateral axis.
    z = np.cross(x, normal)
    z_norm = np.linalg.norm(z)
    if z_norm < eps:
        return np.eye(3, dtype=np.float64)
    z = z / z_norm

    # 5. Resolve sign ambiguity.
    # SVD normal has arbitrary sign. Use pinky -> index direction to orient z.
    lateral_ref = index_mcp - pinky_mcp
    lateral_ref = lateral_ref - np.dot(lateral_ref, x) * x
    lateral_norm = np.linalg.norm(lateral_ref)

    if lateral_norm > eps:
        lateral_ref = lateral_ref / lateral_norm
        if np.dot(z, lateral_ref) < 0:
            normal *= -1.0
            z *= -1.0

    # 6. Recompute normal to enforce right-handed orthonormal frame.
    # Since z = x × normal, the consistent inverse is normal = z × x.
    normal = np.cross(z, x)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < eps:
        return np.eye(3, dtype=np.float64)
    normal = normal / normal_norm

    frame = np.stack([x, normal, z], axis=1)

    # 7. Final orthogonalization via SVD.
    # This removes small numerical drift and makes frame closer to SO(3).
    try:
        u, _, vh = np.linalg.svd(frame)
        frame = u @ vh
    except np.linalg.LinAlgError:
        pass

    # Keep right-handed rotation. det should be +1.
    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1.0

    return frame



def test_quest_hand_tracker():
    tracker = QuestHandTracker(
        transport="tcp_server",
        host="0.0.0.0",
        port=8000,
        hand_side="right",
        output_frame="flu",
        verbose=True,
    )

    config = XHandConfig(
        comm_type="RS485",
        device_name="/dev/ttyUSB0",
    )
    xhand = XHand(config)

    retargeter = XHandRetargeter()

    if not xhand.connect():
        raise RuntimeError(f"Failed to connect XHand: {xhand.last_error_message}")

    try:
        with tracker:
            while True:
                frame = tracker.get_latest()
                if frame is not None:
                    landmarks = frame["landmarks"]
                    mediapipe_wrist_rot  = estimate_frame_from_hand_points(landmarks)
                    mediapipe_landmarks = landmarks @ mediapipe_wrist_rot @ OPERATOR2MANO_RIGHT
                    target_qpos = retargeter.retarget(mediapipe_landmarks)
                    if target_qpos is not None:
                        print("Retargeted qpos:", target_qpos)
                        action = target_qpos
                    else:
                        print("Retargeting failed, using last action.")
                        action = xhand.last_qpos_cmd  # 如果重定向失败，保持上一个动作不变
                    xhand.send_action(action)
                else:
                    print("No frame received")
                    time.sleep(1 / 83.0)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        xhand.reset()
        xhand.disconnect()

if __name__ == "__main__":
    test_quest_hand_tracker()