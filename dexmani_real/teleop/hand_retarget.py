import time
import numpy as np
from typing import Optional
from dex_retargeting.retargeting_config import RetargetingConfig
from dexmani_real import ASSET_DIR, CONFIG_DIR
from dexmani_real.teleop.xhand_ref_adapter import XHandRefAdapter

class XHandRetargeter:
    def __init__(
        self,
        fixed_joint_values: Optional[np.ndarray] = None,
        hand_type: str = "right",
        retargeting_type: str = "dexpilot",
        enable_ref_adapter: bool = True,
        pinky_extension_range=(0.03, 0.07),
        pinky_scale=(1.2, 2.2),
        pinky_blend: float = 1.0,
        debug_adapters: bool = False,
    ):
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.fixed_joint_values = np.array([]) if fixed_joint_values is None else np.array(fixed_joint_values)
        self.debug_adapters = bool(debug_adapters)
        self.last_debug = {}

        self.sapien_joint_names = [
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]

        self.load_retargeter()
        self.ref_adapter = XHandRefAdapter(
            enable=enable_ref_adapter and retargeting_type != "position",
            pinky_extension_range=pinky_extension_range,
            pinky_scale=pinky_scale,
            pinky_blend=pinky_blend,
            debug=debug_adapters,
        )

    def load_retargeter(self):
        config_path = CONFIG_DIR / "retargeting" / f"xhand_{self.hand_type}_{self.retargeting_type}.yml"

        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        self.retargeter = RetargetingConfig.load_from_file(str(config_path)).build()
        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.retargeted_joint_order = np.array(
            [retargeter_joint_names.index(name) for name in self.sapien_joint_names]
        ).astype(int)

    def build_ref_value(self, hand_joint_pos: np.ndarray) -> np.ndarray:
        if self.retargeting_type == "position":
            return hand_joint_pos[self.indices, :]

        origin_indices = self.indices[0, :]
        task_indices = self.indices[1, :]

        ref_value = hand_joint_pos[task_indices, :] - hand_joint_pos[origin_indices, :]
        ref_value = self.ref_adapter.apply(
            ref_value,
            hand_joint_pos,
            origin_indices,
            task_indices,
        )

        return ref_value

    def retarget(self, hand_joint_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if hand_joint_pos is None:
            return None

        start_time = time.time()

        ref_value = self.build_ref_value(hand_joint_pos)
        qpos = self.retargeter.retarget(ref_value, fixed_qpos=self.fixed_joint_values)

        if qpos is None:
            print("Warning: Retargeting returned None.")
            return None

        qpos = np.asarray(qpos, dtype=float)[self.retargeted_joint_order]

        if self.debug_adapters:
            self.last_debug = {
                "retarget_ms": 1000 * (time.time() - start_time),
                "ref_adapter": self.ref_adapter.last_debug,
            }
            print("retarget_debug:", self.last_debug)

        return qpos




def retarget_example():
    import cv2
    from dexmani_real.teleop.camera_teleop import SingleHandDetector
    from dexmani_real.robot.xhand import XHand, XHandConfig

    
    cap = cv2.VideoCapture(0)
    hand_retargeter = XHandRetargeter()
    camera_teleop = SingleHandDetector(hand_type="Right")

    config = XHandConfig(
        comm_type="RS485",
        device_name="/dev/ttyUSB0",
    )
    xhand = XHand(config)

    if not xhand.connect():
        raise RuntimeError(f"Failed to connect XHand: {xhand.last_error_message}")

    action = xhand.last_qpos_cmd

    while True:
        # 从摄像头获取RGB图像
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        num_box, hand_joint_pos, keypoint_2d, wrist_rot = camera_teleop.detect(frame)
        img = camera_teleop.draw_skeleton_on_image(frame, keypoint_2d, style="default")

        if hand_joint_pos is not None:
            action = hand_retargeter.retarget(hand_joint_pos)
        else:
            action = xhand.last_qpos_cmd  # 如果重定向失败，保持上一个动作不变
        
        xhand.send_action(action)

        qpos = xhand.get_state(full=False)["qpos"]
        # print("Current qpos:", qpos)

        cv2.imshow("Hand Detection", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()

    xhand.reset()
    xhand.disconnect()


if __name__ == "__main__":
    retarget_example()