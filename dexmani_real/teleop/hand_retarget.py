import numpy as np
from typing import Optional
from dex_retargeting.retargeting_config import RetargetingConfig
from dexmani_real import ASSET_DIR, CONFIG_DIR


class XHandRetargeter:

    def __init__(
            self, 
            fixed_joint_values: np.ndarray=np.array([]), 
    ):
        self.hand_type = "right"
        self.retargeting_type = "dexpilot"
        self.fixed_joint_values = np.array(fixed_joint_values)

        sapien_joint_names = [
            "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
            "right_hand_mid_joint1", "right_hand_mid_joint2",
            "right_hand_ring_joint1", "right_hand_ring_joint2",
            "right_hand_pinky_joint1", "right_hand_pinky_joint2"
        ]

        # 加载重定向配置
        file_path = CONFIG_DIR / "retargeting" / f"xhand_{self.hand_type}_{self.retargeting_type}.yml"
        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        self.retargeter = RetargetingConfig.load_from_file(str(file_path)).build()
        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        try:
            self.retargeted_joint_order = np.array(
                [retargeter_joint_names.index(name) for name in sapien_joint_names]
            ).astype(int)
        except ValueError as e:
            print(f"Error: Joint name mismatch - {e}")
            raise

    def retarget(self, hand_joint_pos: np.ndarray) -> Optional[np.ndarray]:
        if self.retargeting_type == "position":
            ref_value = hand_joint_pos[self.indices, :]
        else:
            origin_indices = self.indices[0, :]
            task_indices = self.indices[1, :]
            ref_value = hand_joint_pos[task_indices, :] - hand_joint_pos[origin_indices, :]

        qpos = self.retargeter.retarget(ref_value, fixed_qpos=self.fixed_joint_values)
        if qpos is None:
            print("Warning: Retargeting returned None.")
            return None

        qpos = np.array(qpos)[self.retargeted_joint_order]
        return qpos



def retarget_example():
    import cv2
    from dexmani_real.teleop.camera_teleop import SingleHandDetector
    from dexmani_real.robot_interface.xhand import XHand, XHandConfig

    
    cap = cv2.VideoCapture(0)
    hand_retargeter = XHandRetargeter()
    camera_teleop = SingleHandDetector(hand_type="Right")

    xhand = XHand(config=XHandConfig(
        comm_type="EtherCAT",
        verbose=True,
    ))
    xhand.start()
    print(xhand.get_meta_info())

    action = xhand.last_qpos

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
            action = xhand.last_qpos  # 如果重定向失败，保持上一个动作不变
        
        xhand.send_action(qpos=action)

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