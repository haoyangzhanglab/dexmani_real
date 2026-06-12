"""Camera-based hand detection → XHand retargeting demo.

Requires: camera + XHand hardware.
"""

import cv2

from dexmani_real.robot.xhand import XHand, XHandConfig
from dexmani_real.teleop.camera_teleop import SingleHandDetector
from dexmani_real.teleop.hand_retarget import XHandRetargeter


def retarget_example():
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
            action = xhand.last_qpos_cmd

        xhand.send_action(action)

        cv2.imshow("Hand Detection", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    xhand.reset()
    xhand.disconnect()


if __name__ == "__main__":
    retarget_example()
