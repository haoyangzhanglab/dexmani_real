"""Quest VR hand tracking → XHand retargeting test script.

Requires: Quest VR device with hand-tracking-sdk running.
"""

import time

import numpy as np

from dexmani_real.robot.xhand import XHand, XHandConfig
from dexmani_real.teleop.hand_retarget import XHandRetargeter
from dexmani_real.teleop.quest_hand_tracker import QuestHandTracker
from dexmani_real.utils.hand_utils import OPERATOR2MANO_RIGHT, estimate_frame_from_hand_points

np.set_printoptions(precision=3, suppress=True)


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
                    mediapipe_wrist_rot = estimate_frame_from_hand_points(landmarks)
                    mediapipe_landmarks = landmarks @ mediapipe_wrist_rot @ OPERATOR2MANO_RIGHT
                    target_qpos = retargeter.retarget(mediapipe_landmarks)
                    if target_qpos is not None:
                        print("Retargeted qpos:", target_qpos)
                        action = target_qpos
                    else:
                        print("Retargeting failed, using last action.")
                        action = xhand.last_qpos_cmd
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
