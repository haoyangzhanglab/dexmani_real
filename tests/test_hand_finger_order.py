"""Offline checks for anatomical, SDK joint, fingertip and tactile ordering."""

import unittest

from dexmani_real.config.defaults import hand
from dexmani_real.robot.model import (
    HAND_FINGER_COUNT,
    HAND_FINGER_NAMES,
    HAND_FINGER_ORDER_ID,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_SDK_JOINT_NAMES,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
    XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID,
)


class TestHandFingerOrder(unittest.TestCase):
    def test_canonical_order(self):
        self.assertEqual(HAND_FINGER_NAMES, ("thumb", "index", "middle", "ring", "pinky"))
        self.assertEqual(HAND_FINGER_COUNT, 5)
        self.assertEqual(HAND_FINGER_ORDER_ID, "thumb_index_mid_ring_pinky")
        self.assertEqual(XHAND_TACTILE_SENSOR_FINGER_IDS, (2, 5, 7, 9, 11))
        self.assertEqual(XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID,
                         {2: 0, 5: 1, 7: 2, 9: 3, 11: 4})
        self.assertEqual(hand.fingertip_link_names, XHAND_FINGERTIP_LINK_NAMES)
        for index, sdk_name in enumerate(("thumb", "index", "mid", "ring", "pinky")):
            finger_id = XHAND_TACTILE_SENSOR_FINGER_IDS[index]
            self.assertIn(f"right_hand_{sdk_name}_", XHAND_SDK_JOINT_NAMES[finger_id])
            self.assertIn(f"right_hand_{sdk_name}_", hand.fingertip_link_names[index])


if __name__ == "__main__":
    unittest.main()
