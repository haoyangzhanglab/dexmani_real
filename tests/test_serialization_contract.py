import unittest

from dexmani_real.planning.planner import MotionPlanningConfig


class SerializationContractTest(unittest.TestCase):
    def test_unknown_profile_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MotionPlanningConfig.from_dict({"unknown_field": 1})


if __name__ == "__main__":
    unittest.main()
