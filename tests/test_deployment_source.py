"""来源准入的离线检查；不构造模型或启动设备。"""

import unittest
from types import SimpleNamespace

from dexmani_real.config import RealConfig
from dexmani_real.deployment.config import resolve_deployment_config
from dexmani_real.runtime.observation import ObservationNeeds


class DeploymentSourceTests(unittest.TestCase):
    def info(self, fields, source=None):
        return SimpleNamespace(
            policy_name="renamed_policy", observation_fields=fields,
            saved_config={} if source is None else {"tactile_source": source},
            n_action_steps=8, future_steps=20, hand_history_steps=None,
            supported_execution_modes=("sync",), control_dt_s=0.0625,
            fingertip_link_names=("thumb", "index", "middle", "ring", "pinky"),
        )

    def test_tactile_requires_real_before_runtime_access(self):
        for field in ("finger_tactile_force", "taxel_tactile_force"):
            for source in (None, "sim", "???", "unknown"):
                with self.subTest(field=field, source=source):
                    with self.assertRaises(ValueError):
                        resolve_deployment_config(self.info((field,), source), None, None)
            resolved = resolve_deployment_config(
                self.info((field,), "real"), RealConfig(), ObservationNeeds(),
            )
            self.assertEqual(resolved.execution.action_steps, 8)

    def test_geometry_and_recording_do_not_require_policy_source(self):
        info = self.info(("joint_state", "fingertip_xyz"))
        # Recording may request tactile independently of the policy.
        needs = ObservationNeeds(required_tactile=("aggregate", "dense"))
        resolved = resolve_deployment_config(info, RealConfig(), needs)
        self.assertEqual(resolved.execution.action_steps, 8)


if __name__ == "__main__":
    unittest.main()
