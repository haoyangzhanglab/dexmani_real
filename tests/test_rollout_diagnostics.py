"""Offline only: no SDK instances, device connections, or physical workers."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from dexmani_real.deployment.config import with_action_steps
from dexmani_real.deployment.diagnostic_analysis import (
    analyze_session,
    compare_chunks,
    match_commands,
)
from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE, ROBOT_COMMAND_DTYPE
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.runtime.diagnostics import DiagnosticWriter, trace_sdk_call


class TraceTests(unittest.TestCase):
    def test_deployment_chunk_override_preserves_training_snapshot(self):
        original = {
            "n_action_steps": 8,
            "agent": {"horizon": 16, "n_obs_steps": 2, "n_action_steps": 8},
        }
        changed = with_action_steps(original, 12)
        self.assertEqual(changed["agent"]["n_action_steps"], 12)
        self.assertEqual(original["agent"]["n_action_steps"], 8)
        self.assertEqual(with_action_steps(original, None), original)
        for invalid in (0, 16, True):
            with self.assertRaises(ValueError):
                with_action_steps(original, invalid)
        old = np.arange(16 * 19).reshape(16, 19)
        new = old + 12 * 19
        comparison = compare_chunks(old, new, n_obs_steps=2, n_action_steps=12)
        np.testing.assert_array_equal(comparison["revision"], np.zeros(19))
        self.assertEqual(comparison["overlap"].shape, (3, 19))

    def test_copy_drain_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = DiagnosticWriter(tmp, "policy")
            x = np.arange(16 * 19).reshape(16, 19)
            trace.record("queries", pred_action=x, status="ok")
            x[:] = -1
            trace.record("queries", pred_action=x, status="longer_status")
            trace.close()
            with h5py.File(Path(tmp) / "policy.h5", "r") as f:
                self.assertTrue(f.attrs["complete"])
                self.assertEqual(f["queries/pred_action"][0, -1, -1], 303)
                self.assertEqual(f["queries/status"][1], b"longer_status")
            self.assertTrue(json.loads((Path(tmp) / "policy.status.json").read_text())["complete"])

    def test_overflow_never_waits_or_reports_complete(self):
        entered, release = threading.Event(), threading.Event()
        original = DiagnosticWriter._append

        def blocked(*args):
            entered.set()
            release.wait(3)
            return original(*args)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(DiagnosticWriter, "_append", staticmethod(blocked)),
        ):
            trace = DiagnosticWriter(tmp, "arm", capacity=1)
            trace.record("feedback", qpos=np.zeros(7))
            self.assertTrue(entered.wait(2))
            trace.record("feedback", qpos=np.ones(7))
            start = time.monotonic()
            trace.record("feedback", qpos=np.ones(7) * 2)
            self.assertLess(time.monotonic() - start, 0.1)
            release.set()
            trace.close()
            self.assertIn("overflow", trace.error)
            with h5py.File(Path(tmp) / "arm.h5", "r") as f:
                self.assertFalse(f.attrs["complete"])
                self.assertEqual(f.attrs["dropped"], 1)

    def test_disk_failure_keeps_incomplete_header(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(DiagnosticWriter, "_append", side_effect=OSError("disk unavailable")),
        ):
            trace = DiagnosticWriter(tmp, "hand")
            trace.record("feedback", qpos=np.zeros(12))
            trace.close()
            self.assertIn("disk unavailable", trace.error)
            with h5py.File(Path(tmp) / "hand.h5", "r") as f:
                self.assertFalse(f.attrs["complete"])

    def test_sdk_timing_and_exception_preserved(self):
        events = []
        trace = SimpleNamespace(record=lambda group, **data: events.append(data))
        target = np.zeros(7)

        def send(q):
            self.assertIs(q, target)
            time.sleep(0.003)
            return 0

        self.assertEqual(trace_sdk_call(trace, send, target, run_id=2, sequence=9), 0)
        self.assertGreaterEqual(events[-1]["sdk_end_ns"] - events[-1]["sdk_start_ns"], 2e6)
        with self.assertRaisesRegex(RuntimeError, "failure"):
            trace_sdk_call(
                trace,
                lambda _: (_ for _ in ()).throw(RuntimeError("failure")),
                target,
                run_id=2,
                sequence=10,
            )
        self.assertEqual(events[-1]["status"], "exception")

    def test_publication_receipt_rejection_and_disabled(self):
        written, events = [], []

        def write(frame):
            self.assertEqual(frame.dtype, ROBOT_COMMAND_DTYPE)
            written.append(frame.copy())
            return len(written)

        shared = SimpleNamespace(
            motion_lock=threading.Lock(), robot_command_ring=SimpleNamespace(write=write)
        )
        target = RobotCommand(5, np.zeros(7), np.ones(12))
        trace = SimpleNamespace(record=lambda group, **data: events.append(data))
        with patch("dexmani_real.robot.commands._command_may_cross_sdk_locked", return_value=True):
            plain = publish_command(shared, target)
            stamped = publish_command(shared, target, diagnostics=trace, context={"chunk_step": 0})
        self.assertGreater(plain, 0)
        self.assertEqual(stamped, events[-1]["publish_end_ns"])
        self.assertEqual(events[-1]["sequence"], 2)
        np.testing.assert_array_equal(written[0], written[1])
        with patch("dexmani_real.robot.commands._command_may_cross_sdk_locked", return_value=False):
            self.assertEqual(publish_command(shared, target, diagnostics=trace), 0)
        self.assertEqual(len(written), 2)
        self.assertFalse(events[-1]["accepted"])

    def test_overlap_uses_same_logical_time(self):
        old = np.repeat(np.arange(16)[:, None], 19, axis=1) * 0.01
        new = old + 0.08
        result = compare_chunks(old, new, n_obs_steps=2, n_action_steps=8)
        np.testing.assert_allclose(result["revision"], 0, atol=1e-14)
        np.testing.assert_allclose(result["overlap"], 0, atol=1e-14)
        np.testing.assert_allclose(result["boundary"], 0.01, atol=1e-14)
        new[:, 4] -= 0.06
        result = compare_chunks(old, new, n_obs_steps=2, n_action_steps=8)
        self.assertAlmostEqual(result["revision"][4], -0.06)
        self.assertAlmostEqual(result["boundary"][4], -0.05)
        self.assertEqual(result["overlap"].shape, (7, 19))

    def test_missing_sequence_distinct_from_rejected(self):
        pubs = {
            "run_id": np.array([2, 2, 2, 2]),
            "sequence": np.array([1, 2, 3, 0]),
            "accepted": np.array([True, True, True, False]),
        }
        commands = {
            "run_id": np.array([2, 2]),
            "sequence": np.array([1, 3]),
            "sdk_start_ns": np.array([100, 0]),
        }
        pairs, absent, unsent = match_commands(pubs, commands)
        self.assertEqual(pairs, [(0, 0)])
        self.assertEqual(absent, [(2, 2)])
        self.assertEqual(unsent, [(2, 3)])

    def test_policy_bridge_same_rng_same_output_single_call(self):
        import torch
        from dexmani_policy.deployment.runtime import LoadedPolicy

        calls = []

        def predict_action(obs, inference_steps):
            calls.append(inference_steps)
            full = torch.randn(1, 16, 19)
            return {"pred_action": full, "control_action": full[:, 1:9]}

        model = LoadedPolicy.__new__(LoadedPolicy)
        model.agent = SimpleNamespace(horizon=16, action_dim=19, predict_action=predict_action)
        model.info = SimpleNamespace(
            observation_fields=("joint_state",),
            n_action_steps=8,
            action_mode="joint",
            inference_steps=4,
        )
        model._device = "cpu"
        observation = {"joint_state": np.zeros((2, 19), dtype=np.float32)}
        torch.manual_seed(7)
        plain = model.predict(observation)
        state_plain = torch.get_rng_state()
        torch.manual_seed(7)
        traced = model.predict_with_diagnostics(observation)
        np.testing.assert_array_equal(plain, traced["control_action"])
        np.testing.assert_array_equal(plain, traced["pred_action"][1:9])
        self.assertTrue(torch.equal(state_plain, torch.get_rng_state()))
        self.assertEqual(calls, [4, 4])

    def test_runner_keeps_targets_and_exposes_boundary_delay(self):
        from dexmani_real.deployment.runner import PolicyRunner
        from dexmani_real.runtime.observation import ObservationRow
        from dexmani_real.runtime.safety import SafetyState

        def execute(traced):
            now = [1_000_000_000]
            targets, events, stamps = [], [], []

            def write(frame):
                targets.append(frame["arm_qpos"][0].copy())
                stamps.append(now[0])
                return len(targets)

            shared = SimpleNamespace(
                motion_lock=threading.Lock(),
                robot_command_ring=SimpleNamespace(write=write),
                **{
                    k: SimpleNamespace(value=v)
                    for k, v in dict(
                        is_running=True,
                        error_state=False,
                        estop_request=False,
                        run_id=4,
                        safety_state=int(SafetyState.RUNNING),
                        quit_requested=False,
                        run_ended_reason=0,
                    ).items()
                },
            )
            runtime = SimpleNamespace(
                arm=SimpleNamespace(
                    joint_limit_lower=np.full(7, -3), joint_limit_upper=np.full(7, 3)
                ),
                hand=SimpleNamespace(qpos_min_rad=np.full(12, -3), qpos_max_rad=np.full(12, 3)),
            )
            info = SimpleNamespace(
                n_obs_steps=2,
                n_action_steps=8,
                control_dt_s=0.0625,
                action_mode="joint",
                observation_fields=("joint_state",),
            )
            calls = []

            def predict(obs):
                calls.append(obs["joint_state"].copy())
                now[0] += 20_000_000
                full = np.repeat(np.arange(16)[:, None], 19, axis=1) * 0.001 + len(calls) * 0.01
                return {"pred_action": full, "control_action": full[1:9]}

            model = SimpleNamespace(
                predict=lambda obs: predict(obs)["control_action"], predict_with_diagnostics=predict
            )
            trace = (
                SimpleNamespace(record=lambda group, **data: events.append((group, data)))
                if traced
                else None
            )
            runner = PolicyRunner(
                shared,
                runtime,
                info,
                model_runtime=model,
                fingertip_runtime=None,
                execute=True,
                max_running_s=30,
                diagnostics=trace,
            )
            runner.run_id, runner.started_ns = 4, now[0]

            def observe():
                arm, hand = np.zeros(1, dtype=ARM_STATE_DTYPE), np.zeros(1, dtype=HAND_STATE_DTYPE)
                arm["timestamp_ns"] = hand["timestamp_ns"] = now[0]
                return ObservationRow(arm, hand, None, None, None, now[0])

            runner._read_observation = observe
            with (
                patch(
                    "dexmani_real.deployment.runner.time.monotonic_ns", side_effect=lambda: now[0]
                ),
                patch(
                    "dexmani_real.robot.commands._command_may_cross_sdk_locked", return_value=True
                ),
            ):
                for _ in range(16):
                    now[0] = max(now[0], runner.next_step_ns)
                    runner.step()
                shared.estop_request.value = True
                runner._finish_episode = lambda *a, **k: None
                runner.step()
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(targets), 16)
            return targets, stamps, events

        plain, plain_stamps, _ = execute(False)
        traced, stamps, events = execute(True)
        np.testing.assert_array_equal(plain, traced)
        self.assertEqual(plain_stamps, stamps)
        self.assertEqual(stamps[8] - stamps[7], 82_500_000)
        self.assertEqual(stamps[7] - stamps[6], 62_500_000)
        queries = [d for g, d in events if g == "queries"]
        publications = [d for g, d in events if g == "publications"]
        self.assertEqual([d["pred_index"] for d in publications], list(range(1, 9)) * 2)
        self.assertEqual(queries[0]["infer_end_ns"] - queries[0]["infer_start_ns"], 20_000_000)

    def test_analysis_synthetic_session(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            (session / "run_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "policy": {
                            "action_mode": "joint",
                            "control_dt_s": 0.0625,
                            "n_obs_steps": 2,
                            "n_action_steps": 8,
                        }
                    }
                )
            )
            writers = {
                name: DiagnosticWriter(session / "diagnostics", name)
                for name in ("policy", "arm", "hand")
            }
            for chunk in range(2):
                base = 1_000_000_000 + chunk * 520_000_000
                full = np.repeat(np.arange(16)[:, None], 19, axis=1) * 0.001 + chunk * 0.008
                writers["policy"].record(
                    "queries",
                    run_id=2,
                    chunk_id=chunk,
                    infer_start_ns=base,
                    infer_end_ns=base + 20_000_000,
                    observation_ns=[base - 62_500_000, base],
                    arm_ns=[base - 62_500_000, base],
                    hand_ns=[base - 62_500_000, base],
                    cloud_ns=[base - 62_500_000, base],
                    pred_action=full,
                    control_action=full[1:9],
                )
                for step in range(8):
                    seq = chunk * 8 + step + 1
                    stamp = base + 20_000_000 + step * 62_500_000
                    writers["policy"].record(
                        "publications",
                        run_id=2,
                        chunk_id=chunk,
                        chunk_step=step,
                        sequence=seq,
                        accepted=True,
                        publish_start_ns=stamp - 10_000,
                        publish_end_ns=stamp,
                        raw_action=full[step + 1],
                        arm_target=full[step + 1, :7],
                        hand_target=full[step + 1, 7:],
                    )
                    for owner in ("arm", "hand"):
                        send = stamp + (10_000_000 if owner == "arm" else 15_000_000)
                        writers[owner].record(
                            "commands",
                            run_id=2,
                            sequence=seq,
                            sdk_start_ns=send,
                            sdk_end_ns=send + 1_000_000,
                            status="0" if owner == "arm" else "accepted",
                        )
            for writer in writers.values():
                writer.close()
            report, _ = analyze_session(session)
            run = report["runs"]["2"]
            self.assertEqual(run["arm"]["sent"], 16)
            self.assertEqual(run["publication_interval_ms"]["boundary"]["median"], 82.5)
            self.assertEqual(run["arm"]["publish_to_sdk_lower_ms"]["median"], 10.0)
            self.assertEqual(run["hand_minus_arm_send_ms"]["median"], 5.0)
            self.assertAlmostEqual(run["boundaries"][0]["arm"]["overlap_rmse_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
