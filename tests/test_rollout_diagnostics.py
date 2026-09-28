"""Offline only: no SDK instances, device connections, or physical workers."""

import json
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from copy import deepcopy
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
from dexmani_real.robot.commands import RobotCommand, publish_command, read_robot_command
from dexmani_real.runtime.diagnostics import DiagnosticWriter, trace_sdk_call
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    command_may_cross_sdk,
    request_policy_stop,
)


@contextmanager
def runner_case(*, traced=True, inference_ns=20_000_000, diagnostic_ns=0):
    """Existing runner fixture with an explicit clock and no hardware owners."""
    from dexmani_real.deployment.runner import PolicyRunner
    from dexmani_real.runtime.observation import ObservationRow

    now = [1_000_000_000]
    targets, events, stamps, calls = [], [], [], []

    def write(frame):
        targets.append(np.r_[frame["arm_qpos"][0], frame["hand_qpos"][0]])
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
                stop_request=0,
                start_request=False,
                physical_home_completed=False,
            ).items()
        },
    )
    runtime = SimpleNamespace(
        arm=SimpleNamespace(joint_limit_lower=np.full(7, -3), joint_limit_upper=np.full(7, 3)),
        hand=SimpleNamespace(qpos_min_rad=np.full(12, -3), qpos_max_rad=np.full(12, 3)),
    )
    info = SimpleNamespace(
        n_obs_steps=2,
        n_action_steps=8,
        control_dt_s=0.0625,
        action_mode="joint",
        observation_fields=("joint_state",),
    )

    def predict(obs):
        calls.append(obs["joint_state"].copy())
        now[0] += inference_ns
        full = np.repeat(np.arange(16)[:, None], 19, axis=1) * 0.001 + len(calls) * 0.01
        return {"pred_action": full, "control_action": full[1:9]}

    def record(group, **data):
        events.append((group, data))
        if group == "queries":
            now[0] += diagnostic_ns

    runner = PolicyRunner(
        shared,
        runtime,
        info,
        model_runtime=SimpleNamespace(
            predict=lambda obs: predict(obs)["control_action"], predict_with_diagnostics=predict
        ),
        fingertip_runtime=None,
        execute=True,
        max_running_s=30,
        diagnostics=SimpleNamespace(record=record) if traced else None,
    )
    runner.run_id, runner.started_ns = 4, now[0]

    def observe():
        arm, hand = np.zeros(1, dtype=ARM_STATE_DTYPE), np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm["timestamp_ns"] = hand["timestamp_ns"] = now[0]
        return ObservationRow(arm, hand, None, None, None, now[0])

    runner._read_observation = observe
    with patch("dexmani_real.deployment.runner.time.monotonic_ns", side_effect=lambda: now[0]):
        yield SimpleNamespace(
            runner=runner,
            shared=shared,
            now=now,
            targets=targets,
            events=events,
            stamps=stamps,
            calls=calls,
        )


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

    def test_sequence_matching_does_not_cross_runs(self):
        pubs = {
            "run_id": np.array([2, 3]),
            "sequence": np.array([1, 1]),
            "accepted": np.array([True, True]),
        }
        commands = {
            "run_id": np.array([3]),
            "sequence": np.array([1]),
            "sdk_start_ns": np.array([100]),
        }
        self.assertEqual(match_commands(pubs, commands), ([(1, 0)], [(2, 1)], []))

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

    def test_runner_keeps_targets_and_recovers_boundary_delay(self):
        results = []
        for traced in (False, True):
            with runner_case(traced=traced) as case:
                for _ in range(16):
                    case.now[0] = max(case.now[0], case.runner.next_step_ns)
                    case.runner.step()
                self.assertEqual(len(case.calls), 2)
                self.assertEqual(len(case.targets), 16)
                self.assertEqual(case.runner.stats.inference_ms, [20.0, 20.0])
                np.testing.assert_array_equal(
                    case.runner.stats.action_step_intervals_ms, np.diff(case.stamps) / 1e6
                )
                results.append(case)
        plain, traced = results
        np.testing.assert_array_equal(plain.targets, traced.targets)
        self.assertEqual(plain.stamps, traced.stamps)
        expected = np.concatenate(
            [
                np.repeat(np.arange(1, 9)[:, None], 19, axis=1) * 0.001 + query * 0.01
                for query in (1, 2)
            ]
        )
        np.testing.assert_array_equal(traced.targets, expected)
        stamps = traced.stamps
        self.assertEqual(stamps[8] - stamps[7], 82_500_000)
        self.assertEqual(stamps[9] - stamps[8], 42_500_000)
        self.assertEqual(stamps[9] - stamps[7], 125_000_000)
        for i in (*range(2, 8), *range(10, 16)):
            self.assertEqual(stamps[i] - stamps[i - 1], 62_500_000)
        queries = [d for g, d in traced.events if g == "queries"]
        pubs = [d for g, d in traced.events if g == "publications"]
        self.assertEqual([d["pred_index"] for d in pubs], list(range(1, 9)) * 2)
        self.assertEqual(queries[1]["infer_start_ns"] - queries[0]["infer_start_ns"], 500_000_000)
        self.assertEqual(queries[1]["observation_ns"], [1_437_500_000, 1_500_000_000])
        for query, pub in zip(queries, (pubs[0], pubs[8])):
            self.assertEqual(query["infer_end_ns"] - query["infer_start_ns"], 20_000_000)
            self.assertEqual(pub["execution_observation_ns"], query["infer_end_ns"])
            self.assertEqual(query["observation_ns"][-1], query["infer_start_ns"])

    def test_inference_accounting_excludes_diagnostics_and_recorder(self):
        targets = []
        for traced in (False, True):
            for recorder_delay in (0, 7_000_000):
                with self.subTest(traced=traced, recorder_delay=recorder_delay):
                    with runner_case(traced=traced, diagnostic_ns=7_000_000) as case:
                        checks = []

                        def check_error():
                            checks.append(case.now[0])
                            # Only the check immediately after inference has an injected cost.
                            if len(checks) == 2:
                                case.now[0] += recorder_delay

                        case.runner.recorder = SimpleNamespace(
                            check_error=check_error, add_frame=lambda *a, **kw: None
                        )
                        with patch("dexmani_real.deployment.runner.build_episode_frame"):
                            case.runner.step()
                        self.assertEqual(case.runner.stats.inference_ms, [20.0])
                        self.assertEqual(case.runner.next_step_ns, 1_062_500_000)
                        self.assertEqual(
                            case.stamps, [1_020_000_000 + traced * 7_000_000 + recorder_delay]
                        )
                        if traced:
                            query = case.events[0][1]
                            self.assertEqual(
                                query["infer_end_ns"] - query["infer_start_ns"], 20_000_000
                            )
                        targets.append(case.targets)
        for target in targets[1:]:
            np.testing.assert_array_equal(target, targets[0])

    def test_overrun_consumes_one_action_and_reanchors_next_tick(self):
        for inference_ns in (80_000_000, 200_000_000):
            with (
                self.subTest(inference_ns=inference_ns),
                runner_case(inference_ns=inference_ns) as case,
            ):
                start = case.now[0]
                case.runner.step()
                self.assertEqual(len(case.targets), 1)
                self.assertEqual(len(case.runner.action_queue), 7)
                self.assertEqual(case.runner.next_step_ns, start + 62_500_000)
                self.assertEqual(case.now[0], start + inference_ns)
                case.runner.step()
                self.assertEqual(len(case.targets), 2)
                self.assertEqual(case.runner.next_step_ns, start + inference_ns + 62_500_000)
                case.runner.step()
                self.assertEqual(len(case.targets), 2)
                for _ in range(6):
                    case.now[0] = case.runner.next_step_ns
                    case.runner.step()
                pubs = [d for g, d in case.events if g == "publications"]
                self.assertEqual([d["pred_index"] for d in pubs], list(range(1, 9)))
                self.assertEqual(len(case.calls), 1)
                np.testing.assert_array_equal(
                    case.targets,
                    np.repeat(np.arange(1, 9)[:, None], 19, axis=1) * 0.001 + 0.01,
                )

    def test_tick_budget_includes_work_and_ik_failure(self):
        from dexmani_real.planning.kinematics.ik import IKFailureKind, IKResult
        from dexmani_real.robot.action import ActionRealization

        for phase in ("observation", "realization", "publication", "recording", "ik_failure"):
            with self.subTest(phase=phase), runner_case() as case:
                runner = case.runner
                if phase == "observation":
                    original = runner._read_observation
                elif phase == "publication":
                    original = case.shared.robot_command_ring.write
                else:
                    original = runner.realizer.realize

                def work(*args, **kwargs):
                    case.now[0] += 7_000_000
                    if phase == "ik_failure":
                        return ActionRealization(
                            None,
                            np.zeros(12),
                            ik_result=IKResult(
                                success=False,
                                qpos=None,
                                failure_kind=IKFailureKind.NO_SOLUTION_FOUND,
                            ),
                        )
                    return original(*args, **kwargs)

                if phase == "observation":
                    runner._read_observation = work
                elif phase == "publication":
                    case.shared.robot_command_ring.write = work
                elif phase == "recording":

                    def add_frame(*args, **kwargs):
                        case.now[0] += 7_000_000

                    runner.recorder = SimpleNamespace(check_error=lambda: None, add_frame=add_frame)
                else:
                    runner.realizer.realize = work
                with patch("dexmani_real.deployment.runner.build_episode_frame"):
                    runner.step()
                    self.assertEqual(runner.next_step_ns, 1_062_500_000)
                    publications = len(case.targets)
                    remaining = len(runner.action_queue)
                    history = tuple(runner.history.rows)
                    case.now[0] = runner.next_step_ns - 1
                    runner.step()
                    self.assertEqual(len(case.targets), publications)
                    self.assertEqual(len(runner.action_queue), remaining)
                    self.assertEqual(tuple(runner.history.rows), history)
                    case.now[0] += 1
                    runner.step()
                self.assertEqual(runner.next_step_ns, 1_125_000_000)
                self.assertEqual(len(case.calls), 2 if phase == "ik_failure" else 1)
                if phase == "ik_failure":
                    self.assertEqual(case.targets, [])
                    self.assertFalse(runner.action_queue)
                    self.assertEqual(runner.stats.ik_failure_counts, {"no_solution_found": 2})

    def test_tick_start_reuses_gate_clock_sample(self):
        with runner_case() as case:
            samples = []

            def clock():
                samples.append(case.now[0])
                case.now[0] += 1_000_000
                return samples[-1]

            with patch("dexmani_real.deployment.runner.time.monotonic_ns", side_effect=clock):
                case.runner.step()
            self.assertEqual(case.runner.next_step_ns, samples[0] + 62_500_000)

    def test_revocation_during_inference_never_publishes(self):
        for reason in ("stop", "estop", "fault", "epoch"):
            with self.subTest(reason=reason), runner_case() as case:
                predict = case.runner.model.predict_with_diagnostics

                def revoke(obs):
                    result = predict(obs)
                    if reason == "stop":
                        request_policy_stop(case.shared)
                    elif reason == "estop":
                        case.shared.estop_request.value = True
                    elif reason == "fault":
                        case.shared.error_state.value = True
                    else:
                        case.shared.run_id.value += 1
                    return result

                case.runner.model.predict_with_diagnostics = revoke
                case.runner.step()
                self.assertEqual(len(case.calls), 1)
                self.assertFalse(case.runner.action_queue)
                self.assertEqual(case.targets, [])
                case.runner.step()
                self.assertIsNone(case.runner.run_id)
                self.assertEqual(case.targets, [])
                self.assertFalse(command_may_cross_sdk(case.shared, run_id=4))

    def test_stop_queued_actions_and_publication_rejection(self):
        for phase in ("queued", "realization"):
            with self.subTest(phase=phase), runner_case() as case:
                if phase == "queued":
                    case.runner.step()
                    request_policy_stop(case.shared)
                else:
                    realize = case.runner.realizer.realize

                    def revoke(*args):
                        result = realize(*args)
                        request_policy_stop(case.shared)
                        return result

                    case.runner.realizer.realize = revoke
                case.runner.step()
                self.assertEqual(len(case.targets), 1 if phase == "queued" else 0)
                if phase == "realization":
                    self.assertEqual(case.runner.stats.publication_rejections, 1)
                    self.assertFalse(case.events[-1][1]["accepted"])
                    self.assertEqual(case.runner.next_step_ns, 1_062_500_000)
                    case.runner.step()
                self.assertFalse(case.runner.action_queue)
                self.assertIsNone(case.runner.run_id)
                self.assertFalse(command_may_cross_sdk(case.shared, run_id=4))

    def test_timeout_and_stale_observation_prevent_publication(self):
        for phase in ("timeout_before", "timeout_inference", "stale_before", "stale_after"):
            with self.subTest(phase=phase), runner_case() as case:
                if phase.startswith("timeout"):
                    case.runner.max_running_s = 0.01
                    if phase == "timeout_before":
                        case.now[0] += 10_000_000
                else:
                    observe = case.runner._read_observation
                    rows = [None] if phase == "stale_before" else [observe(), None]
                    case.runner._read_observation = lambda: rows.pop(0)
                case.runner.step()
                self.assertEqual(case.targets, [])
                self.assertFalse(case.runner.action_queue)
                self.assertIsNone(case.runner.run_id)
                if phase.startswith("timeout"):
                    self.assertEqual(case.shared.run_ended_reason.value, int(RunEndReason.TIMEOUT))
                self.assertFalse(command_may_cross_sdk(case.shared, run_id=4))

    def test_latest_only_transport_reproduces_short_interval_sequence_loss(self):
        from dexmani_real.ipc.ring import SharedMemoryRingBuffer

        # Real ring read/write code, but private bytes instead of OS shared memory.
        # SDK callbacks and 30 Hz polls are synthetic; no physical worker is started.
        for inference_ns in (20_000_000, 40_000_000):
            for phase_ns in (0, 10_000_000, 20_000_000):
                with self.subTest(inference_ns=inference_ns, phase_ns=phase_ns):
                    with runner_case(inference_ns=inference_ns) as case:
                        origin = case.now[0]
                        for _ in range(16):
                            case.now[0] = max(case.now[0], case.runner.next_step_ns)
                            case.runner.step()
                        self.assertEqual(case.stamps[9] - case.stamps[8], 62_500_000 - inference_ns)
                        schedule = [(stamp, "publish", i) for i, stamp in enumerate(case.stamps)]
                        schedule.extend(
                            (origin + phase_ns + i * 1_000_000_000 // 30, "poll", None)
                            for i in range(31)
                        )
                        with patch(
                            "dexmani_real.ipc.ring.shared_memory.SharedMemory",
                            side_effect=lambda **kw: SimpleNamespace(buf=bytearray(kw["size"])),
                        ):
                            ring = SharedMemoryRingBuffer("offline", ROBOT_COMMAND_DTYPE)
                        case.shared.robot_command_ring = ring
                        evidence = {owner: [] for owner in ("arm", "hand")}
                        last_sequence = {owner: 0 for owner in evidence}
                        publications = []
                        trace = SimpleNamespace(
                            record=lambda group, **data: publications.append(data)
                        )
                        for stamp, kind, index in sorted(schedule):
                            case.now[0] = stamp
                            if kind == "publish":
                                target = case.targets[index]
                                receipt = publish_command(
                                    case.shared,
                                    RobotCommand(4, target[:7], target[7:]),
                                    diagnostics=trace,
                                )
                                self.assertEqual(receipt, stamp)
                                continue
                            for owner in evidence:
                                latest = read_robot_command(case.shared)
                                if latest is None:
                                    continue
                                command, sequence = latest
                                if sequence == last_sequence[owner]:
                                    continue
                                last_sequence[owner] = sequence
                                self.assertTrue(
                                    command_may_cross_sdk(case.shared, run_id=command.run_id)
                                )
                                sdk_trace = SimpleNamespace(
                                    record=lambda group, **data: evidence[owner].append(data)
                                )
                                trace_sdk_call(
                                    sdk_trace,
                                    lambda target: 0,
                                    getattr(command, f"{owner}_qpos"),
                                    run_id=command.run_id,
                                    sequence=sequence,
                                )
                        pubs = {k: np.array([p[k] for p in publications]) for k in publications[0]}
                        for owner, sends in evidence.items():
                            commands = {k: np.array([s[k] for s in sends]) for k in sends[0]}
                            pairs, absent, unsent = match_commands(pubs, commands)
                            self.assertEqual(unsent, [])
                            expected_missing = (
                                [(4, 1), (4, 9)]
                                if inference_ns == 40_000_000 and phase_ns == 0
                                else []
                            )
                            self.assertEqual(absent, expected_missing, owner)
                            self.assertEqual(len(pairs), 16 - len(expected_missing))
                        if inference_ns == 40_000_000 and phase_ns == 0:
                            self.assertEqual(
                                case.stamps[8:10], [origin + 540_000_000, origin + 562_500_000]
                            )
                            self.assertEqual(evidence["arm"][7]["sequence"], 10)

    def test_analysis_synthetic_session(self):
        import yaml

        for fixed in (False, True):
            with self.subTest(fixed=fixed), tempfile.TemporaryDirectory() as tmp:
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
                    base = 1_000_000_000 + chunk * (500_000_000 if fixed else 520_000_000)
                    full = np.repeat(np.arange(16)[:, None], 19, axis=1) * 0.001 + chunk * 0.008
                    writers["policy"].record(
                        "queries",
                        run_id=2,
                        episode=1,
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
                        stamp = (
                            base + step * 62_500_000 + (20_000_000 if step == 0 or not fixed else 0)
                        )
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
                original_files = {p: p.read_bytes() for p in session.rglob("*") if p.is_file()}
                report, traces = analyze_session(session)
                run = report["runs"]["2"]
                self.assertEqual(run["arm"]["sent"], 16)
                self.assertEqual(run["publication_interval_ms"]["boundary"]["median"], 82.5)
                self.assertEqual(run["arm"]["publish_to_sdk_lower_ms"]["median"], 10.0)
                self.assertEqual(run["hand_minus_arm_send_ms"]["median"], 5.0)
                self.assertAlmostEqual(run["boundaries"][0]["arm"]["overlap_rmse_deg"], 0.0)

                self.assertEqual(
                    run["post_boundary_interval_ms"]["median"], 42.5 if fixed else 62.5
                )
                self.assertEqual(run["boundary_pair_span_ms"]["median"], 125.0 if fixed else 145.0)
                self.assertEqual(run["query_drift_ms"]["median"], 0.0 if fixed else 20.0)
                self.assertEqual(run["boundary_recovery_skipped"], 0)
                if fixed:
                    for scenario in (
                        "missing",
                        "step_gap",
                        "rejected",
                        "incomplete_old",
                        "run",
                        "episode",
                        "timeout_before",
                        "timeout_after",
                    ):
                        with self.subTest(scenario=scenario):
                            edited = deepcopy(traces)
                            policy = edited["policy"]
                            pubs, queries = policy["publications"], policy["queries"]
                            if scenario == "missing":
                                policy["publications"] = {k: v[:9] for k, v in pubs.items()}
                            elif scenario == "step_gap":
                                policy["publications"] = {
                                    k: np.delete(v, 9, axis=0) for k, v in pubs.items()
                                }
                            elif scenario == "rejected":
                                pubs["accepted"][9] = False
                            elif scenario == "incomplete_old":
                                pubs["accepted"][3] = False
                            elif scenario == "run":
                                pubs["run_id"][8:] = 3
                                queries["run_id"][1] = 3
                                for owner in ("arm", "hand"):
                                    edited[owner]["commands"]["run_id"][8:] = 3
                            elif scenario == "episode":
                                queries["episode"][1] = 2
                            else:
                                # End either before the new chunk, or between its steps 0 and 1.
                                stamp = pubs["publish_end_ns"][8] + (
                                    1 if scenario == "timeout_after" else -1
                                )
                                policy["episode_ends"] = {
                                    "run_id": np.array([2]),
                                    "timestamp_ns": np.array([stamp]),
                                    "reason": np.array([b"timeout"]),
                                }
                            with patch(
                                "dexmani_real.deployment.diagnostic_analysis.load_trace",
                                side_effect=lambda path: (
                                    edited[path.stem],
                                    True,
                                    {"complete": True},
                                ),
                            ):
                                changed, _ = analyze_session(session)
                            for metrics in changed["runs"].values():
                                self.assertEqual(metrics["boundary_pair_span_ms"], {"count": 0})
                                self.assertEqual(metrics["post_boundary_interval_ms"], {"count": 0})
                            skipped = scenario in (
                                "missing",
                                "step_gap",
                                "rejected",
                                "timeout_after",
                            )
                            self.assertEqual(
                                changed["runs"]["2"]["boundary_recovery_skipped"], int(skipped)
                            )
                self.assertEqual(
                    original_files, {p: p.read_bytes() for p in session.rglob("*") if p.is_file()}
                )


if __name__ == "__main__":
    unittest.main()
