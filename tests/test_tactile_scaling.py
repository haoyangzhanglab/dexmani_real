import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import zarr

from dexmani_real.data.export import ExportConfig, create_store, export_zarr
from dexmani_real.data.raw import (
    TACTILE_FINGER_ORDER,
    TIMING_ROW_DTYPE,
    RawReader,
    RawWriter,
    snapshot_recording_metadata,
)
from dexmani_real.data.recorder import EpisodeRecorder
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.model import XHAND_TACTILE_SCALE_DIVISOR
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import XHandFeedback, XHandState
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_device_feedback,
    check_observation,
)


def sdk_state(timestamp_ns=1_000_000_000):
    return XHandState(
        qpos=np.arange(12, dtype=np.float64),
        tactile_aggregate=np.full((5, 3), 255.0),
        tactile_dense=np.full((5, 120, 3), 127.5),
        timestamp_ns=timestamp_ns,
    )


def observation(hand):
    stamp = int(hand["timestamp_ns"][0])
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    arm["timestamp_ns"] = stamp
    return Observation(
        arm=arm, hand=hand,
        camera={"rgb": np.zeros((16, 16, 3), dtype=np.uint8),
                "depth": np.ones((16, 16), dtype=np.uint16), "timestamp_ns": stamp,
                "ring_sequence": 1},
        vr={"wrist_pos": np.zeros(3), "wrist_quat_wxyz": np.array([1., 0., 0., 0.]),
            "landmarks": np.zeros((21, 3)), "recv_ts_ns": stamp, "sequence_id": 1},
        point_cloud=None, control_timestamp_ns=stamp + 1000,
        observation_timestamp_ns=stamp + 2000,
    )


class FeedbackScalingTest(unittest.TestCase):
    def test_scale_shape_dtype_and_readonly_without_mutating_sdk(self):
        feedback = XHandFeedback()
        self.assertIsNone(feedback.consume(None))
        state = sdk_state()
        result = feedback.consume(state)
        assert result is not None
        for channel, shape, value in (("aggregate", (5, 3), 1.0), ("dense", (5, 120, 3), 0.5)):
            actual = result[f"tactile_{channel}"][0]
            self.assertEqual(actual.shape, shape)
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, np.full(shape, value))
            np.testing.assert_array_equal(getattr(state, f"tactile_{channel}"), value * 255)
            self.assertFalse(np.shares_memory(actual, getattr(state, f"tactile_{channel}")))
            self.assertEqual(result[f"tactile_{channel}_timestamp_ns"][0], state.timestamp_ns)
        np.testing.assert_array_equal(result["qpos"][0], state.qpos)
        self.assertEqual(result["timestamp_ns"][0], state.timestamp_ns)
        self.assertFalse(result.flags.writeable)
        with self.assertRaises(ValueError):
            result["tactile_aggregate"] = 2
        self.assertIs(feedback.consume(None), result)

    def test_channels_hold_independently_and_clear_rescales_only_new_frames(self):
        for held, updated in (("aggregate", "dense"), ("dense", "aggregate")):
            with self.subTest(held=held):
                feedback = XHandFeedback()
                previous = feedback.consume(sdk_state())
                state = sdk_state(2_000_000_000)
                setattr(state, f"tactile_{held}", None)
                getattr(state, f"tactile_{updated}")[:] = -510
                state.qpos[:] = 0.25
                frame = feedback.consume(state)
                assert frame is not None and previous is not None
                np.testing.assert_array_equal(frame[f"tactile_{held}"], previous[f"tactile_{held}"])
                np.testing.assert_array_equal(frame[f"tactile_{updated}"], -2)
                self.assertEqual(frame[f"tactile_{held}_timestamp_ns"][0], 1_000_000_000)
                self.assertEqual(frame[f"tactile_{updated}_timestamp_ns"][0], state.timestamp_ns)
                np.testing.assert_array_equal(frame["qpos"][0], state.qpos)
                self.assertIs(feedback.consume(None), frame)
                feedback.clear_tactile()
                cleared = feedback.consume(None)
                assert cleared is not None
                self.assertFalse(cleared.flags.writeable)
                np.testing.assert_array_equal(cleared["qpos"], frame["qpos"])
                np.testing.assert_array_equal(cleared["timestamp_ns"], frame["timestamp_ns"])
                for name in ("aggregate", "dense"):
                    self.assertTrue(np.isnan(cleared[f"tactile_{name}"]).all())
                    self.assertEqual(cleared[f"tactile_{name}_timestamp_ns"][0], 0)
                # 清零生成新缓存，不污染消费者已持有的快照。
                np.testing.assert_array_equal(frame[f"tactile_{updated}"], -2)
                fresh = feedback.consume(sdk_state(3_000_000_000))
                assert fresh is not None
                np.testing.assert_array_equal(fresh["tactile_aggregate"], 1)
                np.testing.assert_array_equal(fresh["tactile_dense"], 0.5)
                self.assertIs(feedback.consume(None), fresh)

    def test_freshness_uses_each_original_channel_timestamp(self):
        feedback = XHandFeedback()
        feedback.consume(sdk_state())
        state = sdk_state(1_100_000_000)
        state.tactile_dense = None
        row = observation(feedback.consume(state))
        runtime = SimpleNamespace(hand=SimpleNamespace(enabled=True, feedback_max_age_s=0.05),
                                  arm=SimpleNamespace(feedback_max_age_s=0.05))
        needs = ObservationNeeds(required_tactile=("aggregate", "dense"))
        now = row.observation_timestamp_ns
        self.assertEqual(check_device_feedback(row, runtime, now_ns=now), [])
        self.assertEqual(check_observation(row, runtime, needs, now_ns=now),
                         ["hand_tactile_dense_timestamp_ns_unavailable_or_stale"])
        self.assertEqual(check_observation(row, runtime, needs, now_ns=now, allow_tactile_hold=True), [])
        feedback.clear_tactile()
        row = observation(feedback.snapshot())
        self.assertEqual(check_observation(row, runtime, needs, now_ns=now, allow_tactile_hold=True),
                         ["hand_tactile_aggregate_unavailable", "hand_tactile_dense_unavailable"])


class RawTactileTest(unittest.TestCase):
    def test_recording_online_and_export_share_scaled_values(self):
        camera = CameraSnapshot(
            None, CameraIntrinsics(16, 16, 10., 10., 8., 8., "none", (0.,) * 5), 0.001, None,
        )
        metadata = snapshot_recording_metadata(camera, tactile_reset_metadata={})
        self.assertEqual(metadata["hand_tactile_scale_divisor"], 255.0)
        self.assertEqual(metadata["hand_tactile_finger_order"], TACTILE_FINGER_ORDER)
        fields = ("finger_tactile_force", "taxel_tactile_force")
        builder = ObservationBuilder(SimpleNamespace(observation_fields=fields), None)
        feedback = XHandFeedback()
        state = sdk_state()
        # 各指、各 XYZ 通道及 taxel 不同，覆盖负值及超过 255 的值，禁止裁剪或换序。
        state.tactile_aggregate = np.arange(15).reshape(5, 3) * 63.75 - 510
        state.tactile_dense = np.arange(1800).reshape(5, 120, 3) * 0.75 - 510
        expected = {field: np.asarray(values, dtype=np.float32) / np.float32(XHAND_TACTILE_SCALE_DIVISOR)
                    for field, values in zip(fields, (state.tactile_aggregate, state.tactile_dense))}
        rows = [observation(feedback.consume(state))]
        # 两个通道均无效时仅关节时间更新；保持值和原触觉时间戳。
        rows.append(observation(feedback.consume(XHandState(state.qpos, None, None, 1_050_000_000))))
        online = builder.build([builder.project(row) for row in rows])
        for field in fields:
            np.testing.assert_array_equal(online[field], np.stack([expected[field]] * 2))
        with tempfile.TemporaryDirectory() as directory:
            for source in ("teleop", "policy_rollout"):
                with self.subTest(source=source):
                    recorder = EpisodeRecorder(directory, control_hz=20)
                    try:
                        recorder.start(task_label="scaling_test", collection_source=source,
                                       metadata=metadata, episode_name=f"episode_{source}")
                        for row in rows:
                            self.assertIsNone(recorder.append(row))
                        path = recorder.finish(save=True, reason="test_complete")
                    finally:
                        if recorder.writer_pending:
                            recorder.finish(save=False, reason="test_cleanup")
                    self.assertEqual(recorder.result()["status"], "published")
                    with RawReader(path) as raw:
                        raw.validate_tactile_metadata()
                        self.assertEqual(raw.meta["hand_tactile_scale_divisor"], 255.0)
                        self.assertEqual(raw.meta["collection_source"], source)
                        for field, key in zip(fields, ("hand_contact", "hand_tactile_force")):
                            np.testing.assert_array_equal(raw[key][:], online[field])
                            self.assertEqual(raw[key].dtype, np.float32)
                        for channel in ("aggregate", "dense"):
                            np.testing.assert_array_equal(raw.read_timing()[f"tactile_{channel}_ns"], 1_000_000_000)
                    destination = Path(directory) / f"{source}.zarr"
                    export_zarr(path, destination, processing=ExportConfig(), fields=fields)
                    canonical = zarr.open_group(str(destination), mode="r")
                    self.assertEqual(canonical.attrs["tactile_scale_divisor"], 255.0)
                    for field in fields:
                        np.testing.assert_array_equal(canonical[f"data/{field}"][:], online[field])
                        self.assertEqual(canonical[f"data/{field}"].dtype, np.float32)

    def test_writer_rejects_missing_or_wrong_scale_before_opening_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            path.write_bytes(b"existing file")
            for value in (None, 1.0, 254.999, "255.0", float("nan"), float("inf")):
                metadata = {} if value is None else {"hand_tactile_scale_divisor": value}
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "hand_tactile_scale_divisor"):
                    RawWriter(directory, control_hz=20).open(metadata)
                self.assertEqual(path.read_bytes(), b"existing file")

    def test_reader_requires_both_scale_and_finger_order(self):
        with tempfile.TemporaryDirectory() as directory:
            cases: list[tuple[float | str | None, str | None, str | None]] = [
                (255.0, TACTILE_FINGER_ORDER, None),
            ]
            cases += [(value, TACTILE_FINGER_ORDER, "hand_tactile_scale_divisor")
                      for value in (None, 1.0, "255.0", float("nan"), float("inf"))]
            cases += [(255.0, order, "hand_tactile_finger_order") for order in (None, "pinky,ring,middle,index,thumb")]
            for scale, order, error in cases:
                with self.subTest(scale=scale, order=order):
                    with h5py.File(Path(directory) / "data.h5", "w") as data:
                        meta = data.create_group("meta")
                        if scale is not None:
                            meta.attrs["hand_tactile_scale_divisor"] = scale
                        if order is not None:
                            meta.attrs["hand_tactile_finger_order"] = order
                        data.create_dataset("timing/rows", data=np.zeros(1, dtype=TIMING_ROW_DTYPE))
                        data.create_dataset("hand_contact", data=np.ones((1, 5, 3), dtype=np.float32))
                        data.create_dataset("hand_tactile_force", data=np.ones((1, 5, 120, 3), dtype=np.float32))
                    with RawReader(directory) as reader:
                        if error is None:
                            reader.validate_tactile_metadata()
                        else:
                            with self.assertRaisesRegex(ValueError, error):
                                reader.validate_tactile_metadata()
                            for field in ("hand_contact", "hand_tactile_force"):
                                with self.assertRaisesRegex(ValueError, error):
                                    reader[field]

    def test_canonical_scale_metadata_only_for_tactile_fields(self):
        for fields in (("joint_state",), ("finger_tactile_force",), ("taxel_tactile_force",),
                       ("finger_tactile_force", "taxel_tactile_force")):
            with self.subTest(fields=fields):
                root, _ = create_store(zarr.MemoryStore(), ExportConfig(), {}, fields=fields, control_hz=20)
                if fields == ("joint_state",):
                    self.assertNotIn("tactile_scale_divisor", root.attrs)
                else:
                    self.assertEqual(root.attrs["tactile_scale_divisor"], 255.0)


if __name__ == "__main__":
    unittest.main()
