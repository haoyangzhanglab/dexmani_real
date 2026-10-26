"""小型合成 Raw fixture；使用真实 writer、HDF5 和视频，不连接设备。"""

import numpy as np

from dexmani_real.data.recorder import AsyncEpisodeRecorder, snapshot_recording_metadata
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.feedback import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.runtime.observation import ObservationRow


def raw_row(index=0):
    # 超过 float64 的连续整数范围，确保来源纳秒没有经过浮点转换。
    stamp = 9_007_199_254_741_111 + index * 100_000_000
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    arm['timestamp_ns'] = stamp - 30
    for name in ('timestamp_ns', 'tactile_aggregate_timestamp_ns', 'tactile_dense_timestamp_ns'):
        hand[name] = stamp - 20
    camera = dict(
        rgb=np.full((16, 16, 3), index * 32, dtype=np.uint8),
        depth=np.full((16, 16), 1000 + index, dtype=np.uint16),
        timestamp_ns=stamp - 100, ring_sequence=17 + index,
    )
    return ObservationRow(arm, hand, camera, None, None, stamp, control_timestamp_ns=stamp - 10)


def start_raw_recorder(directory, *, collection_source='policy_rollout'):
    camera = CameraSnapshot(
        'offline', CameraIntrinsics(16, 16, 20, 20, 8, 8, 'none', (0,) * 5), .001, np.eye(4),
    )
    recorder = AsyncEpisodeRecorder(directory, control_hz=10)
    recorder.start_episode(
        task_label='offline', collection_source=collection_source,
        metadata=snapshot_recording_metadata(
            camera, tactile_reset_metadata={}, tactile_hold_max_age_s=.2,
        ),
    )
    return recorder


def write_raw_episode(directory):
    recorder = start_raw_recorder(directory)
    try:
        for index in range(2):
            recorder.add_frame(raw_row(index), RobotCommand(1, np.zeros(7), np.zeros(12)))
        return recorder.save_episode(reason='operator', technical_status='normal')
    finally:
        recorder.close()
