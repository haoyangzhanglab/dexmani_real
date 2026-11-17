# Teleop / Policy 性能画像

`collect_teleop.py` 和 `run_policy.py` 的 `--profile-output PATH` 开启采样，默认关闭。启用时只在内存中保存标量，CLI 在会话收尾后写出独立 JSON。自动创建父目录，启动前拒绝已有目标，写入使用独占创建；异常退出也尝试保存，保存失败不会替换实验异常或退出码。

报告包含环境、会话时长、指标统计和计数器，不包含配置副本、设备地址、序列号、路径、错误文本或传感器载荷。配置解析失败、强制杀进程和断电可能没有报告；关闭预算内尚未结束的模型或 writer 只包含已收集的统计。

## 指标定义

执行耗时使用 `perf_counter_ns()`，单位 ms。控制时序和数据年龄使用 `monotonic_ns()`，两个时钟不相减。函数耗时包含子调用，模型与 writer 又可并发执行，因此不能将所有阶段直接相加。抛错调用仍记录耗时和 `<metric>_failures`。

| 指标 | 测量边界 |
| --- | --- |
| `control_period_ms` | 相邻已接纳控制槽的开始时间差，复用原调度时间；每段重新开始，Policy HOI 暂停与恢复不连接正常周期样本。等待预测的控制槽也计入。 |
| `control_target_period_ms` | Teleop 的 `control_dt_ns`，或 Policy 解析 Checkpoint 后的 `dt_ns`。 |
| `control_lateness_ms` | 本次调度时间减去跳槽前的计划时间，下限为 0；完整错过的槽计入 `skipped_slots`。 |
| `control_tick_ms` | 接纳槽后的观测、校验、计算、提交和录制，含提前返回与异常；不含外层键盘等待、反馈服务和准备。 |
| `deadline_misses` | 控制主体结束时已达到或超过原调度器的下一槽时间。 |
| `hold_period_ms` / `hold_lateness_ms` / `hold_target_period_ms` | Policy HOI_HOLD 分支的原调度推进点，与正常控制分开。 |
| `observation_read_ms` | `read_observation()`，含已取回元数据的标量采样。 |
| `raw_camera_read_ms` | Policy 在启用 Raw 且使用 HOI 或场景点云时独立读取 Raw 相机的耗时，单独计量。 |
| `observation_robot_state_ms` / `observation_joint_alignment_ms` | 读取已有 Robot 反馈缓存 / 已有 HOI 关节对齐调用。 |
| `check_device_feedback_ms` / `check_observation_ms` | 原检查函数；长计算后的再次检查仍是独立调用。 |
| `feedback_service_ms` | Teleop / Rollout 的反馈服务，包括准备或收尾期间的调用。 |
| `teleop_target_ms` | `VRController.compute_target()`，包含参考准备、腕部映射、手部重定向和 arm 映射。 |
| `hand_retarget_ms` | `HandRetargeter.retarget()`，包含几何目标、优化器和输出处理。 |
| `action_mapping_ms` | Teleop 的 `ActionMapper.map_eef()` / Policy 的 `Rollout.map_command()`；EEF 模式含 IK，joint 模式不含 IK。 |
| `robot_send_action_ms` | 命令提交接口进入到返回或抛错，包括 HOI hold 提交；不代表设备回执或机械执行完成。 |
| `observation_project_ms` | 主线程 `ObservationBuilder.project()`。 |
| `model_input_build_ms` | Worker 的 `ObservationBuilder.build()`。 |
| `model_predict_ms` | `model.predict()` 调用；reset_episode 在计时之外。不新增 GPU 同步，不代表纯 GPU kernel 耗时。 |
| `prediction_future_latency_ms` | 提交前到 Future 完成回调首次取时，包含排队、输入构建、reset、predict 和结果发布。回调是结果可用时刻的近似上界。 |
| `prediction_poll_delay_ms` | 完成回调标记到主线程取回 Future 的间隔。若 done 先于回调可见，省略该样本并累计 `prediction_completion_marker_races`。 |
| `load_*` / `warmup_*` | 模型加载和预热的 Future 延迟、取回延迟；预热输入构建与投影单独计为 `warmup_input_build_ms` / `warmup_observation_project_ms`。 |
| `episode_prepare_ms` | Teleop `start_episode()` / Policy `prepare()`。 |
| `episode_stop_ms` / `operator_home_ms` / `operator_tactile_reset_ms` | Teleop 收尾、HOME 和触觉复位。 |
| `teleop_initialize_ms` / `session_prepare_ms` / `device_start_ms` | 启动阶段。 |

Observation 耗时和 freshness 包含准备、正常控制及 HOI 暂停期间的已有读取，不能将其全部累计值解释为正式控制占比。不额外读取 Ring，也不细分 Ring 锁等待、载荷复制或传感器生产端耗时。

## 来源年龄

`sensor_age_ms.<source>` 为 Observation 完成时间减去来源时间，仅统计已请求的模态，不代表动作发送时刻的年龄。

| source | 来源时间 | stale 阈值 |
| --- | --- | --- |
| `arm` / `hand` | 对应反馈 `timestamp_ns[0]` | 对应设备的 `feedback_max_age_s` |
| `camera` / `raw_camera` | 已读取相机的 `timestamp_ns`，派生图像仍使用原相机来源时间 | `camera.max_frame_age_s` |
| `pointcloud` | `pointcloud_timestamp_ns` | `camera.max_frame_age_s` |
| `hoi` | `hoi.timestamp_ns` | `execution.hoi_max_age_s` |
| `tactile_aggregate` / `tactile_dense` | 对应触觉的独立 timestamp | `hand.feedback_max_age_s` |
| `vr` | `recv_ts_ns` | `teleop.vr_stale_threshold_s` |

阈值保存在 `sensor_threshold_ms`。每个来源累计 `_observations`、`_stale`（age ≥ 阈值）、`_near_stale`（age ≥ 80% 阈值，包含 stale）和 `_unavailable`。缺失、非正或未来时间戳不生成年龄；未来时间另计 `_future_timestamp`。完整观测缺少点云载荷也算 unavailable，仅读取元数据时不要求该载荷。无效 HOI 姿态另计 `hoi_invalid`。

`vr_new_samples` / `vr_reused_samples` 比较已读取的接收时间戳，表示观测次数，不表示网络吞吐或动作成功次数。已采样来源中未出现的事件计数按 0 解读，未采样来源不解释为 unavailable。

## Recorder

| 指标或计数器 | 定义 |
| --- | --- |
| `recorder_append_ms` | append 调用，含质量拒绝与异常。 |
| `recorder_enqueued_frames` / `recorder_append_inactive` | 成功入队次数 / 停止接纳后的 append 次数。 |
| `recorder_queue_high_water_observed` | 入队后观察到的最大 qsize；Full 时记原容量 16。消费者可能已取走数据，因此只是瞬时峰值的下界。 |
| `recorder_queue_full` | 原 Queue 抛出 Full 的次数。 |
| `recorder_writer_total_ms` | writer 从进入到 finally 完成，包含打开、等待、写入、flush、关闭与排空队列。 |
| `recorder_written_frames` | 已退出 writer 的 `frame_count` 之和，未刷出的缓冲行和在途 writer 不计入；不表示发布成功。 |
| `recorder_writers_started` / `recorder_writers_completed` | writer 进入 / 退出次数。 |
| `recorder_finish_call_ms` / `recorder_finish_elapsed_ms` | 每次 finish 调用 / 首次请求到 writer 退出并完成发布或丢弃尝试的总间隔，后者包含轮询间隔。 |
| `recording_gap_requests` / `recording_gaps` | mark_gap 次数 / 携带新段标记且 append_frame 正常返回的行数；重复请求可能合并，写盘失败仍以录制结果为准。 |
| `recording_truncations` | append 因队列满或质量问题停止接纳的次数。 |

writer 仅在间断行累计 gap，退出时读取已有帧数，不逐帧计时。队列容量、FIFO 顺序及异常传播保持原有实现。

## 统计与导出

每项最多保留最近 50,000 个标量。`count`、`mean`、`max` 覆盖全程；`sample_count`、P50/P95/P99 和 `window_mean` 覆盖保留窗口；`dropped_count` 是已丢弃的旧样本数。窗口不足 100 个时标记 `p99_under_sampled=true`。空统计使用 null，JSON 不输出 NaN/Infinity。

`session.duration_s` 包含启动、准备、待命和收尾，不是 Episode 控制预算。`exit_code` 保存 CLI 返回码；异常传播时为 null，并累计 `session_exceptions`。

控制、模型和 writer 线程通过短锁追加标量。收尾时逐指标复制窗口，释放锁后排序、写文件。关闭 profiling 时只经过转发分支或空上下文，没有额外计时、样本容器或载荷复制。实现见 [utils/profile.py](../dexmani_real/utils/profile.py)。

以下仅演示三个合成样本 `[1, 2, 3]` 的指标片段，不是真机结果：

```json
{
  "count": 3, "sample_count": 3, "dropped_count": 0,
  "p50": 2.0, "p95": 2.9, "p99": 2.98, "p99_under_sampled": true,
  "mean": 2.0, "max": 3, "window_mean": 2.0
}
```

## 手动实验

以下命令由用户执行，使用已有配置、固定 Checkpoint 和执行模式，保持控制频率不变：

```bash
python scripts/collect_teleop.py TASK --config experiment.py --max-duration 120 \
  --profile-output /tmp/teleop-profile.json

python scripts/run_policy.py CHECKPOINT --config experiment.py --num-episodes 1 --max-duration 120 \
  --record-raw --profile-output /tmp/policy-profile.json
```

HOI 模型另加 `--task-dir TASK_ASSETS`。按 B 开始，Teleop 段结束后按 Q 退出以导出。尽量采集至少 1,000 个有效周期，避免同时执行重型任务；暂停和等待预测会减少有效周期。重复实验使用新报告路径，去掉参数可作关闭对照。开发验证和 CI 命令统一见 [CPU 离线回归](offline_cpu.md#cpu-回归)；Fake 测试不能代替真机时延验证。
