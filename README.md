# DexMani Real

面向个人 PhD 研究的真实机器人灵巧操作仓库，当前实验系统为 **xArm7 + XHand + RealSense RGB-D + VR / HTS**。

本仓负责硬件与传感器接入、标定、遥操作采集、Raw 数据、离线处理及策略真机部署。模型、Dataset 和训练位于 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真位于 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

工作流：**标定 → 采集 → Raw → Canonical Zarr → 策略训练 → 真机评估**。

## 安装与配置

使用 Python >= 3.10，在源码 checkout 中安装；机器人资产按 checkout 路径读取。

```bash
python -m pip install -e .
python -m pip install -e ".[interactive]"
```

沿用实验机现有的硬件 SDK、几何与推理环境。离线几何处理需要 Pinocchio，retargeting 需要 NLopt 和所选后端；点云诊断视图需要 Open3D，交互数据查看需要 Rerun。键盘监听使用 pynput，需要桌面显示环境。

默认参数来自包内配置，`--config` 接受 YAML 字段覆盖，CLI 操作开关优先。可先打印声明配置，再编辑现场参数：

```bash
python examples/collect_teleop.py --print-config > local.yaml
python examples/collect_teleop.py --config local.yaml --print-config
python examples/run_policy.py --config local.yaml --print-config
```

`--help`、`--print-config` 不连接设备；打印配置也不读取现场标定或加载策略。实际运行入口检查所需配置、标定与执行条件。日志在运行入口初始化，可通过 `DEXMANI_LOG_DIR` 指定目录，默认写入 `~/.dexmani/logs/`。

## 实验工作流

### 标定

```bash
python examples/calibrate_camera.py --help
python examples/calibrate_vr_heading.py --help
python examples/pointcloud_process_example.py --help
```

相机外参、桌面平面、手部安装与 VR alignment 属于当前实验现场状态；物理设置变化后重新标定或确认。首次桌面标定在 pointcloud 示例中选择 table calibration。指尖 FK 使用当前手部安装补偿，碰撞与规划使用标称资产几何。

采集允许缺相机外参；点云重建需要对应相机的内外参与 depth scale。标定在会话启动时读取。采集可指定 `--vr-transform` / `--camera-calibration`，部署可指定 `--camera-calibration`，相机和 VR 标定可指定 `--output`；桌面平面路径由 `environment.table.plane_path` 配置。

### 采集

```bash
python examples/collect_teleop.py <task> --config local.yaml
```

任务名默认 `test`，用于录制元数据和 `policy.episodes_dir/<task>/` 输出目录。启动时先建立停止监听，再执行手部 HOME。

| 按键 | 操作 |
|---|---|
| H | 整机 HOME |
| T | 空闲、无 capture 且确认手部无接触时，采集并核验触觉基线 |
| B | 开始采集 |
| C | 暂停或恢复；恢复开始新 episode |
| S | 停止并保存 |
| D | 丢弃当前 capture |
| Q | 退出确认，再次 Q 保存退出；空闲且无 capture 时直接退出 |
| ESC | 急停 |

S/Q/ESC 可取消触觉归零；SDK 调用期间的取消需等待调用返回。触觉缺测保留 NaN，不阻断关节任务。键盘 jog 入口仅在 R 时执行 HOME。

### 查看与导出

```bash
python examples/visualize_episode.py <episode> --info
python examples/visualize_episode.py <episode> --max-frames 100 --config local.yaml
python examples/export_policy_zarr.py episodes/<task> --config local.yaml
```

`--info` 查看元数据与数值概要，无需 Rerun；交互 viewer 顺序读取所选 RGB-D 帧。查看和导出使用当前点云、桌面与手部安装配置，相机几何来自对应 Raw。仅当 `pointcloud.remove_table: true` 时读取桌面平面。

公共导出保留全部 **13 字段**、NaN、dispatch 与 episode 边界，下游按模型所需字段筛选。处理逻辑变化后，从 Raw 导出到新路径。

### 策略部署

```bash
python examples/run_policy.py <policy/task/experiment> --config local.yaml
```

H 执行 HOME，T 在空闲且无接触时归零触觉，B 开始，S 停止，Q 退出，ESC 急停。开始前检查 arm/hand HOME 姿态。默认每段运行预算 60 秒，可用 `--max-duration` 修改。

策略提供保存的模态、节拍和点云数值配方，Real 配置提供当前设备与现场状态。执行模式通过 `execution` 配置，支持 sync、async 和 RTC；async/RTC 需指定预取步数，RTC 还需指定 guidance cap。模型加载与 warmup 在设备连接前完成，执行预算需按当前策略和现场条件核对。

checkpoint 默认 `best`；策略覆盖参数见 `--help`。录制会话的 `run_config.yaml` 保存实际配置、checkpoint、覆盖值与源码版本。`--output` 指定部署输出位置。

`--no-record` 仍连接设备并执行动作；程序接口的 `execute=False` 仍连接并读取设备。两者均不能用于离线验证。

## 其他入口

| 用途 | 入口 |
|---|---|
| 键盘 jog / HOME | `examples/keyboard_teleop.py` |
| 真实轨迹 replay | `examples/replay_episode.py` |
| XHand 动作与反馈示例 | `examples/xhand_control_example.py` |
| RGB-D 诊断 | `examples/realsense_record_example.py` |
| 点云诊断与桌面标定 | `examples/pointcloud_process_example.py` |

XHand 示例按 HOME → fist → palm → V → OK → HOME 顺序执行动作，实测到位后各停留 1 秒；`--read-only` 仍连接设备并读取反馈。相机诊断使用 `--config`；RealSense 诊断退出时恢复可读取的原曝光优先级。具体操作与参数见各入口 `--help`。

## 数据与安全

**Raw 是实验 evidence，发布后不可变。** 采集保留接入的 RGB-D、关节与触觉观测。停止、晚帧、控制失败和缺测保留已采前缀及结束原因，仅操作者显式丢弃才删除当前 capture。写盘、编码或发布失败保留 staging 并报告失败；相机停帧不补黑图，触觉按 Raw 保存值导出。

Canonical 是可由 Raw 重建的训练缓存。源数据缺必要字段时明确报错；历史实验使用对应源码版本复现。相机数据保留对应几何和 depth scale，训练筛选不回写 Raw。

真机运行前确认机械限位、安装、标定与工作空间，保持操作者在场并可急停。设备 SDK 或 CUDA 阻塞会影响停止响应；离线检查不能证明现场安全。代码代理未经明确授权不得连接设备、执行运动、实时采集或标定写入。

## 开发

```bash
python -m compileall -q dexmani_real examples
git diff --check
```

按改动补充纯离线逻辑核查，验证时沿用现有依赖环境。长期协作规则见 [AGENTS.md](AGENTS.md)，Claude 入口见 [CLAUDE.md](CLAUDE.md)。当前行为以源码、配置和 CLI 为准。
