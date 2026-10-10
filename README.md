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
| H | 保存并结束当前段，再执行整机 HOME；完成后等待新 B |
| T | 空闲、无 capture 且确认手部无接触时，采集触觉基线 |
| B | 开始采集 |
| C | 暂停或恢复；恢复开始新 episode |
| S | 停止并保存，回到空闲；新 B 可开始 |
| D | 丢弃当前未发布 capture，回到空闲 |
| Q | 退出确认，再次 Q 保存退出；空闲且无 capture 时直接退出 |
| ESC | 急停 |

S/Q/ESC 可取消触觉归零；SDK 调用期间的取消需等待调用返回。提交前取消保留当前会话原有基线；正常完成按通道更新，失败通道失效，技术异常使两通道失效。aggregate 检查残差幅度，dense 只检查载荷与残差有限，均不能证明无接触，dense 也不证明稳定性。触觉缺测保留 NaN，不阻断非触觉关节任务；含缺测的 episode 在导出时整段拒绝。键盘 jog 入口仅在 R 时执行 HOME。

遥操作从首个实际采样建立固定时间网格；晚采样、漏拍或动作跨过下一拍会结束本轮并保存真实前缀，不追赶补发。同批停止/退出按键压过开始和恢复，S/D 同批优先保存。初始采样预算为 `dt/4`（16 Hz 时 15.625 ms），需要现场测量确认；动作计算与下发可用本拍剩余时间，这些检查不是硬实时或真机安全保证。

### 查看与导出

```bash
python examples/visualize_episode.py <episode> --info
python examples/visualize_episode.py <episode> --max-frames 100 --config local.yaml
python examples/export_policy_zarr.py episodes/<task> --config local.yaml
```

`--info` 查看元数据与数值概要，无需 Rerun；交互 viewer 顺序读取所选 RGB-D 帧。查看和导出使用当前点云、桌面与手部安装配置，相机几何来自对应 Raw。仅当 `pointcloud.remove_table: true` 时读取桌面平面。

公共导出要求每段全部 **13 字段**完整，每行浮点数值均有限；缺字段、NaN/Inf、RGB-D 行数不符、相机几何缺失或点云无法重建时，整段拒绝并在日志中记录原因。完整性检查覆盖全部模态，不随模型所需字段而放宽。Raw 原样保留，导出不删单帧、不插值、不填补缺测。下游按模型所需字段加载。

训练准入同时检查逐拍间隔与累计时间偏离，预算与采集共享 `dt/4`；单行会明确提示无间隔可检验。每行 arm/hand dispatch 都必须为 ACCEPTED，训练 action 表示双设备 SDK 接受的绝对目标，不代表物理到位；CRC 未确认或其他状态整段拒绝。任务失败本身不排除技术有效数据。

默认输出为 `datasets/<task>.zarr`，目标必须不存在，可用 `--output` 指定新路径。Zarr 使用 Diffusion Policy 的 `data/*` 和 `meta/episode_ends` 结构，根属性仅保存数值处理、数据身份与 episode 名称。处理逻辑变化后从 Raw 重新导出。文件读写故障或处理逻辑异常会中止导出，已创建的 staging 保留供排查后重建。

### 策略部署

```bash
python examples/run_policy.py <policy/task/experiment> --config local.yaml
```

H 执行 HOME，T 在空闲且无接触时归零触觉，B 开始，S 停止，Q 退出，ESC 急停。开始及 reset 后重新采样，复用运行时策略输入检查并核对 arm/hand HOME 姿态；启动容差默认机械臂 0.5°、手部 5°，机械臂 HOME 运动的收敛容差仍为 0.15°。准备失败保留具体原因，不进入 RUNNING 或消耗运行段数。启用 Raw 时仍要求可匹配且新鲜的 RGB-D 来源，单独报告录制来源失败。默认每段运行预算 60 秒，可用 `--max-duration` 修改。

策略负责保存的 horizon、观测历史、模态、normalizer、控制时间步和点云生产 recipe；encoder 内部采样点数可以不同于生产点数。Real 负责当前设备、现场状态和执行参数。入口一次性解析 `A_exec`：优先 `--n-action-steps`，否则读取策略保存的默认值，结果记为 `execution.action_steps`（由入口生成，不能在 YAML 指定），不覆盖模型配置。模型仍输出扣除观测历史偏移后的完整 `P = horizon - n_obs_steps + 1` 步未来动作；sync 要求 `A_exec <= P`，async/RTC 要求 `1 <= d <= A_exec` 且 `A_exec + d <= P`。

`execution.execution_mode` 支持 sync、async、rtc，预取模式必须指定 `prefetch_steps=d`。sync 与普通 async 在执行时根据当前反馈实现动作；async 在交接前 `d` 步查询，不提前冻结物理命令。RTC 保留前 `d` 步命令预实现与冻结，以对应物理 prefix 做 guidance，执行冻结命令前重新检查；`rtc_guidance_cap=0` 保留冻结、关闭 guidance，可与正值比较。

稳态 slot `k` 查询的模型完成期限固定为 `t0 + (k+d)*dt`：结果必须已经可取得，且完整 CPU 预测的 `completed_ns` 不晚于这个时间；owner 的 `max_tick_lateness_s` 只约束观测处理、目标检查及命令提交，不能延长推理窗口。bootstrap／sync 等待完成后选择首次执行 anchor，保留动作头部；稳态从 query anchor 的索引 `d` 接续。模型加载与 warmup 在设备连接前完成；warmup 仅检查模型耗时的必要条件，输入准备、RTC prefix 构造和余量也占用同一个查询窗口，不能据此保证运行成功。

健康的 bootstrap／sync 换段允许预算内 WAIT，不重复下发旧目标，也不保证上一已接受目标已经停止。async/RTC 预取期间继续执行当前有效计划。必要观测失效、计划过期、错过时隙/交接或 WAIT 超时会撤权、尝试停止并结束本轮；观测恢复后不自动续跑，新 B 仍须满足 HOME、新鲜度和模型空闲检查。论文评估应记录执行模式、预算与终止规则。

checkpoint 默认 `latest`；可用 `--checkpoint 80pct` 选择已保存的 80% milestone，也支持 `best` 或 checkpoint 文件名。百分比必须对应已有 checkpoint；`best` 需要实验中的 `best_ckpt.json`。模型推理参数与 Real 执行参数见 `--help`。保存产物时，`run_config.yaml` 记录实际配置、checkpoint、覆盖值、输出开关与源码版本。`--output` 指定部署输出位置。

Raw 与评估摘要可独立关闭：`--no-record` 关闭 Raw，`--no-results` 关闭摘要，两者同时使用时不创建会话产物。评估摘要保存在 `session_result.json`，包含结束原因、时长、故障和推理、发送、跳过节拍的统计。启用摘要时，在连接设备前写入初始未完成状态；运行期间只更新内存，撤销运动后保存每段结果，资源清理后更新会话结果。session 统一提交幂等模型关闭，并在其他资源释放后用剩余关闭预算等待同一个 Future；`model_close.status` 区分 completed、failed、pending，超预算表示关闭未完成，不等同关闭异常。清理失败或最终仍 pending 都返回非零退出状态，episode 原始结束原因独立保留。有限等待不保证 Python 线程或 CUDA 卡住时进程也能按时退出。

Rollout Raw 默认用于评估 evidence，保留 bootstrap/WAIT 的无动作行与具体终止原因（关闭摘要时也保留）。公共导出按同一时间、完整性与 ACCEPTED 规则判定，通常会拒绝这些含无动作行的 episode；未来用于再训练需另行定义派生片段和时间映射，不能填动作或裁行来通过当前导出。

`data.h5/timing` 保存可选的控制时序与来源证据，关联观测、query、候选动作、命令提交和终止原因，也记录相机取帧、发布及点云处理、发布时刻，便于分解延迟。相机 SDK 的 global/system 时间映射到本机 monotonic；只有 hardware clock 时保留主机接收时间，并记录原始 SDK 时间与时钟域。点云继承 RGB-D 较旧通道的来源时间，计算完成不会刷新其年龄；这些时间不保证代表曝光中点或机器人到位。具体布局与缺失标识见 [Raw schema](dexmani_real/recording/storage/schema.py)，读取与延迟诊断见 [EpisodeReader](dexmani_real/recording/storage/reader.py)，实际 SDK 调用以 dispatch status 为准。历史 Raw 保持原样，未记录的时序证据明确标为不可用。

部署 joint、EEF 的最终关节目标及 RTC 冻结目标统一检查机械/关节限位、跳变、FK 最终 EEF 工作空间、自碰撞及当前配置的 static boxes。最终 FK 工作空间检查默认允许各边界超出 5 mm，可通过 `execution.eef_workspace_margin_m` 调整，设为 0 恢复严格边界；EEF 原始目标投影与 HOME 路径仍使用原始工作空间。Policy 动作不做桌面碰撞检测，允许贴桌/接触操作，由操作员监督并准备硬件急停；H 执行 HOME 时仍按当前配置检查桌面和路径，点云去桌仍使用当前桌面标定。EEF 仍先投影再 IK；拒绝就终止本轮并记录原因，不重规划或恢复。这些是最终目标检查，不证明真实连续运动轨迹安全；检查耗时计入 owner 预算，仍需现场验收。

`--no-record`、`--no-results` 只控制产物保存，部署仍连接设备并执行动作；程序接口的 `execute=False` 仍连接并读取设备，均不能用于离线验证。

### 轨迹回放

```bash
python examples/replay_episode.py episodes/<task>/<episode> --config local.yaml --output replay_results/<run>
```

按 Raw 的标称节拍逐行回放关节目标，要求 teleop 数据、非空有限的目标与关节状态、双设备持续发送状态，并检查目标是否满足当前限位。输出目录须新建或为空，保存 `session_result.json`、已采集的 `replay_data.npz` 与可计算的一致性指标 `metrics.json`。Q 保存前缀并退出，ESC 急停；完成后可按 H 执行规划回 HOME。

## 其他入口

| 用途 | 入口 |
|---|---|
| 键盘 jog / HOME | `examples/keyboard_teleop.py` |
| XHand 动作与反馈示例 | `examples/xhand_control_example.py` |
| RGB-D 诊断 | `examples/realsense_record_example.py` |
| 点云诊断与桌面标定 | `examples/pointcloud_process_example.py` |

XHand 示例按 HOME → fist → palm → V → OK → HOME 顺序执行动作，实测到位后各停留 1 秒；`--read-only` 仍连接设备并读取反馈。相机诊断使用 `--config`；RealSense 诊断退出时恢复可读取的原曝光优先级。具体操作与参数见各入口 `--help`。

## 数据与安全

**Raw 是实验 evidence，发布后不可变。** 采集保留接入的 RGB-D、关节与触觉观测。停止、晚帧、控制失败和缺测保留已采前缀及结束原因，仅操作者显式丢弃才删除当前 capture。写盘、编码或发布失败保留 staging 并报告失败；相机停帧不补黑图，触觉按 Raw 保存值导出。

每个 Raw episode 只使用 `data.h5` 与 `rgb.mp4`，HDF5 保存物理观测、尝试发送的目标、相对秒时间与发送状态，以及采集和相机元数据。

录制端非阻塞提交当前控制行，后台线程完成编码和写盘；启动和结束时先撤销运动，再等待文件准备或发布。录制队列满或写入失败会明确报错并保留 staging，不静默丢帧。

Canonical 是可由 Raw 重建的训练缓存。训练侧只加载模型所需 observation/action，并检查这些字段的数值；训练筛选不回写 Raw。

真机运行前确认机械限位、安装、标定与工作空间，保持操作者在场并可急停。设备 SDK 或 CUDA 阻塞会影响停止响应；离线检查不能证明现场安全。代码代理未经明确授权不得连接设备、执行运动、实时采集或标定写入。

## 开发

```bash
python -m compileall -q dexmani_real examples
git diff --check
```

按改动补充纯离线逻辑核查。长期协作规则见 [AGENTS.md](AGENTS.md)，Claude 入口见 [CLAUDE.md](CLAUDE.md)。当前行为以源码、配置和 CLI 为准。
