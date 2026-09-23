# DexMani Real

面向个人 PhD 灵巧操作实验的真实机器人代码：采集真机 demonstration、部署 learned policy、做真机 evaluation。

当前硬件是 xArm7、XHand、RealSense RGB-D 和 VR / HTS。模型训练与 policy implementation 位于相邻的 dexmani_policy；本仓库负责真实硬件、数据采集、部署适配、raw episode 和离线数据转换。

本仓库会控制真实硬件。home、teleoperation、replay、policy rollout 和 camera calibration 都不是离线测试。运行前检查机械限位、实验环境、标定与急停，并保持操作者在场。软件检查通过不代表真机安全验证通过。

## 安装与配置

要求 Python >= 3.10。

    python -m pip install -e .

在运行 examples 前，先在所用 Python 环境中完成上述 editable install。examples 直接导入已安装的 `dexmani_real` 包，不自行修改 `sys.path`。下文命令从仓库根目录执行。

xArm、XHand、RealSense、HTS SDK，以及运动学/点云和 dexmani_policy 依赖，按实验机环境安装。

普通 import 和普通配置解析不应连接设备。真实 SDK 连接由对应 owning worker 显式建立：

- xArm SDK：arm worker；
- XHand SDK：hand worker；
- RealSense SDK：camera worker；
- model / CUDA：policy worker。

支持 --config 的入口按 CLI > 显式 YAML > config/defaults.py 解析。

可在连接设备前查看采集配置：

    python examples/collect_teleop.py --print-config
    python examples/collect_teleop.py --config experiment.yaml --print-config

VR teleop 的控制频率由 `teleop.control_hz` 拥有，键盘 / 标定 jog 由 `keyboard_teleop.control_hz` 拥有；learned policy 的动作周期由 `PolicySpec.control_dt_s` 拥有，replay 使用录制轨迹的频率。
这些动作生产频率不得高于相关执行器的 worker 服务频率（arm-only jog 只受 arm 限制）。`arm.loop_hz` / `hand.loop_hz` 是命令接收和反馈更新频率，不是硬件物理伺服频率。
正常 arm / hand / camera 观测的新鲜度上限为各自生产周期的 4 倍（30 Hz 时约 133 ms）；点云沿用源相机采集时间。Homing 的 arm 新鲜度、VR 新鲜度和硬件读取失败超时各自独立。
录制 teleop 的首个 IK / retarget 失败行会保留为诊断数据，并立即停止该无效 episode；录制收尾异步完成。

## 研究入口

| 工作流 | 命令 | 输出 / 说明 |
|---|---|---|
| VR collection | python examples/collect_teleop.py --config experiment.yaml --task-name <task> --operator <name> | 真机；raw episode |
| Keyboard jog / home | python examples/keyboard_teleop.py --config experiment.yaml | 真机 |
| Physical replay | python examples/replay_episode.py <episode> --config experiment.yaml | 真机 |
| Policy rollout | python examples/run_policy.py <policy/task/experiment> --config experiment.yaml | 真机；evaluation session |
| Camera calibration | python examples/calibrate_camera.py --config experiment.yaml ... | 真机；标定 |
| VR heading calibration | python examples/calibrate_vr_heading.py | VR / HTS |
| Canonical Zarr export | python examples/export_policy_zarr.py episodes/<task> | 离线 |
| XHand diagnostics | python examples/xhand_diagnostics.py | 连接真机；诊断，不应运动 |
| Inspect raw | python examples/visualize_episode.py <episode> --info | 离线 |

实际参数以各入口 --help 为准。

## Teleoperation / recording

标准采集：

    python examples/collect_teleop.py --config experiment.yaml --task-name <task> --operator <name>

显式无录制调试：

    python examples/collect_teleop.py --config experiment.yaml --task-name <task> --operator <name> --no-record

交互键保持：

- B：开始；
- C：暂停 / 恢复；
- S：停止并保存；
- D：丢弃；
- H：return-home；
- Q：退出；
- ESC：急停。

C 不等于 stop 或 discard。pause 会撤销当前 motion epoch；resume 从 fresh robot / VR feedback 重新锚定。发生过 C 暂停的 raw episode 可以保留用于诊断；无论暂停时长、是否恢复，都不属于 clean training episode，canonical Zarr export 会整条拒绝并报告原因。

录制模式下 camera 与 RecorderIO 是 required resources。只有显式 --no-record 才允许不录制的 teleop。camera / recorder 中途失败时，不应静默降级为无录制 teleop，也不应把残缺数据伪装成完整 demonstration。

## Control semantics

正常 teleop / policy evaluation 使用简单 streaming target 语义：

    controller / policy
        -> target interpretation
        -> absolute joint target preparation
        -> latest RobotCommand
        -> arm worker / hand worker
        -> vendor SDK

run_id 是 motion lifecycle epoch，用于阻止 pause、stop、timeout 或慢 inference 返回后的旧动作重新获得执行权限。

记录的 action 是该 control step 发布到 robot execution boundary 的 high-level absolute executable target，不代表 SDK 消费或物理到达。

### xArm

normal streaming target：

- finite；
- nearest 2pi equivalent；
- absolute operational joint limit；
- worker physical/rated hard-limit fence；
- xArm SDK velocity / acceleration control。

normal streaming 不额外做 high-level arm delta clip。

### XHand

normal streaming target：

- finite；
- absolute operational joint limit；
- worker mechanical hard-limit fence；
- 直接发送 absolute XHand SDK position target。

XHand 接收绝对目标，运动响应由设备控制器决定。

### Collision / workspace

Online Cartesian IK 使用最终准备好的 hand target，拒绝 robot self-colliding target configurations。arm-only Cartesian jogging 使用 fresh measured hand geometry；--no-hand 使用固定 home 几何，要求手已拆除或固定在配置的 home 姿态。

normal teleop / policy eval / replay 不做 current-to-target transition/path/environment collision checking。完整 path/environment collision planning 由 planned return_home 负责。实验仍依赖操作者现场监督、硬件急停、机械限位和 xArm firmware motion controls。

Cartesian teleop / EE policy 可以在 IK 前做简单 EEF workspace clip；joint-action policy 不为了 normal runtime safety 额外做 FK workspace gate。

## Observation

arm、hand、camera、VR、point cloud 都是异步 producer。

每个 control step 只组装一次 current observation snapshot：

- 读取当前已经 available 的 latest required sample；
- 每个 modality 单独做 freshness check；
- snapshot 组装完成后记录 observation timestamp；
- teleop controller 与 raw recorder 复用同一份 snapshot。

各模态按自身时间戳检查新鲜度。

已发布的 point cloud 是自包含的 derived observation；pointcloud-only policy 不要求源 RGB-D 帧仍保留在 camera ring。source_camera_sequence 保留 provenance，只在模型同时消费 RGB + pointcloud 时用于精确源帧匹配。pointcloud-only rollout 的 recorder 独立使用 latest fresh camera telemetry，不要求与点云同源。

policy temporal history由 control-row deque 维护，不在 inference 时重新从 sensor rings 回溯构造历史。warm-up 使用 edge padding；pause / 大 gap 后清空并重新开始 history。

whole sample acquisition failure时 producer不发布一份带 generic invalid flag 的新 sample；旧 sample自然因 timestamp 变 stale。

唯一显式保留的部分模态 validity 是 XHand tactile aggregate / dense validity，因为 joint state 可以有效而某一种 tactile measurement 无效。

## Raw episode

Raw episode 是实验 source of truth。

当前 schema 为 raw v32，保存研究有意义的真实信息：

- arm qpos / qvel / effort；
- hand qpos / current；
- aggregate tactile + validity；
- dense tactile + validity；
- RGB-D 与 calibration；
- VR wrist / hand landmarks；
- observation/action/source timestamps；
- high-level arm / hand absolute action target；
- 必要的 controller intent，例如 pre-IK EE intent；
- 小型 frame-status 诊断。

Raw 保存实验事实；动作不承诺设备已消费或已到达。

Point cloud不作为 demonstration raw source重复保存；由 RGB-D + calibration 离线生成。

## Canonical full Zarr

数据路径：

    raw episode
        -> whole-episode admission
        -> offline point cloud / FK / fingertips
        -> one canonical full Zarr

Zarr 是可重建的 training cache，不是第二份 runtime audit database。

Zarr固定全量保存 learning-relevant dynamic modalities，使 dexmani_policy 再通过 sensor_modalities 选择实际模型输入。

canonical Zarr v14 的 dynamic arrays 包括：

- joint_state；
- arm_qvel；
- arm_effort；
- hand_current；
- contact_force；
- tactile_force；
- eef_pose；
- fingertip_points；
- rgb；
- depth；
- point_cloud；
- action；
- action_ee。

静态 camera intrinsic / extrinsic / depth scale 等只保存一次，不逐帧广播。

Zarr 不保存 runtime time arrays，也不保存 validity masks。

action 与 action_ee 必须描述同一个最终 high-level target：

- action = arm joint target + hand joint target；
- action_ee = FK(arm joint target) + hand joint target。

pre-IK EE intent 是 raw provenance，不是 Zarr action_ee。

## Raw -> Zarr admission

Exporter 不修复 demonstration。

一个 raw episode 要么完整进入 Zarr，要么整条拒绝并说明原因。

禁止：

- 丢单帧后继续；
- 将一条 raw episode 切成多个训练 episode；
- 删除 pause 区间后拼接；
- silent resample；
- zero-fill invalid tactile；
- silent camera repair；
- 将 incomplete recording 当完整 trajectory。

整条拒绝的典型原因包括：

- incomplete recording；
- 非 teleop training workflow；
- 任意非 OK frame status；
- 任意 tactile aggregate / dense invalid；
- media / row count mismatch；
- shape / dtype / finite 错误；
- 明显 pause / missing-control timing gap；
- 任意 offline pointcloud / FK / fingertip transform failure。

允许正常 OS scheduling jitter；不要求每个 control interval 精确等于 nominal dt。相邻 observation 或 action 的时间间隔超过 2 个标称控制周期时，整条 episode 拒收；发生过 C 暂停则不论间隔长短都拒收。

批量 export 可以接受正常 episodes并整条跳过异常 episodes，但每一条 rejection 必须有明确 reason 和最终 summary，不能静默处理。

## Return-home

return_home 是唯一刻意保守的 motion path。

它可以继续使用：

- equivalent-angle handling；
- planned waypoints；
- workspace；
- self collision；
- arm-hand collision；
- table/static obstacle clearance；
- Mode 0；
- abort；
- measured final convergence；
- restore Mode 6。

xArm HOME 成功以 measured qpos / qvel convergence 为准。XHand HOME 在 owning worker 的 SDK 发送成功后完成，不等待实测收敛；随后 arm-home 在现有 feedback 等待期限内取得一份 fresh post-command measured hand qpos 用于路径规划，无需达到 home target。缺少该反馈会阻止 arm-home 规划。无手反馈模式使用固定 home 几何；commissioning 必须验证实际手部运动与随后的机械臂路径。

## Policy rollout

Policy process拥有 model / CUDA，保持同步 inference 与 local action chunk，不增加 async inference、RTC、remote inference 或 catch-up burst。

如果 predict 阻塞期间 run_id 改变，返回的旧 chunk直接丢弃。

输出属于 evaluation telemetry。默认不能因为格式类似就当作 teleop BC source进入 canonical training Zarr。

## 代码风格与离线检查

Python 排版和 imports 使用 `pyproject.toml` 中的 Ruff 配置：100 列、Python 3.10 语法目标。代码保持直接、易读，不为减少 LOC 压缩独立操作；注释保留数学、坐标系、单位、SDK 行为、实验 rationale 和实际安全原因，避免重复代码或叙述实现历史。

环境中已有 Ruff 时，先排序 imports，再格式化：

    ruff check --select I --fix dexmani_real examples
    ruff format dexmani_real examples

离线验收：

    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples
    ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

Ruff 命令不可用时，可尝试已有的 `python -m ruff`；若仍不可用，报告未完成的 Ruff 检查，不为检查安装或升级实验环境依赖。

仓库不维护 committed tests 目录。纯逻辑改动使用一次性 offline smoke checks，不运行真实硬件入口。

## 真机 commissioning

离线重构完成后，真机验证需单独授权。

重点实测：

- xArm去除software delta clip后的流式平滑性；
- XHand direct absolute target的行为、电流和jerk；
- C pause/resume stale-action fence 与 re-anchor；
- arm / hand / camera / VR freshness；
- Cartesian IK 使用最终 hand target 的端点自碰撞几何与在线求解耗时；
- return_home安全路径；
- XHand HOME 接受后的实测手姿态、后续手部运动与机械臂 return_home 路径；
- slow inference无旧动作复活、无catch-up burst；
- recorder / camera failure；
- tactile partial validity；
- raw v32；
- full canonical Zarr export与whole-episode rejection；
- representative dexmani_policy rollout；
- emergency stop与shutdown。

### Deployment preprocessing and offline checks

Policy artifact owns observation modalities, exact ordered robot joints, control period,
RGB preprocessing and point-cloud algorithm parameters. Real passes raw uint8 HWC RGB;
Policy handles resizing, cropping and normalization. Point-cloud deployment uses the
artifact configuration, even when `runtime.pointcloud` differs.

`pointcloud.remove_table` controls perception independently of collision-table enablement.
When enabled, it uses the current resolved Real table plane. Camera calibration, intrinsics,
depth scale, serial and hand mounting also come from current Real setup. Recalibration does
not invalidate a trained policy. Rollout provenance records the effective point-cloud config
and plane. New Policy Zarr schema v15 stores ordered `joint_names` and `pointcloud_config_json`.

With both repositories installed, run offline regressions without connecting hardware:

```bash
conda run --no-capture-output -n real_robot python -m dexmani_real.deployment.smoke_test
conda run --no-capture-output -n policy python -m dexmani_policy.deployment.smoke_test
```
