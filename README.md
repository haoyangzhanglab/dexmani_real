# DexMani Real

DexMani Real 是 DexMani 灵巧操作（Dexterous Manipulation）研究中的真实机器人实验仓库。系统基于 **xArm7 + XHand + RealSense RGB-D + VR/HTS**，连接机器人控制、多模态感知与学习策略，支持遥操作示教、数据采集、策略部署和真机评估。

## 项目定位

DexMani 的研究代码分为三个相互关联的仓库：

| 仓库 | 主要职责 |
| --- | --- |
| **dexmani_real** | 真实设备接入、实验运行、数据采集与真机评估 |
| [**dexmani_policy**](https://github.com/haoyangzhanglab/dexmani_policy) | 策略模型、训练与推理 |
| [**dexmani_sim**](https://github.com/haoyangzhanglab/dexmani_sim) | 仿真环境与仿真实验 |

Real 聚焦真实世界中的感知与动作执行，并对接 Policy 的训练结果；仿真相关实验由 Sim 承载。

## 系统架构

系统以遥操作、策略评估和轨迹回放为实验入口，共享几何、观测表示、控制计算和运行机制。

| 组成 | 模块 | 职责 |
| --- | --- | --- |
| 设备接入 | [`robot/`](dexmani_real/robot/) · [`sensor/`](dexmani_real/sensor/) | 机械臂、灵巧手、RGB-D 相机及 VR 的连接与采集；机器人协调入口为 `robot/system.py` |
| 几何与感知 | [`geometry/`](dexmani_real/geometry/) · [`perception/`](dexmani_real/perception/) | 位姿、坐标变换和 FK；在线/离线共享点云、指尖与手表面表示 |
| 动作控制 | [`control/`](dexmani_real/control/) | 动作映射、在线 IK、目标约束、关节等价角与 HOME 路径；在线接触和 HOME 间隙规则保持分离 |
| 共享运行 | [`runtime/`](dexmani_real/runtime/) · [`runtime/motion/`](dexmani_real/runtime/motion/) · [`ipc/`](dexmani_real/ipc/) | 运行身份、输入、观测、传感进程与退出；HOME/jog 协调及进程间通信 |
| 实验执行 | [`teleop/`](dexmani_real/teleop/) · [`deployment/`](dexmani_real/deployment/) · [`replay/`](dexmani_real/replay/) | 遥操作、学习策略执行与轨迹回放 |
| 数据管理 | [`data/`](dexmani_real/data/) | Raw 记录、Canonical 处理与导出；`session_results.py` 管理会话结果 |
| 配置与标定 | [`config.py`](dexmani_real/config.py) · [`calibration/`](dexmani_real/calibration/) | 实验配置装配、标定算法、交互会话及默认参数；配置定义仍位于各领域 |
| 通用工具 | [`utils/`](dexmani_real/utils/) | 原子写入、计时、日志与可视化 |

遥操作和策略部署分别处理来自操作者与学习模型的动作，但共享机器人状态、运动规划和底层执行机制。数据记录和结果分析围绕实验过程展开，不与特定策略模型绑定。

## 实验流程

- **遥操作示教**：通过 VR/HTS 获取操作者运动，完成机械臂与灵巧手的动作映射及示教采集。
- **策略部署与评估**：将真实设备观测送入训练得到的策略，执行预测动作并记录真机实验结果。
- **离线分析与回放**：查看、处理实验数据，并利用已记录的动作轨迹开展回放和分析。

标定与设备诊断为这些流程提供共同的实验支持。

相机标定算法、外参读写与交互会话位于 [`calibration/camera/`](dexmani_real/calibration/camera/)，桌面拟合与读写位于 [`calibration/table_plane.py`](dexmani_real/calibration/table_plane.py)，VR 对齐与启动质量检查位于 [`calibration/vr_alignment.py`](dexmani_real/calibration/vr_alignment.py)。默认参数集中在 [`calibration/parameters/`](dexmani_real/calibration/parameters/)，由轻量的 `calibration/paths.py` 定位，消费者需要时才读取。

实验配置使用 `from dexmani_real.config import load_experiment_config`，覆盖优先级为 CLI > YAML > dataclass 默认值。配置加载不读取标定或连接设备，运行约束在使用边界验证。VR 标定脚本的 `--force` 仅允许将低质量结果导出到显式指定、非默认且未占用的 `--output` 文件；遥操作启动仍拒绝低质量标定。

## 代码导航

核心实现位于 [`dexmani_real/`](dexmani_real/)，实验入口与设备工具集中在 [`scripts/`](scripts/)；[`assets/`](assets/) 存放机器人模型等资源，[`third_party/`](third_party/) 包含第三方相关代码。

部署保留 `deployment/run.py → session.py → rollout/episode.py`：入口装配配置，会话持有设备和模型，单轮协调观测、提交与录制。`rollout/action_schedule.py` 只根据传入的时钟与状态决定 sync/async/RTC 调度，`inference.py` 独占模型线程。回放的帧索引一致性指标位于 [`replay/consistency.py`](dexmani_real/replay/consistency.py)。

`control/joint_angles.py` 统一关节等价角、绝对跳变阈值和关节目标拒绝规则，由 `target_validation.py` 与 `online_ik.py` 直接使用；纯关节检查不依赖 IK 求解器。

项目依赖与打包配置见 [`pyproject.toml`](pyproject.toml)，开发约定见 [`AGENTS.md`](AGENTS.md)。

## 迁移说明

默认标定由 `dexmani_real/state/*.json` 迁至 `dexmani_real/calibration/parameters/*.json`，保留实际内容。外部 YAML 若显式指定旧位置，需要操作者显式修改路径；不会自动回退到另一份参数。`environment.table.plane_path` 的相对路径仍以仓库根目录解析，绝对路径及显式相机/VR 路径保持原有含义。

旧 Python 模块路径已移除，应按上面的领域导航更新外部导入。`SafetyParams` 改名为 `RuntimeTimeouts`，`EpisodeResults` 改名为 `SessionResults`；YAML 的 `safety` 键、默认数值、`session_result.json` 字段与实验命令保持不变。离线部署契约测试从 `scripts/check_deployment.py` 迁至 `tests/test_deployment_contract.py`。

## 离线验证

在已安装项目及测试依赖的 Python 环境中运行：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q dexmani_real scripts tests
python scripts/run_policy.py --print-config
python scripts/collect_teleop.py --print-config
git diff --check
```

已有 Ruff/mypy 的开发环境可补充静态检查，无需为运行实验安装它们：

```bash
ruff check --isolated --select E9,F dexmani_real scripts tests third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py
mypy --ignore-missing-imports --follow-imports silent --explicit-package-bases \
  dexmani_real/geometry dexmani_real/perception dexmani_real/control \
  dexmani_real/calibration dexmani_real/runtime dexmani_real/deployment \
  dexmani_real/data/session_results.py
```

mypy 按现有类型标注检查上述模块，不检查未标注函数体或外部无类型声明的 SDK；不能替代数值与运行边界测试。

数值测试使用真实 Pinocchio、MPlib、trimesh、OpenCV 和 fpsample；部署契约测试还需要可导入的 `dexmani_policy` 与其配置依赖。缺失依赖必须单独记录，不能将导入失败或全部跳过视为通过。键盘遥操作入口保留 `--help`，没有 `--print-config` 参数。

测试覆盖共享表示、控制候选、碰撞诊断、调度边界、来源时钟、运行状态、模型线程及清理行为。点云 worker 使用合成 RGB-D 和真实共享内存验证发布时钟及 sequence；设备、录制与模型替身仅提供软件边界证据。真机启停、取消/急停、HOME/jog、遥操作录制、三种策略模式和回放仍需操作者单独验收。软件授权撤销和 SDK 接受请求不代表物理停稳或到位。

默认标定 JSON 随 wheel 打包；机器人 `assets/` 仍沿用 checkout 定位方式，验证 wheel 中的 JSON 不等于验证了完整独立安装的机器人运行环境。
