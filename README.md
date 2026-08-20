# DexMani Real

面向 xArm7（7 DoF）、XHand（12 DoF）、Quest VR 和 RealSense L515 的遥操作、数据采集、回放与策略部署系统。

## 从哪里开始

| 你要做什么 | 先读/先改 | 相关入口 |
|---|---|---|
| 理解项目约束、修改代码 | [`AGENTS.md`](AGENTS.md) → [`CLAUDE.md`](CLAUDE.md) | — |
| 采集遥操作 episode | `teleop/session.py`、`teleop/loop.py`、`teleop/episode_samples.py` | `examples/collect_teleop.py` |
| 物理重现 episode | `robot/episode_replay.py`、`recording/episode_reader.py` | `examples/replay_episode.py` |
| 读取 v17 数据 | `recording/episode_schema.py`、`recording/episode_reader.py` | [`docs/dataset/hdf5_episode.md`](docs/dataset/hdf5_episode.md) |
| 清洗并生成训练视图/Zarr | `data_processing/` | [`docs/dataset/processed_hdf5.md`](docs/dataset/processed_hdf5.md) |
| 修改手部 retarget | `teleop/hand_control.py`、`teleop/hand_retarget.py` | [`docs/hand_retargeting.md`](docs/hand_retargeting.md) |
| 部署 learned policy | `deployment/coordinator.py`、`deployment/worker.py` | `examples/run_policy.py` |
| 对照 Sim 数据 | — | [`docs/dataset/sim_hdf5_zarr.md`](docs/dataset/sim_hdf5_zarr.md)、[`docs/dataset/real_to_sim_mapping.md`](docs/dataset/real_to_sim_mapping.md) |

完整文档索引见 [`docs/README.md`](docs/README.md)。代码是运行行为的最终事实来源；文档只记录跨模块边界、读取合同和稳定的使用方式。

正式数据目录与仓库根目录同级组织：

```text
episodes/<task_name>/episode_*                 # raw v18；reader 兼容 v17
episodes_processed/<task_name>/episode_*.h5   # 每个 raw 对应一个压紧 HDF5 + processing_index.json
dataset/<task_name>.zarr                       # dexmani_policy 训练容器
```

不使用额外的 `inputs/` staging 层；旧平铺 episode 只通过显式迁移/annotation 兼容。
任务失败由操作者删除整条 raw episode，不由处理代码推断。Policy Zarr 只包含
`data/*` 与 `meta/episode_ends`；不写 `task_success` 或 raw episode provenance。
清洗默认审计停滞/抖动，只自动删除硬无效行；压紧产生危险动作跳变时拒绝整条轨迹。

## 系统边界

```text
camera / VR / arm / hand
          │
          ▼
  SharedStorage（固定 dtype、ring、queue、状态标志）
          │
          ├── teleop ──► SafetyGate ──► arm queue / hand ring ──► device workers
          │       └────► fixed-grid samples ──► EpisodeRecorder ──► v17 episode
          │
          └── policy worker ──► policy_plan_ring ──► deployment coordinator
                                                  └──► 同一 SafetyGate 边界
```

必须保持的边界：

- 跨进程数据只经 `SharedStorage`；固定 shape/dtype 由 `utils/schema.py` 定义。
- xArm、XHand、RealSense SDK 只在各自 worker/driver 内初始化和使用。
- teleop/deployment 决定动作和固定控制网格；Recorder 只序列化已选样本。
- arm queue 有序且有界；hand command ring 和 policy plan ring 是 latest-wins。
- `run_generation` 使暂停、回零或反馈故障前的命令失效；worker 在 SDK 边界前再次检查。
- 固件是最后安全保护；应用层负责数据有效性、生命周期和协调停止。

## 常用入口

下面的命令只表示入口和参数形状。除标注“离线”者外，运行前都需要确认硬件、工作空间和操作授权。

| 用途 | 命令 | 性质 |
|---|---|---|
| 遥操作采集 | `python examples/collect_teleop.py --task-name <task>` | 硬件；写入 `episodes/<task>/episode_*` |
| 键盘控制 | `python examples/keyboard_teleop.py` | 硬件 |
| learned policy | `python examples/run_policy.py --help` | 离线（仅帮助） |
| 回放参数 | `python examples/replay_episode.py --help` | 离线（仅帮助） |
| 物理回放 | `python examples/replay_episode.py <episode>` | 硬件；启动前执行严格 episode/provenance/几何预检 |
| 数据清洗 | `python examples/process_episodes.py --input-root episodes/<task> --profile rgb_pc --dry-run` | 离线；先审计、不写输出 |
| 导出 Policy Zarr | `python examples/export_policy_zarr.py --help` | 离线 |
| 手部调参 | `python examples/tune_hand_retarget.py --help` | 离线 |
| episode 可视化 | `python examples/visualize_episode.py <episode>` | 离线/GUI |
| 相机标定 | `python examples/calibrate_camera.py` | 硬件/写标定文件 |
| VR 朝向标定 | `python examples/calibrate_vr_heading.py` | 设备/写标定文件 |

## 环境与验证

目标环境是 Python 3.10 conda 环境 `real_robot`，从仓库根目录运行并设置 `PYTHONPATH=.`。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate real_robot
export PYTHONPATH=.

# 不初始化硬件的最低成本检查
python -m compileall -q dexmani_real examples
git diff --check
```

仓库没有常规单元测试套件；`examples/test_*.py`（若存在）也按硬件程序处理，不要当作自动化测试运行。修改纯函数、schema、reader 或生命周期分支时，优先补充不连接硬件的 deterministic check。

## 代码地图

| 目录 | 责任 |
|---|---|
| `config/` | 默认值、运行时配置、相机标定读取 |
| `shm/` | 共享内存、ring、queue 和 seqlock 读取 |
| `sensor/` | RealSense、VR 和点云输入 |
| `robot/` | xArm/XHand worker、SDK、反馈和回零 |
| `planning/` | FK、IK、碰撞和路径 |
| `teleop/` | VR 映射、retarget、动作与采样决策 |
| `policy/` | action candidate 与统一安全门 |
| `deployment/` | learned-policy 推理和动作协调 |
| `recording/` | v17 episode 写入、读取和相机 sidecar |
| `data_processing/` | v17 → real-domain 离线视图，以及 processed HDF5 → Policy Zarr |
| `examples/` | 薄 CLI/实验入口，不承载通用合同 |

### 修改闭环

1. 先按 [`CLAUDE.md`](CLAUDE.md) 的任务路由找到 owner。
2. 读取 producer、consumer 和对应 schema；不要只改一个调用点。
3. 保留当前工作区已有修改，做最小垂直变更。
4. 运行与风险匹配的离线检查，记录未运行的硬件验证。
5. 查看 `git diff`、`git status --short`，确认没有混入无关文件。
