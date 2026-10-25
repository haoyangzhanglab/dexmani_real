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

系统以遥操作、策略评估和轨迹回放为实验入口，复用底层的设备接口、运行时与运动规划能力。

| 组成 | 模块 | 职责 |
| --- | --- | --- |
| 设备与感知 | [`robot/`](dexmani_real/robot/) · [`sensor/`](dexmani_real/sensor/) | 机械臂、灵巧手、RGB-D 相机及 VR 设备接入 |
| 运行与规划 | [`runtime/`](dexmani_real/runtime/) · [`ipc/`](dexmani_real/ipc/) · [`planning/`](dexmani_real/planning/) | 运行状态、进程间通信、运动学与动作规划 |
| 实验执行 | [`teleop/`](dexmani_real/teleop/) · [`deployment/`](dexmani_real/deployment/) · [`replay/`](dexmani_real/replay/) | 遥操作、学习策略执行与轨迹回放 |
| 数据管理 | [`data/`](dexmani_real/data/) | 实验记录、数据读取及离线处理 |
| 基础支持 | [`calibration/`](dexmani_real/calibration/) · [`config/`](dexmani_real/config/) · [`utils/`](dexmani_real/utils/) | 标定、实验配置与通用工具 |

遥操作和策略部署分别处理来自操作者与学习模型的动作，但共享机器人状态、运动规划和底层执行机制。数据记录和结果分析围绕实验过程展开，不与特定策略模型绑定。

## 实验流程

- **遥操作示教**：通过 VR/HTS 获取操作者运动，完成机械臂与灵巧手的动作映射及示教采集。
- **策略部署与评估**：将真实设备观测送入训练得到的策略，执行预测动作并记录真机实验结果。
- **离线分析与回放**：查看、处理实验数据，并利用已记录的动作轨迹开展回放和分析。

标定与设备诊断为这些流程提供共同的实验支持。

## 代码导航

核心实现位于 [`dexmani_real/`](dexmani_real/)，实验入口与设备工具集中在 [`scripts/`](scripts/)；[`assets/`](assets/) 存放机器人模型等资源，[`third_party/`](third_party/) 包含第三方相关代码。

项目依赖与打包配置见 [`pyproject.toml`](pyproject.toml)，开发约定见 [`AGENTS.md`](AGENTS.md)。
