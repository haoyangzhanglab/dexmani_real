# 策略部署

部署模块将 `dexmani_policy` 提供的模型推理接入真实机器人。模型描述所需的观测与动作表示，本仓库负责采集观测、调度预测动作、执行控制和记录评估数据。

## 数据流

**传感器与机器人反馈 → 观测历史 → 模型推理 → 动作序列调度 → 动作映射 → 机器人**。

观测构造按模型需求组合图像、关节、触觉、点云及手部几何。模型输出的关节动作经范围投影生成目标；末端动作经共享 IK 转为机械臂关节目标，并与手部目标一起提交。局部 IK 给出当前可执行的下一步目标，允许保留末端跟踪残差。

评估会话可复用示教采集的 Raw 录制组件，保存观测与指令提交尝试，供后续可视化、导出和分析。

## 核心模块

以下模块位于 [deployment](../dexmani_real/deployment/) 下。

| 模块 | 职责 |
| --- | --- |
| [session.py](../dexmani_real/deployment/session.py) | 装配模型、传感器和机器人，组织多段评估及结果汇总。 |
| [observation.py](../dexmani_real/deployment/observation.py) | 将观测历史编码为模型所需输入。 |
| [inference.py](../dexmani_real/deployment/inference.py) | 对接模型加载、预热、预测和关闭。 |
| [schedule.py](../dexmani_real/deployment/schedule.py) / [rollout.py](../dexmani_real/deployment/rollout.py) | 管理动作序列接续，推进单段评估中的观测、推理、控制和录制。 |

机器人交互、运动约束和 Raw 存储分别复用主包的 `robot`、`control` 和 `data` 模块。

## 执行模式

| 模式 | 方式 |
| --- | --- |
| sync | 等待一次预测，再顺序执行一段动作，随后请求下一次预测。 |
| async | 执行当前动作序列时提前请求预测，依据观测时间衔接仍有效的新动作。 |
| rtc | 在支持 RTC 的模型中，将当前计划的动作前缀交给下一次预测，按已提交进度接续动作。 |

三种模式共用观测与控制链，区别在于推理时机和动作序列的接续方式。

返回[项目架构](../README.md)。
