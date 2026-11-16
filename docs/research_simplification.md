# 研究代码精简计划与实施记录

目标是让个人研究中的采集、导出、评估与回放容易运行、理解和修改。成功标准是减少维护同一个事实的位置、缩短常用调用链、降低无用计算，同时保持动作与数据语义正确。文件数量和删行数只作为辅助信息。

本轮起始基线为 `9b6c7c97c1a75cab42f12ce8077d22d6faa528dd`，交付前已整合主分支更新 `8ab6b11efe371d0961d1ce805757b6604d70a990`。配套 Policy 核对版本为 `248acb8131210fe214035b210f53fe0969b3e4e5`。之前审计使用的 `d09095d` 已不是当前主分支；图像针孔契约和点动 HOME 拒绝后的恢复已有修复。

## 参考项目的具体取舍

阅读以下固定版本的实现，只借鉴设计思路，没有复制第三方实现代码。

| 参考 | 借鉴内容 | 在本项目中的取舍 |
| --- | --- | --- |
| [LeFranX：组合机器人](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py) | 直接组合 arm/hand，观测和动作接口易追踪。 | 继续由 `Robot` 持有实际设备；不引入设备注册器、插件工厂或通用机器人框架。其 `send_action` 的同步/异步分支仍顺序发送，不照搬重复分支或同步命名。 |
| [ManiUniCon：模型观测适配](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/customize/obs_wrapper/ppt_rgb_wrapper.py) | 将模型观测转换留在模型边界，区分 sensor、robot、policy。 | 保持 `ObservationBuilder` 与 `ActionMapper` 的实际职责；不迁入多机器人配置平台。其 [chunk wrapper](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/customize/act_wrapper/chunk_wrapper.py) 会把全过期序列的末动作重新排入未来，本项目继续拒绝过期结果。 |
| [T-Rex：目标快照](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/teleop/teleop_targets.py)及[执行模块](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/teleop/arm_hand_control.py) | 明确目标快照与执行端的分工，teleop/policy 复用设备执行。 | 保持观测、动作生成、提交和录制之间的清晰边界；不将现有模块合成巨型脚本，也不引入第二套配置格式。 |

## 架构梳理：每件事只有明确的负责方

本轮沿入口和实际调用链检查，现有目录已大体按职责组织。先减少内部重复，不做只改变路径的大规模搬迁。

| 工作 | 负责方 | 调用和数据边界 |
| --- | --- | --- |
| 实验参数 | `config/real.py` 与具体配置类 | 可信 Python 文件导出 `CONFIG`；运行入口解析一次，具体 I/O/设备使用处检查真实约束。 |
| 设备连接、反馈与提交 | `robot/system.py`、设备 driver/owner | `Robot` 是组合动作入口；arm SDK 留在 owner 进程，避免阻塞主控制流程。 |
| 目标计算 | `teleop/vr.py`、`control/actions.py` | VR/策略生成意图；动作映射完成裁剪或 IK，不自行发送设备命令。 |
| 来源快照 | `sensor/`、`ipc/`、`runtime/observation.py` | 保留来源时间和同源图像；不把重复读取当成新采样。 |
| 模型输入与推理 | `deployment/observation.py`、`inference.py` | 模型转换集中在这里；执行端不承担模型特有预处理。 |
| 一段实验 | `TeleopLoop` / `Rollout` | 持有本段观测历史、目标、录制和结束结果；会话持有跨段设备、模型和传感器。 |
| 资源结束 | 入口 `finally`、`close_runtime`、`PolicySession.close` | 明确停止与关闭顺序；失败透明，不新增统一资源管理平台。 |
| 实验事实与训练表达 | `data/recorder.py`、`raw.py`、`export.py` | Raw 保存来源与提交尝试；Zarr 完成训练所需对齐、变换和分段。 |

## 行动顺序与已实施内容

| 顺序 | 问题 | 实施 | 验收依据 |
| --- | --- | --- | --- |
| 1 | Teleop 的 phase 与同步调用流程、cancel epoch 重复表达段状态。 | 删除 `TeleopPhase`、单次调用的 `try_begin`；`start_episode()` 明确返回是否准备成功，成功后才进入控制循环，epoch 只表示段所有权。 | 新增重复启动、准备阶段 S/Q/Esc/故障、观测丢失、撤销启动、部分 writer 失败测试。 |
| 2 | VR 启动拥有另一份就绪等待循环。 | 复用 `SensorWorkers.start` 的就绪等待；回调继续服务设备、处理取消和检查键盘健康。 | 首帧、取消、worker 退出、超时行为测试。 |
| 3 | Rollout 为一条日志维护 `has_submitted`，另有单调用包装；关闭错误处理重复。 | 删除冗余状态及 `commit_arm`，IK 参考提交就地可见；将重复错误归属处理收敛到 `close` 局部函数，关闭顺序仍明确展开。 | 原有恢复、EEF、调度测试，以及多阶段关闭失败、急停、SystemExit、模型关闭超时测试。 |
| 4 | Raw 是否记录由恢复成功后的状态决定，导致失败提交尝试遗漏。 | 本轮独立复现并修正后，发现主分支已提交同一修复；最终采用 `8ab6b11` 的实现、文档和更完整测试，删除本轮重复修复。 | 保留上游的部分发送失败、Raw 开关、恢复前拒绝、录制失败与提交失败并发、超时及连续区间测试；仅调整已删除内部状态的断言。 |
| 5 | Export 重复预检、重复读取 HOI timing/分段、读取对齐后会丢弃的关节数组，逐段重写 provenance。 | 删除重复预检包装，复用实际读取边界校验；每块数据源直接选定；跳过全裁掉块的数值读取；HOI 来源记录最后写一次。 | 现有导出测试，普通/HOI/HOI 裁剪三种基线对照，数组、属性、连续区间和返回值一致。 |
| 6 | 手点云变换展开逐点 `[T,N,4,4]` 矩阵。 | 使用模板已有的 link 分段应用变换，删除展开索引和大临时数组。 | 标量参考对照、真实 FK、固定点序、空 batch、signed/unsigned offsets；批量基准。 |

保持 CLI、配置格式、设备接口、Raw/Zarr 字段和关节顺序不变。内部移除的辅助方法不保留兼容包装。没有新增运行依赖、恢复机制、配置开关或调度框架。

## 验证记录

完整 CPU 回归：最终基线 `8ab6b11` **358 passed / 4 skipped / 0 failed**，修改后 **379 passed / 4 skipped / 0 failed**，增加 21 个参数化用例。4 项跳过均为 CUDA 用例。所有 pytest 调用使用 `-p no:cacheprovider`，临时目录在仓库之外。采集与部署入口的 `--print-config` 均通过，未连接设备。

环境：Python 3.12、NumPy 1.26.4、PyTorch 2.4.1 CPU、torchvision 0.19.1 CPU、Pinocchio 2.7.0、Ruckig 0.17.3；没有加载真实机器人、相机或策略 checkpoint。运行示例（假设 `dexmani_policy` 位于同级目录）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:../dexmani_policy:third_party/cordvip_geometry/src \
  python -m pytest -q -p no:cacheprovider --basetemp=/tmp/dexmani-tests
```

Ruckig 0.17.3 在本环境需以 `scikit-build-core<0.10`、`cmake<4` 的构建约束及 GCC 构建；约束只用于本次测试环境，没有修改项目运行依赖。

数据对照使用两个合成 Raw Episode，包含三个连续区间，分别导出普通、HOI 和裁剪 HOI 三组。逐项比较所有数组、Zarr 属性、episode ends 与返回结果。HOI 的 timing/segment 读取次数各减半，关节读取字节减半；裁剪用例被观测的 HDF5 数据集读取字节从 114,776 降为 63,480。这是合成用例的逻辑读取量，不是磁盘吞吐基准。

点云基准使用仓库真实 1024 点、24 link 模板及合成刚体变换，NumPy CPU，预热后五组计时取中位数；内存为 `tracemalloc` 跟踪峰值，非进程 RSS。只测 `transform_batch`，不包含 FK、模型推理或真机控制。

| Batch | 原实现 | 分段实现 | 跟踪峰值内存：原 → 新 |
| ---: | ---: | ---: | ---: |
| 1 | 0.044 ms | 0.138 ms | 0.173 → 0.027 MiB |
| 128 | 4.785 ms | 1.526 ms | 20.50 → 2.71 MiB |
| 512 | 26.389 ms | 6.117 ms | 82.00 → 10.83 MiB |

本次样例输出最大绝对误差为 0，输入未修改。大批量收益明显；单帧因逐 link 循环约增加 94 μs。选择简单的统一实现以降低批量导出的内存开销，没有为这一小幅绝对差值增加双实现或阈值配置。不声称真机控制、模型推理或端到端加速。

## 后续优化的进入条件

| 方向 | 继续工作的证据与验收 |
| --- | --- |
| 常用流程易用性 | 在实际站点配置上完成一次采集→导出→训练→部署，记录具体阻塞步骤，再缩短对应入口；不预先添加 CLI 层或配置框架。 |
| 进一步压缩 session/loop | 当多个调用方实际共享同一流程与取消语义时才合并；每次移除状态必须对应唯一权威事实并有异常路径测试。 |
| 热路径优化 | 对实际输入测 CPU/GPU 的分段耗时与 P95/P99，优先处理占比最高的环节；不为短历史插值增加缓存失效机制。 |
| 调度模式或功能删除 | 先有研究路线、实际使用情况或用户明确选择；当前没有证据可以删除 async/RTC、HOI、回放或录制功能。 |
| 真机验证 | 在工作站分别检查准备取消、暂停恢复、采集保存和提交失败后的停止结果；CPU 测试不替代设备时序和 GPU 推理验证。 |

这些是有具体进入条件的后续工作，不作为本轮已完成事项。当前实现集中处理已找到且可验证的重复与无用计算，避免为了重构再建立一套平台。
