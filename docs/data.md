# Raw 与 Canonical 数据

本文说明当前数据表示、缺测与导出规则。采集和导出入口见 [README](../README.md)，离线贯通示例见 [复现入口](reproduction.md)。

## Raw

Raw 保存真实实验中实际发生的观测与控制证据。

原则：

- Raw 一经发布保持不可变；
- 不用后处理掩盖传感器缺测、动作未发布或异常终止；
- demonstration 与 policy rollout 都保留真实实验边界；
- 停止、晚帧、控制失败均保留可保存前缀，显式丢弃才删除当前 capture；
- 写盘、编码或发布失败保留 staging 并报错，不报告为成功 Raw。

录制队列满会停止接收新帧并报告失败；仍健康的 writer 在收尾预算内排空已接受帧，保存前缀和失败详情到 staging，不发布为成功 Raw。真实写入失败或收尾超时仍中止写入，由 writer 自己关闭句柄。

Raw 保存全量已接入物理模态，公共导出再重建末端、指尖与点云。动作是最终尝试的关节目标；未调用的设备目标为 NaN，dispatch 区分未调用、SDK 接受、CRC 不确定、拒绝和未知，均不证明物理到达。
SDK 调用抛出异常或被 Ctrl+C 中断不能证明设备拒绝或未收到命令，对应 dispatch 保留为 UNKNOWN；已经确认的 ACCEPTED 不回退，已采前缀及本次尝试目标随终止保存。

时间、源帧号和 dispatch 直接读取 `data.h5` 中的逐行数据集。导出要求当前 Raw 的全部数据集，缺字段或 shape/dtype 不符时明确拒绝；不从 metadata 读取旧 dispatch，也不补造缺失的时间或状态数组。已保存的时间 0、UNKNOWN dispatch 和辅助模态 NaN 保留原义。旧数据需要使用对应源码版本读取，文件本身不改写。

逐行时间记录主机 monotonic 观测、机器人读取完成及下发完成时间，并保留 RGB/depth 源帧号。新 Raw 的相机时间取 RGB/depth 两通道各自最近推进接收时间的较早者，停帧不会被另一通道的新接收刷新；旧 Raw 按其 metadata 解释。时间 0 表示未知或未下发；旧 Raw 不伪造实测时间。latest-sample 不保证跨模态严格同步，重复源帧可以是正常采样结果。

## Canonical Zarr

Canonical 数据是面向策略训练的派生缓存。

它负责：

- 保留全部 13 个公共字段，训练端再选择所需字段；
- 检查实际 shape/dtype/行数；缺测保留 NaN，模型输入处检查所需数据；
- 保存必要的 numerical preprocessing 信息；
- 保持可由 Raw 重新生成。

Canonical 不是新的实验事实，也不承担长期格式兼容。

一个 store 当前只组织一个 task；`dt` 必须一致，属于训练/部署数值语义。`depth_scale` 在 store 内保持一致以解释存储深度，不作为 Policy 兼容元数据。导出只接受未占用的新路径；分块转换完成后发布带唯一 `data_revision` 的 canonical cache。重新导出使用新路径和新身份，精确续训保留原缓存；失败保留 staging，始终不修改 Raw。

H/W、depth scale、名义 dt 不同的数据分开导出，不静默 resize 或插值。

### 公共字段

以下为单行 shape；除 RGB/depth 外均为 float32。关节向量按 arm 7 + hand 12 的 SDK 顺序排列；空间位置在 xArm base 坐标系中，单位为 m。rot6d 拼接旋转矩阵前两列，无单位。

| 字段 | 单行 shape | 含义与单位 |
| --- | --- | --- |
| `joint_state` | `(19,)` | 实测关节位置，rad |
| `arm_qvel` | `(7,)` | 机械臂关节速度，rad/s |
| `arm_effort` | `(7,)` | 机械臂 effort，保留 SDK 原生单位 |
| `hand_current` | `(12,)` | 手部电流，驱动 mA 语义 |
| `action` | `(19,)` | 尝试下发的关节目标，rad |
| `action_ee` | `(21,)` | 目标 arm FK 的位置 3 + rot6d 6 + hand 目标 12（rad） |
| `contact_force` | `(5,3)` | 已扣偏置的 SDK 聚合触觉值 |
| `tactile_force` | `(5,120,3)` | 已扣偏置的 SDK 稠密触觉值 |
| `fingertip_points` | `(5,3)` | 实测关节重建的五指指尖位置，m |
| `eef_pose` | `(9,)` | 实测 arm FK 的位置 3 + rot6d 6 |
| `rgb` | `(H,W,3)` | uint8，RGB 顺序，保持采集分辨率 |
| `depth` | `(H,W)` | uint16，对齐彩色；乘 depth scale 得到 m |
| `point_cloud` | `(P,6)` | XYZ（m）+ RGB（0–1），P 由点云配方指定 |

聚合/稠密触觉独立有效，导出不重复扣偏置；电流不是触觉力。RGB 视频编码与深度对齐不等于全部原生相机字节无损。

### 重建与取窗

桌面和安装/运动学按当前代码配置重建，不要求历史参数快照；相机参数仍使用对应 Raw 的信息。缺重建参数或空点云只使点云为 NaN，并在 export_report 中说明。
Zarr 的 `row_info` 保留 Raw 的逐行时间、源帧号和 dispatch，`meta/episode_ends` 保留 episode 边界；`dt` 仅为名义周期。缺可解码 RGB-D 或有效 depth scale 时不能完整导出，数值字段仍可单独读取。

[离线窗口示例](../examples/read_policy_windows.py) 对 RGB 与触觉策略分别选所需字段和连续原行号窗口，使用 dispatch，不跨 episode 或缺失所需样本的行；不把时间抖动当缺测。
Policy 训练按原始记录行取窗，不假设等间隔或自动插值。实际训练窗口的有限性、dispatch、时间连续性筛选与 normalizer 规则，以及 Raw 外的评估结果和 query trace，统一见 [数据与追踪](policy_execution.md#数据与追踪)。
