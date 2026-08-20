# Episode 数据处理原则

本文是数据处理的短版设计原则，不是新的 schema 或阈值合同。当前可执行行为以 [`processed_hdf5.md`](processed_hdf5.md)、`data_processing/` 和 `recording/` 为准。

## 先区分三层数据

```text
Real raw episode（事实，只读；v18，兼容 v17）
        ↓
real-domain processed view（带版本、可重建）
        ↓
model-specific training view（由训练配置决定）
```

- **Raw/Real episode** 保存采集事实：状态、命令、时间、质量、相机和 provenance。不要为某个模型覆盖它。
- **Processed view** 可以删行压紧、resize、派生点云或转换 dtype，但一个 raw 只对应一个 HDF5，并记录 source mapping、profile、规则和输出版本。
- **Training view** 只负责采样、窗口、权重和 batch 组织，不应反向修改前两层。

## 不变的判断顺序

1. 先确认容器和 schema 版本。
2. 再确认时间和 action 的因果语义。
3. 再按模态检查 shape、finite、frame、单位和有效性。
4. 记录质量原因、source mapping 和压紧 bridge 风险。
5. 最后才做 resize、采样、重加权或模型特定变换。

时间错位、错误 frame 和伪造的 unknown 值不能靠平滑或补零修复。

## Action 与 observation

项目中可能同时存在 VR 输入、IK/retarget 结果、最终发布命令和设备反馈。写文档或训练配置时必须写出具体字段，不要把它们都简称为 `action`。

- `joint_state`/`arm_qpos`/`hand_qpos` 是观测。
- `action_arm_joint_sent` 表示已转发给 arm worker 的目标，不是硬件执行 ACK。
- `action_hand_joint` 表示 hand queued target，没有独立 sent/ACK stream。
- `action_arm_joint` 是安全门后的候选；在没有 sent stream 时不能冒充 applied action。
- recorder `success` 是保存/发布结果，不是任务成功；没有任务标签时不要生成 `done`。
- processed `action_ee` 是 `action_arm_ee[9] + action_hand_joint[12]` 的 Real-native
  tracking intent，不是 `action` 的无损 FK，也不是 Sim action 的物理等价物。

## 无效、暂停和小动作

- 先标记原因，再决定整 episode 拒绝、逐行过滤或降低采样权重；不切分 demo。
- 删除无效行后按原顺序压紧，并保留 source index/time/reason；若 bridge 隐藏真实运动，默认拒绝整条。
- 普通暂停不是静态 hold 样本；不要为不存在的观测或命令制造数据。
- 小动作可能是接触、对齐或释放动作，不能只按幅值删除；应结合任务标注、连续性和质量字段。
- 缺失的 `segmentation`、`done` 等语义应省略，不能填零；Real tactile 可以作为
  `contact_force` 保存，但必须声明 SDK-scaled、sensor-native frame 和 SI 未验证，不能
  冒充 Sim world-frame force。

## 对当前 pipeline 的最小要求

修改 `data_processing/` 时，保持以下性质：

- raw source 只读；输出目录通过临时目录校验后原子发布。
- profile 明确决定 hard-valid 模态；不能让缺失模态静默降级。
- point cloud 从 depth 和 metadata 确定性派生；不把 Real frame 重命名成 Sim world。
- processed HDF5 保留 source episode、逐行映射、profile、规则版本和验证结果；Zarr 不复制这些 provenance。
- 默认 `quality-policy=audit` 不依据 suspect 自动删帧；只有显式 `strict` 才删除高置信度
  impulse/stall。无论哪种 policy，压紧 bridge 都独立审计并默认 fail closed。
- 运行前用 `--compare-profiles`/`--dry-run` 检查保留率和 bridge；实际写入后重新读取验证。

更多具体字段、mask 和 CLI 见 [`processed_hdf5.md`](processed_hdf5.md)。
