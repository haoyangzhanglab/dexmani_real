# 文档索引

文档按“导航 → 当前合同 → 专题审计”分层。README/CLAUDE 用于快速定位；数据和手部文档用于修改前核对边界；审计内容不应替代代码验证。

## 快速入口

| 主题 | 文档 | 什么时候读 |
|---|---|---|
| 工作约束与任务路由 | [`../AGENTS.md`](../AGENTS.md)、[`../CLAUDE.md`](../CLAUDE.md) | 所有代码修改前 |
| Real v17 episode | [`dataset/hdf5_episode.md`](dataset/hdf5_episode.md) | writer、reader、replay、数据读取 |
| Real 清洗输出 | [`dataset/processed_hdf5.md`](dataset/processed_hdf5.md) | 生成训练视图或改质量规则 |
| Real → Sim label 边界 | [`dataset/real_to_sim_mapping.md`](dataset/real_to_sim_mapping.md) | 设计字段映射，避免伪造等价关系 |
| Sim HDF5/Zarr | [`dataset/sim_hdf5_zarr.md`](dataset/sim_hdf5_zarr.md) | 读取或审计 Sim 数据 |
| Hand retarget | [`hand_retargeting.md`](hand_retargeting.md) | 改输入、求解、整形、缓存或 hand worker |
| 数据处理原则 | [`dataset/data_process_reference.md`](dataset/data_process_reference.md) | 需要判断“该不该修数据”时 |

## 文档规则

- 代码、schema 和运行时配置是最终事实来源；文档不复制完整默认值或实现细节。
- “当前合同”只写已经实现且有代码路径支持的行为；未来想法放 issue/design note，不放在运行指南中。
- 带日期、样本数量或外部依赖版本的内容属于审计快照，变更实现后必须重新核对，不能当作永久默认值。
- 数据格式文档必须说明 shape、dtype、单位、frame、时间语义和缺失值语义；不确定时写 `unknown`，不要猜测。
