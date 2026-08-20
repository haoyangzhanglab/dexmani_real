# 文档索引

文档按“导航 → 当前合同 → 专题审计”分层。README/CLAUDE 用于快速定位；数据和手部文档用于修改前核对边界；审计内容不应替代代码验证。

## 快速入口

| 主题 | 文档 | 什么时候读 |
|---|---|---|
| 工作约束与任务路由 | [`../AGENTS.md`](../AGENTS.md)、[`../CLAUDE.md`](../CLAUDE.md) | 所有代码修改前 |
| 代码风格与重构准则 | [`CODE_STYLE.md`](CODE_STYLE.md) | 新建或重构 Python 模块前 |
| 关键行为护栏 | [`golden_paths.md`](golden_paths.md) | 修改入口、硬件、IPC、录制或部署链路前 |
| Real episode（写 v18，读 v17+v18） | [`dataset/hdf5_episode.md`](dataset/hdf5_episode.md) | writer、reader、replay、数据读取 |
| Real 清洗输出与 Policy Zarr | [`dataset/processed_hdf5.md`](dataset/processed_hdf5.md) | 生成训练视图、改质量规则或核对 Zarr 合同 |
| Real → Sim label 边界 | [`dataset/real_to_sim_mapping.md`](dataset/real_to_sim_mapping.md) | 设计字段映射，避免伪造等价关系 |
| Sim HDF5/Zarr | [`dataset/sim_hdf5_zarr.md`](dataset/sim_hdf5_zarr.md) | 读取或审计 Sim 数据 |
| Hand retarget | [`hand_retargeting.md`](hand_retargeting.md) | 改输入、求解、整形、缓存或 hand worker |
| 数据处理原则 | [`dataset/data_process_reference.md`](dataset/data_process_reference.md) | 需要判断“该不该修数据”时 |

## 迁移与审计快照

带日期的审计/迁移文档属于快照：实现变更后必须重新核对，不能当作永久合同。

| 主题 | 文档 | 状态 |
|---|---|---|
| 全仓分阶段迁移与简化 | [`dexmani_real 分阶段迁移与简化指南.md`](dexmani_real%20分阶段迁移与简化指南.md) | Phase 0 与第一批薄入口已实施；其余目录经依赖审计后暂不机械合并 |
| XArm7 arm 栈化简迁移计划 | [`Dexmani XArm7 大幅化简迁移操作指南.md`](Dexmani%20XArm7%20大幅化简迁移操作指南.md) | 第一轮（阶段 1–4）已实施，见文末执行记录 |
| XArm7 化简迁移对账（指南 vs 代码审计） | [`XArm7 化简迁移对账.html`](XArm7%20化简迁移对账.html) | 2026-08-19 审计快照，含实施进展区块 |
| XHand 简化重构执行指南 | [`xhand_simplification_execution_guide.md`](xhand_simplification_execution_guide.md) | 手部工作线计划 |

## 文档规则

- 代码、schema 和运行时配置是最终事实来源；文档不复制完整默认值或实现细节。
- “当前合同”只写已经实现且有代码路径支持的行为；未来想法放 issue/design note，不放在运行指南中。
- 带日期、样本数量或外部依赖版本的内容属于审计快照，变更实现后必须重新核对，不能当作永久默认值。
- 数据格式文档必须说明 shape、dtype、单位、frame、时间语义和缺失值语义；不确定时写 `unknown`，不要猜测。
- Real 与 Sim 数据按 domain 分开；共享键名只表示各自已声明的语义，不证明 frame、单位或物理来源等价。
