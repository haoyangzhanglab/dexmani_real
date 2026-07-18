# DexMani 50Hz→16Hz 迁移架构复查报告

> 生成方式：2026-07-17 多智能体复查（6 维并行审查 → 每条发现独立对抗验证 → 汇总裁决），35 agents。
> 姊妹文档：`docs/16hz-rationale-2026-07-17.md`（为什么是 16Hz —— 决策原因与设计原则）。

## 1. 总体裁决：**有条件通过**

两速率架构（决策/录制外环 16Hz、ArmInnerLoop 50Hz 看门狗零改动、相机/点云 30Hz）架构上成立。8 条设计原则在核心链路上落实到位：滤波系数按 τ 换算（signal_utils）、速度/时长语义正确换算（手 clip、min/max_frames）、秒计超时不换算（target_timeout_s）、绝对期限调度、eps=0.5 就近取整、16 仅在入口点单点定义。对抗验证的决定性结论是：**本次迁移 diff 未引入任何 critical/major 级代码缺陷**——迁移新引入的问题全部为 minor/info（录制回写边界、回退陷阱注释、测试缺口）。全部 3 个 major 均为历史遗留被迁移放大或复查错失（力矩安全门恒死、Home→Begin 锚定竞态、Zarr 速率元数据缺失），不阻塞本 diff 合入，但前两项是 P6 真机验收的硬前置、第三项是首次混速率导出/训练前的硬前置。

**通过条件**：① 提交前修复迁移新增回写的边界缺陷（#4）并同步 CLAUDE.md 与失实注释（#10-13）；② P6 上机前修复 #1、#2、#5；③ 首次导出前修复 #3、#7。

## 2. 确认问题清单（severity 降序，跨维度重复已合并）

> 显著声明：**不存在 migration-introduced 的 critical/major 发现**。以下 major 均为历史遗留。

### Major（3 条，均历史遗留）
1. **生产 episode /arm_qvel、/arm_tau 全 NaN，validate_action 力矩门恒被静默跳过、温度门无调用点**（候选 E）— `dexmani_real/robot/interface.py:171-173`、`dexmani_real/robot/validate.py:56`、`controller.py:304,338`。历史遗留被放大。触发：所有生产遥操 episode 两流全 NaN（DataValidator 放行）；碰撞/卡阻时力矩预发送门不拦截，仅剩固件防线——安全门呈"已接线假象"。修：内环 50Hz 回读顺带缓存 tau/qvel 供外环取用（勿外环直调 SDK，避免与内环争用）+ tau 全 NaN 节流告警，10-30 行。
2. **Home→Begin 竞态：内环重建后不 wait_ready，mapper 锚定 FK(zeros)**— `dexmani_real/teleop/core/controller.py:545`、`inner_loop.py:143`。历史遗留（迁移 P3 复查错失；实为确定性命中非窗口竞态）。触发：H 归位后按 B，全 episode 目标偏移数十 cm，臂以 ramp ≥11°/s 物理移向错误位姿，无兜底。修：`_ensure_inner_running` 重建后 `wait_ready(10)`，失败拒入 TELEOP；`__init__` 同理，5-8 行。
3. **Zarr 导出零速率元数据；混速率目录静默拼接；--align 缺省 dt 仅取首 episode**— `dexmani_real/tools/export_hdf5_to_zarr.py:577-585, 745-748`。历史遗留被放大（episodes/ 固定目录下 50/16Hz 必然共存）。触发：不带 --align 时两种 dt 帧无标记拼接毒化训练；带 --align 时 16Hz 数据被静默上采样到 50Hz（2/3 帧为插值），产物无速率痕迹不可事后审计。修：load_episodes 收集各 episode control_hz，不一致告警/要求显式 --align_dt；zarr meta 写 control_hz，15-20 行。

### Minor（12 条）
4. **【迁移引入】手回写双缺陷：发送失败帧录上一 tick 值且 flag_held 不覆盖；XHand stub 模式下 /action_hand_joint 全程冻结为 home_qpos**— `controller.py:354-359`、`xhand.py:509-510,537-541`。触发：瞬时 RS485 CRC 错误帧数据错位；无 SDK 机器录臂主导数据时手动作流被静默替换。修：改用 `send_action` 返回值 `hand_ok`/`hand_cmd`（stub 返回 None），`not hand_ok` 并入 held，~6 行。
5. **_last_hand_cmd/_last_good_hand 存 pre-clip 请求值：16Hz clip 常态饱和下 hold/力矩超限时手反而继续向陈旧超前目标闭合加压**— `controller.py:311-312,431-432`。历史遗留被放大（50Hz 时 clip 从不触发）。修：回写处同步两个 hold 变量为实发值，2-4 行。
6. **【迁移引入】手 E3 clip 回退陷阱：CTRL_HZ 改回 50 得 0.031 rad 而非旧默认 0.3（紧 9.6x），A/B 对照被污染**（候选 B）— `examples/real/vr_teleop_shm.py:70,130`。已被 docs/16hz-rationale 自我文档化，残余为代码注释缺回退指引 + 手 clip 链路零测试。修：定义处注释 + 换算单测，5-10 行。
7. **export --validate 的 min_frames 兜底 50 帧在 16Hz=3.125s，与采集端 16 帧/meta 矛盾，误导数据清洗**— `export_hdf5_to_zarr.py:717`、`data_validator.py:65`。历史遗留被放大。修：按 /meta control_hz 派生缺省；`or 50` 改 `is not None`，5-10 行。
8. **forward-fill 重复时间戳必然触发 timestamp_monotonic 严格递增校验失败（连带 no_duplicate_frames）**— `data_validator.py:232-234`、`timestamp_buffer.py:175`。历史遗留设计冲突（两条校验路径当前均 opt-in 非阻塞）。触发：任一 tick 超期 >62.5ms → 健康 episode 判 FAIL。修：/timestamp 改写栅格时刻或校验放宽为非递减+重复率阈值，5-15 行。
9. **【迁移引入】replay_hz 从 fps 派生无钳制且消费端零测试**— `examples/real/replay_traj.py:135,585`。触发：含暂停的旧 v6 文件 fps 被稀释 → 静默慢速重放（方向安全但验收失真）；fps=0 干净崩溃。修：fps 范围钳制（1-100 外 warning+50）+ 2 个消费端单测，~30 行。
10. **timestamp_buffer 两处 docstring 与行为相反：快源实为 first-wins（eps=0.5 下迟到样本挤掉新鲜样本）、缺槽实为下一样本回填非 forward-fill**（候选 A）— `dexmani_real/recording/timestamp_buffer.py:33-34,79`。文档过时。修：如实改写 4 行；latest-wins 语义改造单独评估。
11. **low_pass_alpha docstring 颠倒 alpha=0 语义：写 0.0=pass-through，实际 0.0 冻结输出（1.0 才是直通）**— `dexmani_real/teleop/vr/hand_retarget.py:396,407-409`。文档过时（迁移使其成主入口；候选 G 经核实覆盖真正生效，setter 改 filter.alpha 每帧实时读取）。修：4 行注释。
12. **inner_loop E3 注释按 50Hz 外环书写：16Hz 实际余量 ~3x 非 10x；单异常目标含容 0.3→0.9 rad（1:1→3:1）**（候选 D）— `dexmani_real/robot/inner_loop.py:57-59,76-78,466-468`。文档过时。修：注释按外环频率参数化，~8 行。
13. **CLAUDE.md schema 章节五处失配**（候选 C）— `CLAUDE.md:98,149,155,176,184`：schema v6→v7 且缺 control_hz/hand_max_qvel_deg_s、0.5s→0.2s、dt=1/50、20ms spacing、外环 50Hz 表述（:111/:114 内环表述勿改）。修：~6 行文档。
14. **【迁移引入】eps=0.5 的 ±dt/2 边界与快源同窗双样本路径零测试**— `tests/test_episode_recorder_hz.py:55`（抖动仅 ±5ms）。修：2 个 get_accumulate_timestamp_idxs 直接单测，~25 行。
15. **【迁移引入】tau_from_alpha 本体及边界分支零覆盖，测试用手工 math.log 自证**— `tests/test_signal_utils.py:14,22`、`signal_utils.py:33,46-49`。修：保留独立公式 oracle，另加 tau_from_alpha(1.0/0.0/负值) 直接断言，~8 行。

### Info（3 条）
16. **【迁移引入】skip_initial_frames=3 帧=0.1875s，较旧 0.2s 短 12.5ms**（候选 F）— `vr_teleop_shm.py:181`。原则 3 固有舍入；如需严格 ≥0.2s 改 ceil→4 帧，1 行。
17. **CLAUDE.md:107 将 EMA 0.8/0.4 描述为管线行为，生产 SHM 路径实为 1.0/1.0 直通**— 文档过时，16Hz 下更误导（平滑全靠 Mode 6 固件）。修：1-2 行。
18. **RateManager 新语义透传至 50Hz 内环（"内环零改动"仅指文件未动）；re-anchor 测试仅下界断言**— `inner_loop.py:279`、`tests/test_rate_manager.py:46`。修：补上界断言 + 文档措辞，~3 行。

## 3. 修复优先级建议

**提交本 diff 前必修**（合计 ~30 行，全部低成本）：
- #4 手回写双缺陷（本次迁移唯一的行为级代码缺陷）
- #13+#17 CLAUDE.md 同步（项目指令文件事实错误会被后续开发直接采信）
- #10、#11、#12 三处失实 docstring/注释（#11 有误操作致手冻结撞物的诱导风险）
- #6 回退指引注释 + 手 clip 换算单测

**P6 上机前应修**（真机安全 / 数据管道硬前置）：
- #2 Home→Begin wait_ready（非预期臂运动 + 整段数据报废）
- #1 力矩死门/NaN 流（policy rollout 硬前置，最低限度先加 tau 全 NaN 告警使死门可见）
- #5 hold 变量 pre-clip 同步（抓握中持续加压风险）
- #3 Zarr 混速率检测 + meta control_hz、#7 export min_frames（首次混速率导出/训练前）

**可排期**：#8 timestamp 校验语义调和、#9 replay fps 钳制+测试、#14/#15/#18 测试补齐、#16 ceil 取整（或接受现状）。

## 4. 误报驳回记录（供人工复核）

- 「手 clip 回退陷阱构成缺陷」（filter-math 维度版本）— 驳回：90°/s 速度语义是定案设计，回退陷阱已被 docs/16hz-rationale-2026-07-17.md 逐字文档化并给出完整回退步骤，属已知权衡重复上报。注意：safety/tests 维度对同一事实以收窄口径（残余注释+测试缺口）确认为 minor（本报告 #6），两者不矛盾——驳回的是"未知缺陷"定性，确认的是残余可操作项。
- 「min_frames 贯穿链路无集成测试」— 驳回：validate_on_stop 默认 False 且全仓无处置 True，场景不可达；假设异常非静默（writer 线程仅捕获四类异常）；风险为双重假设且 mypy 可检出。

## 5. 测试缺口清单（tests-rollback 维度）

1. 手 clip 链路零测试：tests/ 无一处覆盖 max_delta_rad/XHandConfig，16Hz→0.098 换算与回退语义差异无 pin（对应 #6）。
2. 原则 5 消费端零测试：replay `load_trajectory` 的 control_hz→fps→50 回退链、export align_dt 派生均无覆盖，且 replay_hz 直驱真机速率无钳制（对应 #9）。
3. eps=0.5 边界零测试：现有抖动 ±5ms << 声明边界 dt/2=31.25ms；负向越界丢帧与快源同窗 first-wins（候选 A 的文档谎言路径）均无测试（对应 #14）。
4. tau_from_alpha 零覆盖：生产入口直接调用的函数本体及 alpha>=1.0/<=0 边界分支无任何断言（对应 #15）。
5. re-anchor 断言强度不足：test_reanchor_after_long_block 仅下界，误写 now+2*period 仍绿（对应 #18）。
