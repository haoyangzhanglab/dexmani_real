# 触觉力单位语义 fact-check 与决策记录

日期 2026-09-11 · 状态：结论已记录，未改变任何行为 · Hardware validation: NOT RUN

## 1. 背景与问题

2026-09-11 提案：**"XHand SDK 输出的力单位可直接视为牛顿（N）——历史上的 0.1 scale
错误已经修正；据此删除 `tactile_unit_code` / `xhand_sdk_native_unknown_si` 相关冗余
机制与变量，并重新修复 validity。"**

由于该提案会改变已冻结数据集（processed v18 / Zarr v11，14112 行）的物理单位声明，
先行执行了只读 fact-check（5 路独立证据线 + 对全部 load-bearing 引用的人工复核）。
本文是结论记录。规则的单一 owner 仍是
[xhand_tactile_correctness_upgrade_guide.md](xhand_tactile_correctness_upgrade_guide.md)
（:133、:290-296）；本文不改变任何规则或代码行为。

代码行号对应 2026-09-11 工作区（含当日未提交的 contact validity 修复）。

## 2. 结论摘要

| 命题 | 判定 | 关键依据 |
|---|---|---|
| "0.1 scale 错误已修正" | ✅ 成立 | driver 自 `aac23d2` 起零缩放；salvage v26 数据经冻结转换器 ×10 恰好还原一次 |
| "SDK 输出单位是 N" | ❌ 不成立（无证据） | 厂商零文档；唯一 N 声明（PI-R2）无出处；指南 :133 已明确拒绝照抄 |
| "unit 机制冗余、可删除" | ❌ 不成立 | 执行链 78 处触点；删除需双 schema bump + 全量重建，而 mask 变化为零 |
| 最终决定 | **搁置，仓库零改动** | 提案人 2026-09-11 确认；解锁条件 = known-load 实验（§7） |

核心逻辑一句话：**"0.1 修复"确立的是"数值 = SDK 原生"，而不是"SDK 原生 = N"。**
0.1 本身就是无依据的猜测，而 `unknown_si` 标签诞生于移除 0.1 的同一个 commit
（`aac23d2`）——两者是同一决定（"单位未知，诚实标注"）的两面。

## 3. 0.1 scale 完整考古

| Commit | 日期 | 变化 |
|---|---|---|
| `2a57d44` | 2026-05-26 | xhand.py 初版即带 `tactile_scale: float = 0.1`（:78，应用于 :388/:405），**无任何依据注释** |
| 2026-08-19 时代（`c445373` 前后） | | 模块常量 `_TACTILE_SCALE = 0.1`；注释明确 "Preserve the deployed scale without claiming an SI conversion"、"not claimed to be a verified SI conversion" |
| `f2756ae` | 2026-09-06 | 目录迁移 robot/xhand.py → robot/drivers/xhand.py，scale 保留 |
| `aac23d2` | 2026-09-09 | **移除 0.1**（"remove driver 0.1 scale; calc_force/raw_force = SDK native − software bias"）；同一 commit 新增冻结 v25→v26 转换器与单位字符串 `xhand_sdk_native_unknown_si` |

当前状态（HEAD）：

- driver 零缩放直通：`_parse_tactile_sum`（robot/drivers/xhand.py:650-659）与
  `_parse_tactile_force`（:661-676）直接读取 SDK `calc_force`/`raw_force`；
  唯一运算是软件 bias 减法（:519-520、:529-530）
- 冻结转换器方向与次数：tools/convert_raw_v25_to_v26_tactile.py:113（直接 v25 →
  scale=10.0）、:145（应用于 `hand_contact` 与 `hand_tactile_force`，:67）；
  v24-derived 数据拒绝猜测、要求显式 `--converted-v24-scale`（:97-113）
- forensic 确认无二次缩放：invalid_frames_export_incident.md:293
  "绝未对已恢复的 v26 再次缩放"

结论：**当前 v28 录制与 salvage v26 数据集的数值均为 SDK 原生值**；"0.1 已修正"对
两者都成立。但原生值以什么物理单位计量，厂商没有说，也从未被测过。

## 4. "单位 = N" 证据盘点

| 证据源 | 内容 | 判定 |
|---|---|---|
| SDK 包（xhand_controller 1.5.2） | 不透明 pybind11 `.so` + 2 个平凡 `.py`；无头文件/stub/手册；`strings` 扫描只有字段名（`calc_force`、`raw_force`、`PXSR_ForceData` fx/fy/fz），**零单位字符串** | 无证据 |
| PI-R2（参考代码库） | xhand_robot.py:267 `verify_thresh_n: max acceptable \|F\| post-reset in N (default 2.0)`；:282 "Vendor reset_sensor() alone leaves ~5-30 N residual offsets"；:317/:323 打印带 N 后缀——但**未引用任何厂商文档、标定实验或 SDK 源码** | 唯一 N 声明，无出处 |
| DexUMI | 同样直通 native scale；其 contact cutoff=10 是经验策略值，"provides no evidence that 10 SDK units == 10 N"（指南 :156-160） | 中性 |
| 本仓库立场 | 指南 :133 明确拒绝照抄 PI-R2 的 N 声明（"we do not have independent known-load calibration for this installation"）；:290-296 规定保留 `unknown_si` "until a known-load experiment proves an SI conversion"；:404-407 "Interpretation: `2.0 XHand SDK-native units` **not** `2 N`"；xhand.py:76-78 阈值注释 "(not Newtons)" | **明确拒绝 N 声明** |

跨仓库强制执行（不只是文档约定）：

- policy 硬门禁：dexmani_policy/dexmani_policy/deployment/export.py:525-527
  （`contact_force_unit ≠ xhand_sdk_native_unknown_si` → 拒收）、:534-536
  （`contact_force_si_verified ≠ False` → 拒收）
- real 验证器：dataset/processed.py:102-105 硬编码
  `_CONTACT_FORCE_SI_VERIFIED = False` / `_TACTILE_FORCE_SI_VERIFIED = False`；
  :414-429 对声称 `True` 的文件直接验证失败——架构刻意悲观

## 5. unit 机制为什么不是冗余

它是"未经证明的物理标定不得写入数据集"的执行链，real 侧 28 文件 64 处 + policy 侧
5 文件 14 处触点：

```text
driver/hand_worker（每帧发 unit_code=0，hand_worker.py:112）
  → IPC dtype（ipc/schema.py:272）→ raw v28 字段（recording/storage/schema.py:63）
  → processing validity 合取项（dataset/processing.py:526、:559）
    + processed telemetry 数组复制（:516）+ unit attrs（:388-397）
  → processed validator 精确比对（dataset/processed.py:399、:406、:414-429）
  → export allowlist（dataset/export.py:100-112）→ Zarr root attrs 精确比对（:291）
  → policy 硬门禁（export.py:525-536）→ deployment semantics 精确比对（deployment/config.py:192、:204）
```

存储代价：raw/processed 各 1 字节每行（14112 行 ≈ 14 KB）+ 若干 attr 字符串。换来的是
每行数据自述单位身份，任何未来的单位变更都被迫显式地过门禁，而不是静默混入。

注：2026-09-11 的 `contact_force_valid` 修复（finite ∧ provenance ∧ calibrated ∧
unit native）是 **unit-agnostic** 的——它 gate 的是"单位身份与声明的 representation
一致"，与单位最终是 N 还是 native 无关，任何路线下该修复都原样成立，无需重修。

## 6. 删除/改写代价矩阵（实测）

基线：全部 14112 行 processed 与 Zarr 的 `tactile_unit_code == 0`（只读实测），unit
合取项恒真；唯一 invalid 行（episode_20260827_195951 row 157）因 `calibrated=False`，
与 unit 无关。

| | X：单位声明改 newton | Y：删 unit_code 合取项 + 停止复制数组 |
|---|---|---|
| mask 变化 | 0（字符串只在 attrs） | 0（合取项恒真，bit-identical） |
| 磁盘 artifacts | 60 processed + Zarr attrs 全部失配 → validator/export 拒收 → **必须全量重建** | dataset key 结构失配 → **processed v18→19 + Zarr v11→12 bump + 全量重建** |
| 跨仓库 | policy 硬门禁 + 4 个测试必须协同修改 | policy 不受影响 |
| raw 侧 | 不可动 | **不可删**：raw 不可变；v28 精确 layout 校验缺字段即拒（recording/storage/schema.py:87-102）；若 bump v29 删字段，现有全部 v28 raw 会被 runtime reader（reader.py:150-153 只收 current）与 processing（processing.py:86-92 只收 current+26）**双双拒读** |
| 诚实性 | `si_verified=True` 无硬件实验违反 AGENTS.md §4（"Never claim hardware validation unless real hardware was actually exercised"），属 CLAUDE.md 明令禁止的验证门绕过 | 拆掉的是 2026-09-11 刚加入的 defense-in-depth 与逐行审计线索 |
| 重建成本 | 离线可行：examples/process_episodes.py + examples/export_policy_zarr.py；~14 GB I/O，点云重算为主 | 同左 |
| 净收益 | 把一个真标签换成一个未经证明的标签 | ~14 KB 数组 + 2 个恒真合取项 |

结论：两个方向的改动都没有可辩护的收益/代价比。

## 7. 解锁路线：known-load 实验（协议草案，NOT RUN）

指南 :296 写明的唯一解锁条件。最小可行协议：

1. **器材**：精度已知的参考测力计/载荷传感器，或标准砝码（如 100/200/500 g，沿指尖
   fz 轴静态加载，期望力 = m·9.81）
2. **流程**：现有软件 bias 标定归零 → 每指 3 档静态载荷（≈2/5/10 N，覆盖阈值 2.0 与
   部署工作区间）→ 每档采集 ≥5 s `calc_force` 取均值 → 计算每指每轴比值
   `|calc_force| / F_期望`
3. **判据**：5 指比值均落在 1.0 ± 5%（可更严）→ native == N 得证；记录残余分布、
   温度与安装姿态
4. **证据**：实验记录写入 artifacts/ 并被本文与指南引用

实验通过后才具备执行迁移的资格（每一步都有证据背书）：flip processed.py 的
`*_SI_VERIFIED` 常量 → 更新单位字符串（robot/model.py 词汇表）→ next processed /
Policy-Zarr schema version bump → 60 episode 离线重建 + 重导出 → policy 门禁与测试协同更新 →
deployment/config.py semantics → data_schema.md / 指南 / manifest 同步。

若提案人持有**厂商文档**（写明 `calc_force` 单位为 N）：指南要求的是"对这套安装"的
验证，最小组合 = 厂商文档 + 一次单点台架核对，并显式修订指南 :133/:296 本身。

## 8. 文档关系与附带发现

- 规则单一 owner：xhand_tactile_correctness_upgrade_guide.md（:133、:290-296、:404-407）
  ——本文只记录 2026-09-11 的 fact-check 与决定，不改变规则
- 单位声明的数据合同：data_schema.md（§1 :104、§2 attrs 表）
- 0.1 还原 forensic：invalid_frames_export_incident.md（:289-293）
- 相关既有硬件待办：指南 §9（注意 :1281 明确"目的不是证明 Newton 单位"；known-load
  单位实验不在 §9 之列，协议见本文 §7）
- 附带发现：policy 仓遗留的旧 `robot_data/pick_place_toy.zarr`（schema v5，
  `contact_force_unit=sdk_scaled_unknown_si`）会被当前 policy 门禁拒收——历史遗留，
  与本决定无关，是否归档另行决定

Hardware validation: NOT RUN
