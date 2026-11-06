# XHand 灵巧手重定向算法重设计任务书

> 目标仓库：DexMani Real  
> 任务基准：2026-11-06 的 main，执行前必须重新确认最新 HEAD 与工作区状态  
> 编写时基准 commit：911b933fb1f02efcbda27cf0cbf5dfb08fc17056  
> 目标：在吸收 Sharpa 与 Wuji 有效设计的基础上，重构 XHand 专用几何重定向；优先保证机构适配正确、易理解、实时高效，不引入通用框架或未经验证的附加机制。

## 1. 任务目标

重设计 dexmani_real 的 XHand 手部重定向，使人体 21 点手部观测能够稳定映射为 XHand 12-DoF 关节目标。

本次不是将 Sharpa、Wuji 的算法直接拼接，也不是增加一套通用 retargeting 框架。设计应明确考虑三个机器人手的差异：

- Sharpa Wave：22 DoF；拇指 5 DoF，食指/中指/无名指各 4 DoF，小指 5 DoF。四指具有独立 MCP 侧摆和 DIP，小指还有额外基部运动。
- Wuji Hand 2_1：20 DoF；五指各 4 DoF，具有独立的基部外展/内收及 PIP/DIP 运动。
- XHand：12 DoF；拇指 3 DoF，食指 3 DoF，中指/无名指/小指各 2 DoF。只有食指具有有限的基部侧向运动（URDF 约 ±0.174 rad）；中/无名/小指基部固定，非拇指手指没有独立 DIP。

URDF 运动学段长度也不同。按各仓库用于重定向的相邻目标 Frame 在中立构型下的距离，典型非拇指两段约为：

| 模型 | 第一段 | 第二段 |
| --- | ---: | ---: |
| Sharpa Wave | 47.0 mm | 31.5 mm |
| Wuji Hand 2_1 | 37.6 mm | 29.5 mm |
| XHand | 55.8 mm | 42.25 mm |

这些数值是 URDF 目标 Frame 的运动学距离，不是指腹接触面距离。实际实现必须从当前 XHand URDF 和 FK 重新提取，不得将表中数值硬编码。

**核心原则：先根据 XHand 的真实运动学结构构造可实现的几何目标，再进行 IK；不要让优化器持续追踪 XHand 无法表达的人体 MCP 侧摆或独立 DIP 运动。**

## 2. 参考项目与取舍

### Sharpa 借鉴

- Flat-hand Calibration 的个体形态校准思想，但不要求本项目增加平展手掌流程。
- Palm/Finger 分层尺度适配思想，但改为从人体骨段标定和 XHand URDF 自动构造目标段，不增加人工 segment-scale 配置。
- Vector Retargeting 的几何目标清晰、实现直接。

### Wuji 借鉴

- 近端、末端方向与多层手指几何共同表达姿态。
- 通过拇指与其他指尖的距离识别对指状态。
- Pinocchio 批量 FK/Jacobian、解析梯度和 warm-start。

### 明确不照搬

- 不复制 Sharpa/Wuji 的高自由度人体关节一一对应。
- 第一版不引入 Wuji 的 TipDirVec/FullHandVec 全局自适应权重切换。
- 不增加大量逐指、逐段 YAML scaling 参数。
- 不增加固定 DIP-PIP 比例、Hyperextension 软约束或额外 Articulation Loss。
- 不增加多个优化器、通用机器人注册表、任务模式状态机或 arm-hand 联合 IK。

## 3. 最终算法：Embodiment-aware Geometric Retargeting

保留现有输入与控制链：

**HTS 21 landmarks → Palm Frame → Anthropometric Calibration → XHand-aware Target Construction → Position/Direction/Pinch/Temporal IK → 12-DoF command**

只重点重构目标构造和对应的优化目标。继续使用现有 HandRetargeter、HandObjective、HandFK、Pinocchio、NLopt SLSQP 和提交参考生命周期。

### 3.1 人体输入与标定

保留 HTS 右手米制 21 点和当前 Palm Frame 约定，不引入 MANO、PCA 或额外手部模型。

保留现有 15 个骨段长度和掌宽的多帧标定：

- 25 个新鲜、来源时间递增的有效样本；
- 中位数作为标定值；
- 相对 MAD 检查稳定性；
- 运行时骨长偏差检查；
- 不要求平展手掌。

标定骨长应真正参与非拇指远端合成方向的构造，而不只是用作输入校验。对 PIP-DIP-TIP 两段，使用标定骨长加权当前骨段单位方向：

\[
v_{\mathrm{distal}}^h =
l_{\mathrm{PIP-DIP}}^{cal} u_{\mathrm{PIP-DIP}}^h
+
l_{\mathrm{DIP-TIP}}^{cal} u_{\mathrm{DIP-TIP}}^h
\]

该向量表示人体两段远端骨骼的合成几何方向。这样可保留两段骨长比例，并减少单帧骨长估计噪声的影响。

掌宽继续用于人体几何稳定性检查；不再用一个全局 Palm Scale 直接缩放所有指尖距离。XHand 的手掌布局和指根位置由 URDF 决定。

### 3.2 URDF 驱动的机器人几何

在 HandRetargeter 初始化时，基于当前 XHand URDF 和 Pinocchio 中立构型一次性提取：

- 五指目标 Frame；
- 各指基座/第一目标 Frame 的位置；
- 两个机器人运动学段的实际长度；
- 拇指和食指基部关节的轴、轴心及 URDF 限位；
- 每根手指两个屈曲关节在手基座坐标系下的轴方向。

不要硬编码段长、掌宽、旋转轴方向或关节限位。关节顺序继续通过 joint name 映射到 SDK 顺序。

XHand 当前各指的两个屈曲轴在中立构型下应近似平行，形成对应的局部屈曲平面。用 URDF/Pinocchio 实际轴向验证这一事实；不得直接假设所有机器人手指都在同一个全局平面内。

可将初始化得到的少量固定几何量放在一个简单的数据结构中传给目标构造函数；不要新增通用 RobotModel/Registry/Config 子系统。

### 3.3 机构适配与目标构造

所有目标位置均在当前 Palm-aligned/hand-base 对齐约定下构造。目标段长度必须等于对应 XHand URDF 的实际段长。

对向量投影到目标屈曲平面：

\[
P_i = I - n_i n_i^\top,\qquad
\hat u_i = \frac{P_i u_i}{\|P_i u_i\|}
\]

其中 \(n_i\) 是该指当前目标构型下的屈曲平面法向，由 URDF 屈曲轴及可动基部关节的目标转角得到。若投影向量退化到无法定义方向，使用现有 HandGeometryRejected 拒绝该帧；不要引入静默的任意方向 fallback。

**Thumb（3 DoF）**

- 用人体 CMC→MCP 方向估计拇指基部对掌转角。
- 基部转角根据 XHand 的 thumb_bend_joint 轴和轴心计算，并限制在该关节 URDF 限位内。
- 用该转角旋转中立构型下的拇指第一目标 Frame 位置和屈曲平面法向；不要仅凭全局 XY 假设圆轨迹。
- MCP→IP、IP→TIP 分别映射到 XHand 两个实际屈曲段。
- 两个目标位移分别为对应 URDF 段长乘以投影后的单位方向。
- 保留拇指基部位置、指尖位置和末端方向目标。

**Index（3 DoF）**

- 将有限的 index_bend_joint 用于人体食指基部侧摆。
- 通过人体 MCP→PIP 方向在垂直于 URDF 基部侧摆轴平面上的方位，估计目标基部转角；按 URDF 限位截断。
- 用该目标转角旋转食指第一目标 Frame 位置和屈曲平面法向。
- MCP→PIP 作为近端方向；PIP-DIP-TIP 按上述标定骨长加权向量构造合成远端方向。
- 近端和远端目标位移分别使用 XHand 对应段长。
- 食指第一目标 Frame 位置应参与位置损失（使用现有 shape_weight），使有限的基部侧摆确实受到几何目标约束。

**Middle / Ring / Pinky（每指 2 DoF）**

- 基座位置固定为 URDF 中立构型的实际基座位置。
- 不为人体 MCP 外展/内收构造独立机器人目标，也不让固定基座位置产生无意义的位置损失。
- 近端 MCP→PIP 方向投影到各自 URDF 屈曲平面。
- 远端方向使用标定骨长加权的 PIP-DIP-TIP 合成向量，再投影到该指屈曲平面。
- 两个目标位移分别使用对应 URDF 段长。
- 末端方向目标仍保留 DIP→TIP 的投影方向，表达指尖末节朝向；允许它与合成远端位置目标存在有限折中，但不得通过缩短机器人固定 Link 长度来拟合人体弯曲弦长。

**关键几何要求：**

当前实现对非拇指末段使用：

\[
(p_{TIP}-p_{PIP})
\frac{l_{\mathrm{robot}}}
{l_{\mathrm{PIP-DIP}}+l_{\mathrm{DIP-TIP}}}
\]

当人体 PIP-DIP-TIP 弯曲时，PIP→TIP 弦长小于两段骨长之和，结果会构造出短于机器人固定 link2→tip 长度的目标，这是不可达目标。新实现必须改为：

1. 用两段标定骨长加权当前单位骨段方向构造合成方向；
2. 投影到 XHand 对应屈曲平面；
3. 归一化；
4. 乘以 URDF 的固定机器人段长。

对所有目标段都执行相同的长度不变量检查：目标段位移模长应等于对应 URDF 段长（仅允许浮点容差）。

### 3.4 Pinch Target

保留显式 Pairwise Pinch Loss，但改进目标距离来源：

\[
d_i^* =
\|p_{\mathrm{thumb,tip}}^*
-
p_{i,\mathrm{tip}}^*\|
\]

其中两端指尖目标均已按 XHand 的实际基座布局、基部可动范围、局部屈曲平面和 URDF 段长构造。不要再使用单一 palm-width scale 直接缩放原始人体指尖距离。

- 保留四组独立的 thumb-index/middle/ring/pinky 距离。
- 使用现有 pinch_start_m / pinch_full_m 和连续 smoothstep 距离激活。
- 激活只依赖机器人几何适配后的目标距离；移除重复的 60–90° 末段夹角门控。末端方向已经由 Direction Loss 表达。
- 保留当前 pinch loss 形式和默认权重。
- 第一版不压缩目标接触间隙，不设定零距离，不引入触觉/物体接触模型。当前项仍是几何对指代理，不称为 contact-aware。

### 3.5 优化目标

第一版统一保留四项：

\[
L =
L_{\mathrm{position}}
+
L_{\mathrm{direction}}
+
L_{\mathrm{pinch}}
+
L_{\mathrm{temporal}}
\]

- Position：只约束机器人可表达的目标 Frame；Thumb/Index 的可动基部可参与，Middle/Ring/Pinky 固定基部不参与。
- Direction：保留近端和末端方向，均使用投影后的可实现方向。
- Pinch：显式匹配机器人几何适配后的四组指尖距离。
- Temporal：继续惩罚当前候选相对已提交参考的关节变化。

首版沿用现有权重、Huber 尺度、阻尼、滤波、关节速率、求解次数与时间预算，不用 pnp 数据调权。只有当几何映射完成且离线结果表明确有必要时，才单独提出后续消融；本任务不实现 Adaptive Weighting、Articulation Prior 或额外关节耦合。

## 4. 求解与控制边界

以下机制保持不变：

- 单一 12-DoF NLopt SLSQP；
- Pinocchio FK 和批量 Jacobian；
- 手写解析梯度；
- 上一条正常提交的命令作为 warm-start、阻尼中心和滤波参考；
- URDF 限位与当前操作限位；
- 现有 50 次评估、25 ms 预算；
- dt-aware EMA 与 4 rad/s 每关节命令变化率约束；
- 仅在 Robot.send_action 正常返回后推进提交参考；
- 现有几何拒绝、RoundoffLimited 和真实异常处理语义。

不改变 XHand SDK 顺序、RobotCommand、遥操作调度、机械臂 IK 或硬件发送链路。不要把提交命令速率约束描述成真实电机速度保证。

## 5. 本地 pnp_apple_banana 数据验证

必须尝试使用用户本机仓库的 Raw episodes：

**优先路径：** 仓库根目录下的 episodes/pnp_apple_banana/。

注意：assets/pointcloud_template/object/pnp_apple_banana 是苹果、香蕉和碗的点云模板资产，不是 Raw episodes，不能拿它替代示教数据。

先只读检查目录和各 episode 的 data.h5。已知历史文档中有五段包含手部 landmarks 的数据：

- episode_20261101_165909：403 行；
- episode_20261101_173701：403 行；
- episode_20261101_173916：468 行；
- episode_20261101_174250：209 行；
- episode_20261101_174738：397 行。

合计 1,880 行。数据若已更新，以本地实际文件和有效字段为准；缺少 landmarks 的 episode 不得补造或算作算法拒绝。

**只读使用字段：**

- HDF5 中的 vr_landmarks；
- timing/rows 中的 control_ns、observation_ns；
- hand_qpos；
- action_hand_joint_target；
- meta 中已有的输入坐标/标定信息。

具体字段和形状以 Raw schema 为准。不要解码 RGB/Depth 视频，不需要点云模板，不要把 tactile 数值当成接触真值。

**标定与时序：**

1. 优先使用 episode 或本地已有的真实历史标定信息（若存在，先核实其来源和适用范围）。
2. 若没有可用的历史骨长标定，只能用每段最前面的 25 个有效样本按当前运行流程标定，并从后续帧开始评价；不得用整段数据的中位数作为标定值。
3. 若前 25 帧明显处于运动中或 MAD 不通过，记录该段无法可靠标定，不得通过使用未来帧或整段统计来隐藏问题。
4. 离线重放按时间顺序逐帧执行，并在每帧模拟正常 commit，使 temporal reference 与线上一致。
5. 使用有效的 control_ns 计算 dt；若不可用，再使用 observation_ns 并明确说明。禁止用 Python 计算耗时作为控制 dt。
6. 所有方法使用相同的起始 qpos、时间戳、关节限位、求解预算和输出速率约束。

**对照方法：**

- 修改前保存当前实现的离线基线结果，基线脚本/中间结果放在仓库外临时目录。
- 用同一输入依次运行 baseline 与新算法。
- 历史 action_hand_joint_target 仅是当时的命令记录，不是真值；hand_qpos 是实测状态，也不是理想重定向标签。
- 按已有文档的分工，173701 可用于开发检查，174738 用于固定参数复核，其余有效 episode 用于补充覆盖；不要根据全部 episode 反复调参。
- 174250 历史记录存在 hardware_fault 标记，仍可用于有效帧的离线几何重放，但不能作为成功真机操作证据。

**至少报告：**

- 有效/拒绝帧数及原因；
- 五指指尖位置误差；
- 近端与末端方向误差；
- 四组 Pairwise Tip Distance 误差；
- 关节限位触发比例、最大命令变化率；
- SLSQP 评估次数；
- retarget 总耗时 P50/P95；
- baseline 与新方案逐段对照。

所有指标均应使用明确的共同定义。不能只报告各自目标函数下降，也不能把离线重放解释成真实接触、抓取成功或硬件执行效果。若本机 episodes 目录不存在，明确报告未能访问，不得用 pointcloud assets 或虚构数据代替。

## 6. 自动化测试与验收

新增有针对性的手部重定向测试，优先放在 tests/test_hand_retargeting.py；不新增通用测试框架。

至少覆盖：

1. URDF joint name 到 SDK qpos 顺序正确。
2. 五指目标 Frame、段长、基部轴和屈曲轴来自当前 XHand URDF。
3. 各指屈曲轴平面与 URDF 运动学一致；Thumb/Index 基部转角按实际 URDF 限位截断。
4. 固定基座的 Middle/Ring/Pinky 不会被人体 MCP Spread 牵引；Index 有限 Spread 能进入目标构造。
5. 所有机器人目标段长度与 URDF 对应段长一致，尤其覆盖 PIP-DIP-TIP 合并后的 distal target。
6. 目标方向投影后位于相应可实现平面，投影退化时沿用现有几何拒绝语义。
7. Pinch Target 等于映射后机器人指尖目标之间的距离，激活随距离连续变化。
8. Position、Direction、Pinch 和总目标的解析梯度分别通过中心有限差分检查。
9. 输出始终满足操作关节限位和现有 dt-aware 速率约束。
10. 失败候选不推进参考，成功提交后的 commit 行为与当前实现一致。

验收要求：

- 新目标不再包含短于固定机器人 Link 的不可达位置目标。
- 不可表达的 MCP Spread 不再对 Middle/Ring/Pinky 产生持续牵引。
- Thumb/Index 基部目标符合各自 URDF 轴和限位。
- 解析梯度数值检查通过。
- pnp_apple_banana 可用数据完成只读离线比较，或明确报告数据不可用及原因。
- 不以牺牲指尖/对指几何质量换取单纯更低的求解耗时；同环境下 P95 不应出现明显回归，仍满足现有 25 ms 预算。
- 不触碰真实硬件，不声称通过真机验收。

## 7. 实施约束与交付

1. 开始前检查最新 main、工作区改动和 AGENTS.md；保留已有用户修改，不覆盖工作区。
2. 先保存 baseline，再修改代码。
3. 优先在 landmarks.py、objective.py、solver.py 和现有 HandFK 上完成必要修改；同步更新 tests 与 docs/hand_retargeting.md。
4. 不修改无关模块，不增加依赖，不增加一套长期实验脚本或配置系统。
5. 不修改现有默认权重和控制参数，除非测试证明旧值在新目标定义下存在明确问题；如确需修改，先给出证据和影响。
6. 运行相关单元测试、解析梯度检查和 pnp 离线回放；检查完整 diff、命名和文档一致性。
7. 最终报告必须区分：已完成的代码验证、离线数据结论、未执行的真机验证。列出实际使用的 episode、有效行数、基线/候选指标及任何限制。

**最终交付应是一套针对 XHand 的简洁机构适配重定向，而不是一个更复杂的通用重定向平台。**
