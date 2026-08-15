# Hand Retarget 机制与代码实现深度解析

> 适用版本：基于仓库当前工作树与提交 d97c436 的静态审计  
> 审计日期：2026-08-15  
> 范围：Quest 手部关键点进入共享内存后，到 XHand 命令被工作进程接收并发送给设备之前的完整链路  
> 安全说明：本文分析与验证均为离线操作，没有连接、探测或驱动真实硬件

## 1. 结论先行

本仓库的 hand retarget 并不是一个孤立的“关键点到关节角”函数，而是一条带有明确所有权、时序约束、共享内存协议、失效语义与硬件边界的控制流水线：

~~~text
Quest HTS 右手关键点
    │
    ▼
Unity 坐标系 → FLU 坐标系
    │
    ▼
VR_FRAME_DTYPE / vr_ring
    │
    ▼
16 Hz 因果采样与新鲜度检查
    │
    ├── 无效或陈旧：命令静默、generation 失效、等待重新锚定
    │
    ▼
几何合法性检查
    │
    ▼
掌心局部坐标系 + MANO 轴约定
    │
    ▼
自适应小指长度增强
    │
    ├── TAG：两阶段数值优化，默认后端
    │
    └── DexPilot：外部 dex-retargeting 优化器，可选后端
    │
    ▼
外层 EMA → 启动 smoothstep 渐入
    │
    ▼
形状 / finite / 运行限位 / 机械限位 / 单步增量检查
    │
    ▼
ActionCandidate → SafetyGate
    │
    ▼
HAND_COMMAND_DTYPE / latest-wins hand_cmd_ring
    │
    ▼
30 Hz hand worker：generation / 过期 / 限位 / 增量再次检查
    │
    ▼
XHand driver：第三次边界检查，原样下发目标关节角
~~~

几个最重要的设计事实是：

1. **默认后端是 TAG，DexPilot 是兼容保留的可选后端。**
2. **retarget 只使用右手 21 个关键点，不使用 Quest 手腕四元数。** 手腕四元数属于机械臂位姿映射；手指姿态由关键点几何重新建立掌心坐标系。
3. **TAG 的核心是两阶段优化。** 第一阶段匹配五个指尖的腕心相对位置并保持时间连续；第二阶段只在检测到捏合时，用相对指尖 Jacobian 强化拇指与目标手指闭合。
4. **系统拒绝非法整条手命令，而不是逐关节裁剪。** 这样不会在边界处悄悄改变手型，也不会把一个求解器错误伪装成合法动作。
5. **时间状态按“已发布命令”推进，而不是按“求解器产生的候选”推进。** generation、因果采样、重锚、渐入和 worker 侧 last accepted command 一起防止暂停、相机重热、反馈故障后执行旧动作。
6. **当前碰撞检查使用上一条已发布手姿态。** 新手型和新机械臂目标没有在同一个候选构型上做联合碰撞验证；手内自碰撞对也被禁用。这是当前实现边界，而不是 TAG 优化器的隐含能力。
7. **记录链路足以分析最终动作，但尚不足以完全复现实验环境。** HDF5 保存了原始 VR 关键点、最终手命令和 TAG 的外层滤波前输出，但没有完整记录优化器状态、激活量、模型资产哈希和原生依赖版本。

## 2. 阅读范围与代码地图

### 2.1 主调用链

| 层次 | 主要文件 | 关键职责 |
|---|---|---|
| 入口与生命周期 | [examples/collect_teleop.py](../examples/collect_teleop.py) | 解析配置、生成 provenance、启动采集生命周期 |
| VR 输入进程 | [sensor/vr_receiver_process.py](../dexmani_real/sensor/vr_receiver_process.py) | 接收 Quest HTS 帧、坐标转换、写入 VR ring |
| 固定数据协议 | [utils/schema.py](../dexmani_real/utils/schema.py) | VR frame、hand command、记录样本的 NumPy dtype |
| 因果读取 | [shm/causal_reader.py](../dexmani_real/shm/causal_reader.py) | 在控制网格锚点之前选择最新可用观测 |
| 共享内存 | [shm/shared_storage.py](../dexmani_real/shm/shared_storage.py) | ring、flag、generation、生命周期 |
| teleop 主循环 | [teleop/loop.py](../dexmani_real/teleop/loop.py) | 16 Hz 动作决策、重锚、候选发布、记录 |
| retarget 包装与算法 | [teleop/hand_retarget.py](../dexmani_real/teleop/hand_retarget.py) | 几何检查、坐标变换、小指增强、TAG/DexPilot 包装 |
| TAG 优化器 | [teleop/tag_retargeting/optimizer.py](../dexmani_real/teleop/tag_retargeting/optimizer.py) | 两阶段优化、捏合激活、warm start |
| TAG 梯度 | [teleop/tag_retargeting/pin_grad.py](../dexmani_real/teleop/tag_retargeting/pin_grad.py) | Pinocchio FK、frame Jacobian、目标函数梯度 |
| 动作整形 | [teleop/hand_control.py](../dexmani_real/teleop/hand_control.py) | 手命令提取、渐入、整条命令校验 |
| 安全门 | [policy/safety.py](../dexmani_real/policy/safety.py) | ActionCandidate 的发布前结构与范围检查 |
| 手工作进程 | [robot/hand_process.py](../dexmani_real/robot/hand_process.py) | 30 Hz latest-wins 消费、命令边界复验、反馈与故障 |
| XHand 驱动 | [robot/xhand.py](../dexmani_real/robot/xhand.py) | SDK 本地所有权、最终限位/增量检查、设备发送 |
| 关节与模型常量 | [planning/constants.py](../dexmani_real/planning/constants.py) | 12 关节顺序映射、19 DoF 模型索引 |
| 手部 FK | [planning/hand_kinematics.py](../dexmani_real/planning/hand_kinematics.py) | 从测量关节角计算指尖位置 |
| 碰撞模型 | [planning/collision_model.py](../dexmani_real/planning/collision_model.py) | 机械臂与手的联合碰撞几何 |
| 记录样本 | [teleop/episode_samples.py](../dexmani_real/teleop/episode_samples.py) | 将控制网格状态和动作编码为样本 |
| 记录读写 | [recording/](../dexmani_real/recording) | HDF5 v16 序列化、校验和读取 |

### 2.2 模型与配置资产

| 资产 | 用途 |
|---|---|
| [xhand_right.urdf](../assets/robots/xhand/xhand_right.urdf) | TAG 独立右手模型 |
| [xhand_right_teleop.urdf](../assets/robots/xhand/xhand_right_teleop.urdf) | DexPilot 使用的右手模型，部分远端关节下限体现运行下限 |
| [xhand_right_dexpilot.yml](../assets/retargeting/xhand_right_dexpilot.yml) | DexPilot 关节、指尖、缩放和滤波参数 |
| [defaults.py](../dexmani_real/config/defaults.py) | hand、policy、rate 的默认值 |
| [runtime.py](../dexmani_real/config/runtime.py) | 配置解析、覆盖与跨字段约束 |
| [teleop/config.py](../dexmani_real/teleop/config.py) | teleop 运行时上下文与派生参数 |

## 3. 系统所有权：retarget 在哪里，哪里不应该做

仓库的跨进程控制面遵循一个很严格的边界：

- VR 进程只负责产生经过基础坐标转换和 finite 检查的观测。
- teleop 进程拥有控制网格、观测选择、retarget、动作候选和记录样本决策。
- hand worker 只消费固定 dtype 命令、检查能否执行并拥有 SDK。
- XHand driver 不做感知或 retarget，只验证并发送已经决定的 endpoint。
- RecorderIO 不参与动作决策，只序列化 teleop 已经选定的样本。

这意味着 hand retarget 的语义不应被拆散到设备进程或记录进程中。特别是：

- 不应让 worker 自己根据最新 VR 帧重新求手型，否则手动作与 16 Hz 的 arm action、observation id 和记录网格失去一致性。
- 不应让 RecorderIO 按到达时间选择样本，否则 HDF5 中的 VR、机械臂和手动作不再属于同一个控制决策。
- 不应让主进程或 teleop 进程持有 XHand SDK 实例；SDK 只能存在于 hand worker/driver。

## 4. 三个时钟与两种数据流语义

### 4.1 默认频率

| 名称 | 默认值 | 作用 |
|---|---:|---|
| coordinator rate | 64 Hz | teleop 协调、事件响应和健康检查的细粒度循环 |
| control / record rate | 16 Hz | arm + hand 动作决策以及 episode 对齐网格 |
| hand worker rate | 30 Hz | 消费最新命令、发送 XHand、读取反馈 |

retarget 不是在每个 64 Hz coordinator tick 上都产生一条记录动作，而是在 16 Hz 的控制网格上运行。30 Hz hand worker 也不会对 16 Hz 端点进行应用层插值；它读取并执行最新有效目标。

### 4.2 VR ring：历史可验证、因果读取

VR ring 保存多个固定 dtype 帧。控制网格通过因果 reader 选择满足以下关系的最新帧：

$$
0 < t_{\mathrm{recv}} \le t_{\mathrm{publish}} \le t_{\mathrm{grid}}
$$

这里的时间是本机单调时钟。这个约束避免把网格锚点之后才发布的帧“倒灌”进较早动作，也避免用远端设备的墙钟作为实时控制依据。

### 4.3 hand command ring：latest-wins

手命令 ring 的语义不同：

- producer 以 16 Hz 发布动作端点；
- consumer 以 30 Hz 读取；
- worker 只关心尚未处理的最新序号；
- 中间命令可以被跳过；
- 最终增量仍相对于 driver 最近一次接受的命令检查。

这符合手部 servo 的实时目标语义：过时动作没有排队执行的价值。但它也意味着单步增量上限必须在 teleop 和 worker 两侧同时成立，否则 consumer 跳帧后可能看到比 producer 相邻帧更大的跨度。

### 4.4 generation：跨状态边界作废旧命令

每条手命令带有 run_generation。begin、pause、home、反馈故障、相机重热等边界会推进 generation。worker 在命令跨过设备边界之前拒绝旧 generation，即使命令仍留在 ring 中。

generation 解决的是“逻辑时代”问题，expires_at 解决的是“时间寿命”问题，两者不可互相替代：

- generation 相同但延迟太久的命令会过期；
- 时间戳仍新但属于暂停前 generation 的命令也会被拒绝。

## 5. 从 Quest HTS 到 VR_FRAME_DTYPE

### 5.1 当前控制契约只使用右手

VR receiver 请求双手数据，以便同时获得头部信息，但正常手动作契约只消费右手 HandFrame：

- 左手 frame 被明确跳过；
- teleop 初始化的 retargeter 也固定为 right；
- schema 保留 side 字段，但当前控制逻辑不是双手通用实现；
- 实现对非 left 的未知 side 没有在 producer 边界一律拒绝，因此这里描述的是预期契约，不是对异常输入的完备证明。

因此，“支持 21 点手骨架”不等于“已支持左右手切换”。左手需要镜像/轴约定、URDF、关节顺序、配置和设备链路一起扩展。

### 5.2 Unity 到 FLU 的转换

VR producer 在写共享内存前完成 Unity 左手坐标约定到 FLU 的转换。手腕位置、手腕四元数和 21 个 landmarks 都在 producer 边界完成转换并做形状/finite 检查。

注意两类姿态信息的用途：

- 手腕位置与四元数：供机械臂 teleop 映射使用；
- 21 个 landmarks：供手部 retarget 使用。

hand retarget 不直接使用 Quest 提供的手腕四元数。它从 wrist、index MCP、middle MCP 等关键点重建掌心基，从而减小上游手腕 orientation 约定对手指求解的耦合。

### 5.3 VR schema

VR_FRAME_DTYPE 的 hand 相关主要字段是：

- wrist position：3 维；
- wrist quaternion：4 维；
- landmarks：21 × 3；
- local receive / publish 单调时间；
- frame sequence、side 等元数据。

schema 没有逐关键点 confidence、tracked bit 或遮挡标记。因此当前 hand retarget 的输入质量判断只能依赖几何一致性和整帧新鲜度，无法区分“数值有限但追踪置信度很低”的单个关节。

## 6. 输入几何合法性检查

入口函数 validate_landmarks 先于掌心变换、pinkie adaptation 和优化器调用。它检查：

1. 数组形状必须严格为 21 × 3；
2. 所有元素必须 finite；
3. wrist 到 index MCP 的距离至少 1 cm；
4. wrist 到 pinky MCP 的距离至少 1 cm；
5. index/pinky 两条掌向量夹角的正弦至少 0.1；
6. 20 条连续骨段中最短一条至少 2 mm。

令：

$$
\mathbf{v}_i = \mathbf{p}_5-\mathbf{p}_0,\qquad
\mathbf{v}_p = \mathbf{p}_{17}-\mathbf{p}_0
$$

掌面退化指标为：

$$
s_{\mathrm{palm}} =
\frac{\lVert \mathbf{v}_i \times \mathbf{v}_p\rVert}
{\lVert \mathbf{v}_i\rVert\lVert \mathbf{v}_p\rVert}
$$

只有 $s_{\mathrm{palm}}\ge 0.1$ 才继续处理。

### 6.1 为什么必须在优化前拒绝

这个顺序有两个目的：

- 防止 SVD 掌心基和目标向量在退化输入下产生任意方向；
- 防止失败帧污染优化器 warm start、捏合激活 EMA 或输出 EMA。

离线验证确认：全零 landmarks 会被拒绝，TAG 与 DexPilot 的上一目标状态不会因该失败输入推进。

### 6.2 当前检查没有覆盖的情况

以下不是已证实 bug，而是输入契约的边界：

- 没有最大骨长或整体手掌尺度上限；
- 没有跨帧速度、瞬移或冻结检测；
- 没有显式 chirality 检查；
- 没有逐关键点追踪置信度；
- 掌心退化 gate 使用 index/pinky，实际 SVD 基使用 wrist/index/middle；没有直接检查 SVD 的次奇异值或 condition number；
- producer 对未知 side 的编码能力与 teleop 固定右手的意图之间，还可以采用更严格的枚举拒绝。

## 7. 掌心局部坐标系与 MANO 轴约定

### 7.1 掌心基估计

_estimate_palm_frame 使用：

- wrist：landmark 0；
- index MCP：landmark 5；
- middle MCP：landmark 9。

实现构造：

$$
\mathbf{x}_0 = \mathbf{p}_0-\mathbf{p}_9
$$

然后对三个点中心化后的矩阵做 SVD，取掌面法向，再用 Gram–Schmidt 将法向与 x 轴正交化，最后：

$$
\mathbf{z} = \mathbf{x}\times\mathbf{n}
$$

通过 $\mathbf{z}\cdot(\mathbf{p}_5-\mathbf{p}_9)$ 的符号统一翻转方向，返回列向量基：

$$
\mathbf{R}_{\mathrm{wrist}} =
\begin{bmatrix}\mathbf{x}&\mathbf{n}&\mathbf{z}\end{bmatrix}
$$

一个容易误读的细节：源码注释可被理解为“wrist 到 middle”的方向，但实际向量是 wrist 减 middle，即从 middle 指向 wrist。分析或重写时应以表达式为准。

### 7.2 右手 operator 到 MANO

landmarks 采用行向量变换：

$$
\mathbf{P}_{\mathrm{mano}} =
\mathbf{P}_{\mathrm{FLU}}\,
\mathbf{R}_{\mathrm{wrist}}\,
\mathbf{R}_{\mathrm{operator\to mano}}
$$

其中：

$$
\mathbf{R}_{\mathrm{operator\to mano}} =
\begin{bmatrix}
0&0&-1\\
-1&0&0\\
0&1&0
\end{bmatrix}
$$

对行向量 $[x,y,z]$，结果是 $[-y,z,-x]$。该矩阵是 determinant 为 +1 的正交旋转，不是镜像反射。Unity 到 FLU 的 chirality 处理已在 VR producer 中完成，retarget 层不应再做一次反射。

### 7.3 平移如何消失

- TAG 显式使用 $\mathbf{p}_{tip}-\mathbf{p}_{wrist}$；
- DexPilot 构造关键点对的差向量。

所以两种后端都对全局平移不敏感。它们优化的是相对于手腕的手型，而不是让 XHand 跟随人手在空间中的位置；空间位置由机械臂 teleop 链处理。

## 8. 自适应小指增强

Quest 侧小指的有效长度和跟踪稳定性容易使机器人小指闭合不足。仓库在两个后端共用的预处理阶段做动态增强。

### 8.1 动态比例

使用 pinky MCP 到 TIP 的伸展距离：

$$
d=\lVert\mathbf{p}_{20}-\mathbf{p}_{17}\rVert
$$

先计算：

$$
r=\operatorname{clip}
\left(\frac{d-0.03}{0.10-0.03},0,1\right)
$$

再得到比例：

$$
s=1.2+(2.2-1.2)r
$$

也就是：

- 小指蜷缩、MCP–TIP 距离接近或低于 3 cm 时，比例接近 1.2；
- 小指伸展、距离达到 10 cm 时，比例达到 2.2；
- 中间状态线性插值。

### 8.2 逐段重建

实现先复制原始 landmarks，然后按照原始骨段向量逐段重建：

$$
\mathbf{p}'_{18}=\mathbf{p}_{17}
+s(\mathbf{p}_{18}-\mathbf{p}_{17})
$$

$$
\mathbf{p}'_{19}=\mathbf{p}'_{18}
+s(\mathbf{p}_{19}-\mathbf{p}_{18})
$$

$$
\mathbf{p}'_{20}=\mathbf{p}'_{19}
+s(\mathbf{p}_{20}-\mathbf{p}_{19})
$$

这样每一段都使用未修改的原始方向，避免把已经移动的父节点混入下一段差分。输入数组本身不会被原地修改。

### 8.3 对两个后端的实际影响

当前 TAG 和 DexPilot 后续都只抽取 wrist 与五个 fingertips。因此 PIP、DIP 重建的直接意义，是为了得到一致的最终 pinky TIP；优化器并没有匹配中间骨节。

TAG 的静态 pinky robot/human length ratio 被设为 1.0，避免在动态 1.2–2.2 之外再叠加一层固定小指尺度。动态增强仍然有效。

## 9. 后端选择与公共接口

运行时配置 policy.hand_retargeting_type 接受：

- tag：默认；
- dexpilot：可选兼容后端。

二者向 teleop 暴露同一类接口：

- retarget(landmarks)：从 21 × 3 关键点产生 12 维 SDK 顺序目标；
- reset(qpos)：在 begin、重锚或恢复时用测量关节角重置时间状态；
- is_initialized：依赖和模型是否可用；
- smoothing 相关属性。

公共包装层负责：

1. 几何合法性检查；
2. 掌心坐标变换；
3. operator-to-MANO 旋转；
4. pinky adaptation；
5. 后端求解；
6. 外层 EMA；
7. SDK 顺序输出。

后端内部状态的含义并不完全相同，这一点在调参和记录解释时必须注意，后文会专门比较。

## 10. TAG 后端：模型、关节顺序与边界

### 10.1 依赖与模型加载

TAGHandRetargeter 延迟导入 SciPy、Pinocchio 和 NLopt，加载 xhand_right.urdf，并为 Pinocchio 模型添加 FreeFlyer。运行时通过私有 overrides 将以下配置注入优化器：

- URDF 路径；
- 5 个 fingertip frame 名称；
- 五指 robot/human length；
- 两阶段权重；
- 捏合距离阈值与激活平滑；
- 运行关节下上限。

延迟导入允许模块静态检查不立刻初始化设备，也把原生依赖错误限制在创建所选 retargeter 时暴露。

### 10.2 FreeFlyer 为什么存在

Pinocchio 模型的 generalized configuration 包含：

- 7 维 FreeFlyer 配置；
- 12 维手关节配置。

优化时 FreeFlyer 被固定在：

- translation = 0；
- quaternion = identity，其中 q[6] = 1。

手腕到 MANO/URDF 的对齐在模型外部完成，所以优化变量只有 12 个手关节。对应 velocity Jacobian 的前 6 列属于自由基座，梯度实现取 $J[:3,6:]$ 作为手关节平移 Jacobian。

### 10.3 优化边界是两个边界的交集

TAG 不只使用 URDF joint limits。它把：

- URDF 模型界；
- runtime operational command bounds

取交集作为 NLopt 上下界。

配置加载还验证 operational bounds 必须嵌套在机械/额定范围内。这样优化器从源头尽量不产生设备策略不允许的目标；发布前 sanitizer 仍会独立复验。

### 10.4 模型顺序与 SDK 顺序

Pinocchio 模型内部 12 关节顺序与 XHand SDK 命令顺序不同。模型到 SDK 的映射为：

~~~text
SDK ← model indices:
[9, 10, 11, 0, 1, 2, 3, 4, 7, 8, 5, 6]
~~~

SDK 到模型的逆映射是 argsort：

~~~text
model ← SDK indices:
[3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2]
~~~

后者与 planning/constants.py 的 HAND_SDK_TO_URDF_IDX 一致。reset 时必须使用逆映射，否则 measured SDK qpos 会被错误地当成优化器顺序，warm start 会跳到另一种手型。

### 10.5 映射表

下表将设备友好名称、默认 home、运行范围和模型索引对齐。角度仅为便于阅读的近似值；代码中以弧度为准。

| SDK index | 关节 | home / deg | operational / deg | mechanical / deg | model index |
|---:|---|---:|---:|---:|---:|
| 0 | thumb abduction | 0.00 | [0.00, 104.97] | [0.00, 104.97] | 9 |
| 1 | thumb joint 1 | 80.66 | [-39.99, 99.98] | [-39.99, 99.98] | 10 |
| 2 | thumb joint 2 | 33.20 | [10.00, 99.98] | [0.00, 99.98] | 11 |
| 3 | index abduction | 0.00 | [-9.97, 9.97] | [-9.97, 9.97] | 0 |
| 4 | index joint 1 | 5.11 | [0.00, 109.95] | [0.00, 109.95] | 1 |
| 5 | index joint 2 | 5.00 | [5.00, 109.95] | [0.00, 109.95] | 2 |
| 6 | middle joint 1 | 6.53 | [0.00, 109.95] | [0.00, 109.95] | 3 |
| 7 | middle joint 2 | 5.00 | [5.00, 109.95] | [0.00, 109.95] | 4 |
| 8 | ring joint 1 | 6.76 | [0.00, 109.95] | [0.00, 109.95] | 7 |
| 9 | ring joint 2 | 5.00 | [5.00, 109.95] | [0.00, 109.95] | 8 |
| 10 | little joint 1 | 10.13 | [0.00, 109.95] | [0.00, 109.95] | 5 |
| 11 | little joint 2 | 5.00 | [5.00, 109.95] | [0.00, 109.95] | 6 |

## 11. TAG 后端：目标构造

### 11.1 五个指尖

TAG 使用 landmark：

~~~text
thumb 4, index 8, middle 12, ring 16, pinky 20
~~~

每个目标先减 wrist landmark 0，转入 URDF 对齐坐标，再按手指独立缩放。

### 11.2 人手到机器人尺度

默认长度：

| 手指 | robot length / m | human length / m | 比例 |
|---|---:|---:|---:|
| thumb | 0.161 | 0.130 | 1.2385 |
| index | 0.208 | 0.180 | 1.1556 |
| middle | 0.206 | 0.190 | 1.0842 |
| ring | 0.204 | 0.180 | 1.1333 |
| pinky | 0.145 | 0.145 | 1.0000 |

加上默认 global boost 1.0，目标为：

$$
\mathbf{t}_i =
b\frac{\ell_i^{robot}}{\ell_i^{human}}
\mathbf{R}_{mano\to urdf}^{T}
(\mathbf{p}_{tip,i}-\mathbf{p}_{wrist})
$$

需要注意：这是对整条 wrist-to-tip 向量的标量缩放，包括掌部横向 offset，而不是只缩放指骨长度。若未来引入更精细的人手标定，可能需要把掌宽、掌长和手指链长度分开处理。

### 11.3 指尖 frame 契约

初始化要求：

- 恰好 5 个 frame；
- 名称互不重复；
- 顺序固定为 thumb、index、middle、ring、pinky；
- 每个 frame 都必须存在于 Pinocchio 模型。

这使优化器的指尖 index、长度表和捏合关系不会因为 YAML 或 URDF 名称变化静默错位。

## 12. TAG 第一阶段：几何匹配与时间正则

### 12.1 目标函数

第一阶段求解：

$$
\mathcal{L}_1(\mathbf{q}) =
\sum_{i=0}^{4}
\lVert
\mathbf{p}_i(\mathbf{q})-\mathbf{t}_i
\rVert^2
+\lambda_s
\lVert\mathbf{q}-\mathbf{q}_{prev}\rVert^2
$$

默认 smooth weight：

$$
\lambda_s=0.02
$$

第一项让机器人五个指尖接近缩放后的人手目标，第二项抑制相邻求解的关节跳变。这里 $\mathbf{q}_{prev}$ 是优化器上一次成功状态或 reset 注入的 measured pose。

### 12.2 求解器

- NLopt algorithm：LD_LBFGS；
- 最大 evaluation：80；
- absolute function tolerance：$10^{-4}$；
- 上下界：URDF 与 operational bounds 的交集；
- 初值：有界的 last qpos。

### 12.3 解析梯度

PinGrad 依次执行：

1. 把 12 维手关节写入固定 FreeFlyer 的 full q；
2. Pinocchio forward kinematics；
3. 更新 frame placement；
4. 为五个 fingertip 取得位置；
5. 取得 LOCAL_WORLD_ALIGNED frame Jacobian；
6. 丢弃 FreeFlyer 的 6 个 velocity columns；
7. 累计目标误差梯度和时间正则梯度。

梯度为：

$$
\nabla\mathcal{L}_1 =
2\sum_i J_i^T
\left(\mathbf{p}_i-\mathbf{t}_i\right)
+2\lambda_s(\mathbf{q}-\mathbf{q}_{prev})
$$

离线中心差分检查得到最大绝对梯度误差约：

$$
9.3\times10^{-11}
$$

这表明当前测试点处的解析梯度与数值梯度高度一致。

### 12.4 失败语义

若 Stage 1：

- 抛出异常；
- 输出形状不对；
- 包含非 finite；
- 违反内部可接受条件，

则本次 retarget 返回失败，teleop 使用上一条已发布手命令。失败不会把无效 q 写回 last qpos。

## 13. TAG 捏合激活

### 13.1 距离到激活量

TAG 在未按机器人手长缩放的人手目标上，计算拇指与其余四个指尖的距离 $d_i$：

$$
a_i =
\operatorname{clip}
\left(
\frac{0.030-d_i}{0.030-0.008},
0,1
\right)
$$

解释：

- 距离大于等于 30 mm：激活 0；
- 距离小于等于 8 mm：激活 1；
- 中间线性变化。

然后使用 EMA：

$$
p_i^{(t)} =
0.6p_i^{(t-1)}+0.4a_i^{(t)}
$$

拇指自身激活固定为 0。若所有 $p_i<0.01$，直接跳过第二阶段。

### 13.2 时序含义

捏合激活有记忆：

- 进入捏合时不会瞬间跳到最大权重；
- 松开后权重也会逐帧衰减；
- reset 会清空该状态。

这是一种低成本抗抖手段，但它不是显式的进入/退出双阈值状态机。

## 14. TAG 第二阶段：捏合精修

### 14.1 目标函数

Stage 2 从 Stage 1 解 $\mathbf{q}_{s1}$ 出发：

$$
\mathcal{L}_2(\mathbf{q}) =
w_a\lVert\mathbf{q}-\mathbf{q}_{s1}\rVert^2
+w_t\lVert\mathbf{q}-\mathbf{q}_{prev}\rVert^2
+\sum_{i=1}^{4}
w_p p_i^2
\lVert
\mathbf{p}_i(\mathbf{q})-\mathbf{p}_{thumb}(\mathbf{q})
\rVert^2
$$

默认权重：

| 项 | 权重 |
|---|---:|
| Stage 1 anchor $w_a$ | 1.0 |
| temporal $w_t$ | 0.8 |
| pinch $w_p$ | 2000 |

激活量平方 $p_i^2$ 让较弱接近对优化的影响更小，而高激活时捏合项快速占主导。

### 14.2 相对 Jacobian

捏合差：

$$
\mathbf{e}_i =
\mathbf{p}_i-\mathbf{p}_{thumb}
$$

对应 Jacobian：

$$
J_{\mathrm{rel},i}=J_i-J_{thumb}
$$

因此捏合梯度使用 $J_i-J_{thumb}$，而不是把两个指尖当作独立绝对目标。这个形式正好表达“让两点相互靠近”。

### 14.3 求解器与回退

- NLopt algorithm：LD_SLSQP；
- 最大 evaluation：100；
- absolute function tolerance：$10^{-6}$；
- 边界与 Stage 1 相同。

Stage 2 失败不会使整次 retarget 失败，而是退回 Stage 1 解。只有 Stage 1 都不能提供合法输出时，包装层才报告 retarget failure。

### 14.4 当前捏合语义的边界

Stage 2 的几何目标是零指尖距离：

$$
\mathbf{p}_i=\mathbf{p}_{thumb}
$$

它没有：

- 非零接触间距；
- 接触法向；
- 指腹几何；
- tactile 闭环；
- 力/力矩目标。

因此它更准确地说是“指尖点闭合强化”，不是物理接触优化。真实设备最终会受机械结构、表面几何、限位和固件保护约束。

## 15. TAG 输出、外层 EMA 与 reset

### 15.1 输出路径

成功解按以下顺序处理：

1. optimizer model order q；
2. 映射到 SDK order；
3. 保存为 last_raw_qpos；
4. 外层 EMA；
5. 返回 teleop；
6. teleop 再执行启动 ramp 与 sanitizer。

外层 EMA：

$$
\mathbf{q}_{out}^{(t)}
=\alpha\mathbf{q}_{raw}^{(t)}
+(1-\alpha)\mathbf{q}_{out}^{(t-1)}
$$

默认 $\alpha=0.5$。首帧没有历史时直接通过。

### 15.2 延迟直觉

一阶 EMA 的低频 group delay 近似为：

$$
\tau_{frames}\approx\frac{1-\alpha}{\alpha}
$$

当 $\alpha=0.5$ 时约 1 个控制帧，即 16 Hz 下约 62.5 ms。它只是低频近似，不代表所有运动频率的固定延迟。

### 15.3 reset

reset 通常在 begin、恢复或新鲜数据重锚时执行：

- 清除外层 EMA 与 raw diagnostic；
- measured SDK qpos 映射为 model order；
- 交给 optimizer reset；
- optimizer 把 warm start 裁入自身上下界；
- 清空捏合激活。

若反馈 qpos 不合法，优化器使用 bounds midpoint 作为保守初始值。正常链路尽量使用真实测量手型，避免从默认数值突然跃迁。

优化器还会统计 warm-start 裁剪信息和固定 0.01 rad 的反馈偏差判据，但这些统计当前没有作为结构化 episode 字段持续记录。

## 16. DexPilot 后端

### 16.1 配置与依赖

DexPilot 包装类是 XHandRetargeter。默认资产配置：

- type：DexPilot；
- URDF：xhand_right_teleop.urdf；
- wrist link：right_hand_link；
- 12 个关节；
- 5 个 fingertips；
- scaling factor：1.05；
- dex-retargeting 内部 low-pass alpha：0.6；
- 仓库包装层 smoothing alpha：运行配置默认覆盖为 0.5；
- projection distance：30 mm；
- escape distance：30 mm。

离线审计环境中的版本：

| 依赖 | 版本 |
|---|---|
| dex-retargeting | 0.4.6 |
| nlopt | 2.7.1 |
| pin | 2.7.0 |
| torch | 2.4.1+cu124 |

这些原生/机器人依赖由 conda real_robot 环境管理，不属于 pyproject.toml 的便携 Python 依赖集合。

### 16.2 参考向量图

外部 DexPilot 优化器为五指构造 15 条向量：

- 5 个指尖两两组合，共 10 条 inter-fingertip vector；
- wrist 到每个 fingertip，共 5 条 vector。

关键点集合仍是：

~~~text
wrist 0; fingertips 4, 8, 12, 16, 20
~~~

仓库先做与 TAG 相同的 palm/MANO 转换和 pinky adaptation，再由 target-origin 配对相减构建 ref_value。平移因此自然抵消。

### 16.3 求解目标

外部 dex-retargeting 使用基于 SLSQP、Huber 型几何误差和时间正则的目标。对进入投影距离的拇指—其他指尖关系，优化器把相应向量投影并提高权重，使闭合关系更强。

默认 project_dist 与 escape_dist 都是 30 mm，因此没有距离迟滞带；抗抖主要来自滤波、优化时间正则和上层控制时序。

### 16.4 两层滤波

DexPilot 默认有两层线性平滑：

1. dex-retargeting SeqRetargeting 内部 low-pass，$\alpha_1=0.6$；
2. 仓库包装层外部 EMA，$\alpha_2=0.5$。

低频 group delay 粗略相加：

$$
\frac{1-0.6}{0.6}+\frac{1-0.5}{0.5}
=1.667\ \mathrm{frames}
$$

16 Hz 下约 104 ms。这里还没有包含 0.5 s 启动渐入；渐入只发生在 begin/reanchor 后，并不是持续滤波延迟。

### 16.5 关节边界差异

xhand_right_teleop.urdf 相比 xhand_right.urdf，主要把几类 distal flexion 的下限写成 operational 下限：

- thumb distal：约 10 deg；
- index/middle/ring/pinky distal：约 5 deg。

外部库设置 joint limit 时会做很小的 $\pm0.001$ 扩展。仓库自己的 sanitizer 仍以 runtime operational bounds 为权威，因此外部求解器偶尔给出的微小越界结果会被整条拒绝，而不是裁剪。

### 16.6 reset

DexPilot reset：

- 重置 SeqRetargeting；
- 重置内部 low-pass；
- 清除 projected flags；
- 清除仓库外层 EMA；
- measured SDK qpos 通过逆映射转换为内部顺序；
- 更新外部优化器的 last target qpos。

### 16.7 raw 诊断语义

TAG 暴露 last_raw_qpos，所以 action_hand_joint_raw 表示“优化器输出、SDK 顺序、外层 EMA 之前”的值。

DexPilot 包装类当前没有对应 raw property，记录层会退回使用其返回值。该返回值已经经过外部 SeqRetargeting low-pass 和仓库外层 EMA。因此：

- TAG 的 raw 与 final 可以用于分析外层 EMA 和 ramp；
- DexPilot 的 raw 实际上已经是过滤结果，不能解释为相同阶段。

这是记录字段跨后端语义不完全一致的地方。

## 17. 两种后端对比

| 维度 | TAG | DexPilot |
|---|---|---|
| 默认状态 | 默认 | 可选 |
| 模型 | 仓库 Pinocchio + NLopt 实现 | 外部 dex-retargeting |
| 人手目标 | 5 条 wrist-to-tip | 10 条 tip-to-tip + 5 条 wrist-to-tip |
| 第一目标 | 五指绝对腕心相对位置 | 向量图匹配 |
| 捏合 | 独立 Stage 2，连续激活 EMA | 外部 projected vector 机制 |
| 捏合终点 | 指尖点零距离 | 投影向量目标 |
| 时间正则 | 两阶段显式 q 正则 | 外部优化器内部正则 |
| 内部滤波 | 无额外 LP | SeqRetargeting LP，默认 0.6 |
| 仓库外层 EMA | 默认 0.5 | 默认 0.5 |
| 失败回退 | Stage 2 → Stage 1；Stage 1 失败 → hold | 外部求解失败 → hold |
| raw 可观测性 | 有 pre-outer-EMA raw | 当前没有等价 raw |
| 优化边界 | URDF ∩ runtime operational | teleop URDF / 外部边界，最终由仓库 sanitizer 复验 |
| 调试可控性 | 仓库内完整实现，容易审计 | 一部分语义依赖安装版本 |

同名 low_pass_alpha 属性在两类包装上的含义也不完全相同：

- TAG：控制仓库外层 EMA；
- DexPilot：控制外部 SeqRetargeting 的内部 LP。

如果上层代码试图用同一属性统一调参，可能得到不同层级的效果。更明确的命名应区分 optimizer_filter_alpha 与 output_ema_alpha。

## 18. 启动渐入与重新锚定

### 18.1 smoothstep ramp

默认渐入时间 0.5 s。在 16 Hz 控制网格上：

$$
N=0.5\times16=8\ \mathrm{frames}
$$

第 k 帧：

$$
u=\frac{k+1}{N},\qquad
w=u^2(3-2u)
$$

输出：

$$
\mathbf{q}_{ramp}
=\mathbf{q}_{start}
+w(\mathbf{q}_{live}-\mathbf{q}_{start})
$$

最后一帧 $u=1$，准确到达当前 live target。

这里的 live target 每帧都可以变化，所以 ramp 不是从起点到某个固定终点的离线轨迹，而是一个随时间放开的混合权重。它的用途是避免 begin/reanchor 后第一条视觉目标直接造成大步长。

### 18.2 重锚序列

当 VR stale、相机重热、反馈故障或运行状态切换触发 quiescence 时，典型语义是：

~~~text
停止发布新动作
    ↓
推进 run_generation
    ↓
等待 arm / VR / hand 反馈越过边界
    ↓
以测量手 qpos 重置 retargeter 与 prev_hand_qpos
    ↓
第一帧新鲜控制网格只做重新锚定
    ↓
后续网格开始发布，执行 8 帧 hand ramp
~~~

第一帧只重锚而不立刻发布，是为了让 pose baseline、优化器时间状态和设备反馈属于同一个新鲜时代。

## 19. 发布前手命令校验

### 19.1 _compute_hand_command

retarget 层返回：

- 12 维候选和 retarget_ok = true；
- 或上一条已发布手命令的拷贝和 retarget_ok = false。

以下都导致后者：

- hand disabled；
- retargeter 不存在；
- landmarks 缺失或非法；
- optimizer 失败；
- 输出不是 12 维。

### 19.2 _sanitize_hand_command

随后严格检查：

1. shape 恰好 12；
2. 全部 finite；
3. operational bounds；
4. mechanical/rated envelope 及嵌套关系；
5. 相对 ctx.prev_hand_qpos 的最大单关节增量不超过 0.20 rad。

默认增量上限换算为控制网格尺度：

$$
0.20\times16=3.2\ \mathrm{rad/s}
\approx183.3\ \mathrm{deg/s}
$$

这只是端点差分对应的速率直觉，不是显式速度轨迹控制器。

### 19.3 为什么整条拒绝而不是 clip

逐关节 clip 会产生几个问题：

- 改变优化器想表达的整体手型；
- 可能使拇指和目标手指的相对几何更差；
- 把模型、映射或配置错误隐藏成“看似合法”的动作；
- 让记录的 raw 与设备动作之间出现难以解释的非线性修改。

因此当前策略是 reject-whole，保持上一次已发布手目标。

## 20. 手和机械臂的耦合失败矩阵

hand retarget failure、hand command invalid 和 arm IK failure 是三种不同事件。

| Arm IK | Retarget | Hand sanitizer | 发布行为 |
|---|---|---|---|
| 成功 | 成功 | 合法 | 发布 arm + 新 hand |
| 成功 | 失败 | 上一 hand 仍合法 | arm 可继续，重新发布上一 hand |
| 成功 | 成功 | 非法 | 耦合候选被拒；arm hold，不发布非法 hand |
| 失败 | 成功 | 合法 | arm hold，但允许 hand-only 动作 |
| 失败 | 失败 | 上一 hand 合法 | arm hold，hand hold |
| 任意 | 任意 | 结构/范围非法 | 非法 hand 不进入共享命令边界 |

两个容易混淆的结论：

1. **retarget_ok = false 不等于手命令结构非法。** 包装层返回上一条合法命令，所以 arm 在 IK 成功时仍可继续。
2. **arm IK 失败不必冻结手。** 只要 hand 候选独立合法，系统允许手指继续动作，同时机械臂保持。

这种策略强调子系统可降级运行，但也要求记录层准确标注 frame status，避免把 hold 当作成功重定向。

## 21. SafetyGate 与最终发布

ActionCandidate 包含：

- generation；
- observation/action id；
- arm endpoint；
- hand endpoint；
- 观测与决策时间；
- hold/valid 语义。

SafetyGate 检查候选：

- 是否结构完整；
- generation 是否匹配；
- 数值是否 finite；
- joint limits；
- 可选机械臂 workspace 等。

当前 SafetyGate 不对动作做平滑或裁剪，也不替代 hand_control 的单步增量检查。通过后，hand endpoint 被编码为固定 HAND_COMMAND_DTYPE，带有：

- qpos；
- run_generation；
- command / observation sequence；
- 产生与过期时间；
- hold 标记。

默认命令有效期约 0.5 s。

## 22. hand worker 与 XHand driver

### 22.1 worker 侧复验

30 Hz hand worker 读取最新尚未处理的命令，并在跨越 SDK 边界前再次检查：

- 固定 dtype；
- qpos shape 与 finite；
- operational、mechanical、rated bounds；
- 相对 driver last_qpos_cmd 的增量；
- generation；
- expires_at；
- 当前 safety state；
- error_state 与 estop。

这不是重复浪费，而是进程边界防御：

- producer 与 consumer 可以异步；
- latest-wins 可能跳过中间帧；
- shared memory 内容必须在消费者边界重新验证；
- worker 才知道设备最近真正接受的目标。

### 22.2 状态门控

worker 只在 ARMED/RUNNING 等允许状态执行命令。e-stop 会退出执行路径；持久发送、读取或板级错误会锁存共享 error_state。error_state 是 sticky，不应被一次成功读写自动清除。

### 22.3 driver 侧最终检查

XHand driver 再次检查：

- 12 维；
- finite；
- 关节范围；
- 相对 last accepted target 的增量。

通过后，关节 endpoint 原样交给 SDK。默认设备控制设置包括 mode 3、位置刚度和 torque maximum 等设备参数。应用层没有在这里生成插值轨迹。

只有 SDK send 成功后，last_cmd_seq / last_cmd_qpos 才反映新命令。这使后续增量判断基于“设备调用已接受”的目标，而不是基于 producer 的愿望。

## 23. 与 IK、碰撞和 tactile 的关系

### 23.1 机械臂 IK 使用哪一个手型

在机械臂候选 IK 前，teleop 调用：

~~~text
planner.set_hand_qpos(ctx.prev_hand_qpos)
~~~

也就是把上一条已发布手命令写入 19 DoF 联合碰撞模型，再评估机械臂候选。

因此当前时刻 t 的逻辑更接近：

$$
\text{collision check}
\left(
\mathbf{q}_{arm}^{candidate,t},
\mathbf{q}_{hand}^{published,t-1}
\right)
$$

而不是：

$$
\text{collision check}
\left(
\mathbf{q}_{arm}^{candidate,t},
\mathbf{q}_{hand}^{candidate,t}
\right)
$$

### 23.2 当前覆盖和未覆盖

- 联合碰撞模型包含 arm–hand 活跃 pair；
- hand–hand pair 被禁用；
- 新 arm 与新 hand endpoint 没有作为一个同步候选再次做联合碰撞；
- SafetyGate 当前不承担 transition collision；
- hand retarget 本身没有碰撞项。

所以 TAG 的 operational bounds 不是碰撞证明。对自碰、手掌贴近机械臂或新 arm/hand 同步过渡的最终保护仍依赖几何设计、保守范围、设备结构和固件。

### 23.3 tactile

XHand tactile/contact 数据会进入测量与记录链路，但当前：

- 不驱动捏合激活；
- 不修改 Stage 2 权重；
- 不触发接触后停止闭合；
- 不形成力闭环。

这使 retarget 行为完全由视觉几何和关节边界决定，易于复现；代价是“视觉上已捏合”和“物理上已接触”之间没有闭环。

## 24. 记录、分析与 replay

### 24.1 HDF5 v16 中的相关字段

固定 16 Hz 网格样本包含：

| 字段 | 含义 |
|---|---|
| vr_landmarks | 原始、网格对齐的 21 × 3 VR landmarks |
| hand_qpos | 对齐时刻的测量手关节角 |
| hand_fingertip | 从测量 qpos 经 FK 得到的世界系指尖位置 |
| action_hand_joint | 最终被 teleop 选中的手动作 |
| action_hand_joint_raw | retarget raw 诊断，后端语义有差异 |
| flag_retarget_ok | 本帧是否得到新的成功 retarget |
| flag_frame_status | ok / held / IK failure / safety reject / retarget failure |
| hand_retarget_time_ms | retarget 包装调用耗时 |

hand_retarget_time_ms 覆盖几何变换和优化器求解调用，不包括后续 ramp、sanitizer、SafetyGate、共享内存传递和设备发送。

### 24.2 raw、final 与 measured 三层

对于 TAG，可用以下三层分析：

~~~text
action_hand_joint_raw
    = optimizer output, SDK order, before outer EMA

action_hand_joint
    = after outer EMA, startup ramp and acceptance logic

hand_qpos
    = device feedback aligned to the sample grid
~~~

这能分别研究：

- 优化器本身的抖动；
- 控制整形造成的延迟；
- 设备跟踪误差。

DexPilot 的 raw 当前已包含更多滤波，不能与 TAG 直接按同一语义比较。

### 24.3 failure 标注

当 retarget 失败但 arm IK 成功时：

- action_hand_joint 通常仍是上一条手命令；
- flag_retarget_ok 为 false；
- frame status 标为 retarget failure；
- arm action可以继续。

分析代码不应只看 action 是否变化来推断求解成功，也不应只看整个 frame 是否 hold 来推断手失败。

### 24.4 replay

replay_episode.py 使用记录的最终 action_hand_joint，不会从 vr_landmarks 重新运行 retarget。这样 replay 重现的是采集时最终选中的控制 endpoint，而不是在当前软件/依赖版本下重新解释人手动作。

原始 vr_landmarks 让离线重跑后端成为可能，但那属于单独的分析工具，不是运行时 replay 语义。

## 25. 可复现性与 provenance 边界

episode 会保存 resolved config、config SHA，以及联合机器人模型、SRDF、相机/VR calibration 等资源哈希。这对控制环境追踪很重要。

但当前 collect provenance 没有完整覆盖：

- TAG 独立模型 xhand_right.urdf 的精确哈希；
- DexPilot 模型 xhand_right_teleop.urdf 的精确哈希；
- xhand_right_dexpilot.yml 的哈希；
- dex-retargeting、Pinocchio、NLopt 版本；
- TAG optimizer 的逐帧 residual、status、activation；
- 输出碰到上下界的饱和信息。

因此：

- **最终动作 replay** 是有定义的，因为 action 已被记录；
- **从 VR landmarks 完全重算相同 hand action** 还不能只依赖 episode 元数据严格保证。

## 26. 配置参数与调参影响

下面只描述机制影响，不构成直接上真机的调参建议。任何范围、增量、捏合权重或滤波变更都应先离线回放和碰撞审计，再在明确授权的安全环境中验证。

### 26.1 输出平滑

| 参数 | 增大后的典型效果 | 风险 |
|---|---|---|
| output EMA alpha | 更快跟随、延迟更小 | 高频抖动与单步增量拒绝增加 |
| DexPilot internal LP alpha | 外部优化结果更快通过 | 优化噪声更直接 |
| hand ramp duration | begin 后更柔和 | 开始阶段意图响应更慢 |

EMA alpha 越大越“快”，不是越“平滑”。该方向很容易被误解。

### 26.2 TAG Stage 1

| 参数 | 增大后的典型效果 |
|---|---|
| finger length boost | 所有目标 wrist-to-tip 向量更长 |
| per-finger robot/human ratio | 对应指尖目标更远 |
| Stage 1 smooth weight | q 更接近上一解，抖动小但跟随慢 |
| max evaluations | 更可能收敛，但最坏时延增加 |

### 26.3 TAG pinch

| 参数 | 作用 |
|---|---|
| pinch enter / far distance | 决定何时开始产生激活 |
| close distance | 决定何时达到满激活 |
| activation EMA alpha | 决定进入/退出记忆速度 |
| pinch weight | 决定 Stage 2 对零距离闭合的强度 |
| Stage 1 anchor weight | 防止 Stage 2 偏离整体手型 |
| temporal weight | 防止 Stage 2 相对上一帧跳变 |

pinch weight 与关节限位并不是独立的：权重再高，若零距离在 bounds 内不可达，解仍会停在边界附近；这时更高权重可能只增加饱和和其他手指姿态牺牲。

### 26.4 命令范围和增量

operational lower/upper bound 影响：

- TAG optimizer 的可行域；
- teleop sanitizer；
- SafetyGate；
- hand worker；
- driver。

它是跨层合同，不能只改一个位置。max command delta 同样必须检查 producer 与 consumer 两侧语义。

## 27. 当前实现的优势

### 27.1 边界清楚

算法、控制决策、共享内存、设备 worker 和记录各自有明确所有者。设备 SDK 不泄漏到 teleop，retarget 不泄漏到 worker。

### 27.2 失效默认保持

无效关键点、求解失败、范围错误和 stale source 不会生成“猜测动作”。系统优先保持上一条合法手命令，并通过 flag/status 保留失败事实。

### 27.3 端到端多层验证

同一命令在：

- retarget 输出；
- teleop sanitizer；
- SafetyGate；
- hand worker；
- driver

经历不同职责的验证。尤其是 worker 使用 last accepted device target，而不是简单相信 producer history。

### 27.4 可分析性较好

原始 VR landmarks、raw/final action、测量 qpos、指尖 FK、成功标志和耗时都进入固定网格记录，比只保存设备目标更适合定位“输入、优化、滤波还是跟踪”哪一层出了问题。

### 27.5 TAG 梯度可审计

TAG 的目标函数和解析 Jacobian 都在仓库内，离线数值差分能够验证；Stage 2 失败回退 Stage 1，也避免捏合精修把整个手型链路拖垮。

## 28. 审计观察与改进方向

以下按优先级组织。它们是基于当前实现边界的工程建议，不代表仓库已经发生对应故障。

### P1：可复现性和回归保护

1. **补充 retarget provenance。** 将实际选择后端的 URDF、YAML 和关键原生依赖版本纳入 episode metadata。
2. **建立专用离线回归检查。** 至少固定覆盖：
   - landmark 退化拒绝且不污染状态；
   - palm basis 正交性与 determinant；
   - pinky 每段同比缩放；
   - SDK/model 顺序 round-trip；
   - TAG Stage 1/Stage 2 梯度；
   - 两后端 reset 后首帧连续性；
   - bounds 和 max-delta 整条拒绝；
   - synthetic pinch activation / decay。
3. **统一 raw 字段语义。** 两个后端都暴露明确的 optimizer_raw、backend_filtered 和 final_command，或在 schema/metadata 中标出可用阶段。

### P1：输入质量

1. 若 HTS 能提供，向 schema 增加 tracked/confidence 信息，并在 producer 边界固定形状验证。
2. 增加合理的最大手掌/骨段尺度和跨帧跳变检查，防止 finite 但明显错误的骨架进入优化器。
3. 直接检查 palm SVD conditioning，或让退化 gate 与实际使用的 wrist/index/middle 三点一致。
4. 明确拒绝非右手/未知 side，而不是依赖上游正常行为。

### P2：碰撞语义

1. 明确评估新 arm candidate 与新 hand candidate 的同步联合碰撞成本。
2. 如果在控制预算内不可行，至少把“collision uses previous hand command”变成显式指标/文档合同。
3. 对手内自碰是否可以继续全禁用进行模型级审计；若保持禁用，应说明由机械耦合、限位还是固件承担约束。

任何新增碰撞检查都必须保持 teleop 所有权，不应把规划逻辑下放到 hand worker，也不应增加应用侧 arm 插值。

### P2：优化诊断

可记录或按采样率降频记录：

- NLopt result code；
- Stage 1 / Stage 2 loss；
- evaluation count；
- Stage 2 是否运行/是否回退；
- 4 个 pinch activation；
- joint-at-bound bitmask；
- warm-start clipping；
- raw-to-final delta。

这些信息能区分“输入几何错”“优化未收敛”“目标不可达”“被滤波”“被安全拒绝”。

### P2：API 语义

1. 将 TAG 和 DexPilot 的 low_pass_alpha 拆成语义明确的字段。
2. 明确 pinky dynamic scale 与 static finger ratio 的组合关系。
3. 把右手固定选择从隐式实现条件提升为验证过的配置约束，或完整实现 left-hand 资产和映射。

### P3：接触质量

若任务确实需要稳定抓取而不只是视觉模仿，可研究：

- 非零指腹接触距离；
- 指腹 frame 和接触法向；
- tactile 仅作为捏合终止/权重调节，而不是直接绕过安全范围；
- 基于离线 recorded tactile 的策略评估。

这会改变控制语义，必须先定义接触失败、传感器失效、sticky fault 和 replay 行为，不能只在 Stage 2 随手加一个力反馈项。

## 29. 扩展或修改时的垂直检查清单

### 29.1 修改关键点或坐标系

- VR SDK 原始约定；
- Unity → FLU；
- VR_FRAME_DTYPE；
- causal reader；
- palm basis；
- operator → MANO；
- 后端 target index；
- raw landmarks 记录和离线工具；
- 左右手 chirality。

### 29.2 修改关节顺序或增加关节

- runtime shape 和上下限；
- SDK order；
- URDF model order；
- TAG mapping 与 inverse；
- DexPilot YAML joint_names；
- HAND_SDK_TO_URDF_IDX；
- HAND_COMMAND_DTYPE；
- hand feedback dtype；
- collision model 19 DoF 索引；
- recorder/reader/replay schema。

如果持久化 shape 改变，不能在 HDF5 v16 中静默改义；需要协调 schema marker 和所有消费者。

### 29.3 修改滤波或 ramp

- reset 时是否清空全部后端状态；
- reanchor 首帧；
- raw 字段阶段；
- max-delta 拒绝率；
- worker latest-wins 跳帧后的有效增量；
- episode 中 measured/action 的相位差；
- replay 是否使用 final action。

### 29.4 修改捏合逻辑

- 原始目标还是缩放目标上计算激活；
- enter/close/escape 阈值；
- 激活是否有历史；
- Stage 2 fallback；
- bounds saturation；
- tactile 缺失或异常；
- 指尖 frame 几何；
- 记录字段与离线可解释性。

### 29.5 修改范围

必须从 config/defaults.py 与 runtime 验证开始，再审计：

- TAG optimizer bounds；
- DexPilot URDF/外部扩界；
- teleop sanitizer；
- SafetyGate；
- worker；
- driver；
- home pose；
- replay preflight；
- metadata。

## 30. 离线验证记录

本次审计执行了一个不创建硬件 SDK、不访问设备地址的确定性检查，覆盖：

1. operator-to-MANO 矩阵 determinant 约为 1；
2. 正交误差约为 $2.22\times10^{-16}$；
3. pinky 三个骨段获得相同动态缩放比；
4. TAG model/SDK 映射 round-trip 误差为 0；
5. PinGrad 解析梯度与中心差分最大绝对误差约 $9.26\times10^{-11}$；
6. TAG synthetic landmarks 输出 finite、shape 为 12；
7. DexPilot synthetic landmarks 输出 finite，内部 reference index shape 为 2 × 15；
8. 两个后端对退化输入拒绝且不推进 temporal last q；
9. DexPilot 当前内部 low-pass alpha 为 0.6。

检查结果：

~~~text
coordinate det 1.0000000000000002
orth_err 2.220446049250313e-16
pinky segment ratios [1.4142857142857144, 1.4142857142857144, 1.4142857142857144]
TAG mapping [9, 10, 11, 0, 1, 2, 3, 4, 7, 8, 5, 6]
roundtrip_err 0.0
PinGrad max_abs_gradient_error 9.263415555460508e-11
TAG output finite True shape (12,)
DexPilot output finite True indices (2, 15) internal_alpha 0.6
offline hand-retarget audit: PASS
~~~

这只能证明离线数学与接口契约在所选 synthetic case 下成立，不能替代：

- Quest 真实追踪质量验证；
- XHand 跟踪和温升验证；
- 实体碰撞与抓取验证；
- 人机延迟主观评估；
- 固件错误恢复验证。

这些都属于需要明确授权、清空工作区并准备硬件后的独立手工验证。

## 31. 推荐阅读顺序

如果要快速理解或修改 hand retarget，建议按以下顺序：

1. teleop/hand_retarget.py：公共预处理、两个包装器和映射；
2. teleop/tag_retargeting/optimizer.py：两阶段目标和状态；
3. teleop/tag_retargeting/pin_grad.py：FK/Jacobian 细节；
4. teleop/hand_control.py：ramp 与 reject-whole；
5. teleop/loop.py：何时求解、何时 hold、如何与 arm 耦合；
6. utils/schema.py 与 shm/causal_reader.py：数据与因果时间；
7. robot/hand_process.py 与 robot/xhand.py：设备边界；
8. teleop/episode_samples.py 与 recording/：可观测性；
9. planning/collision_model.py：当前联合碰撞覆盖。

## 32. 总结

本仓库的 hand retarget 设计重点不是追求单个优化器的最大自由度，而是把视觉手型稳定地嵌入一个可监督、可记录、可失效的实时机器人系统。

TAG 默认后端通过“五指目标匹配 + 时间正则 + 条件式捏合精修”提供了仓库内可审计的数学路径；DexPilot 通过更丰富的指尖向量图保留了成熟外部实现。两者共享关键点合法性、掌心坐标、小指补偿、外层平滑、启动渐入和严格命令边界。

当前最值得继续加强的并不是再堆一层平滑，而是：

- 把 retarget 资产和依赖纳入 provenance；
- 固化离线数学/时序回归检查；
- 统一两个后端的 raw 与滤波语义；
- 提升关键点质量观测；
- 明确同步 arm + hand 候选的碰撞合同；
- 为优化状态、捏合激活和边界饱和增加可解释指标。

这样才能在不破坏现有共享内存、控制网格和硬件所有权架构的前提下，让 hand retarget 从“可工作”进一步走向“可证明、可比较、可复现”。
