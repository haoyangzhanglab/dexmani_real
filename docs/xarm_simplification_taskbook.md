# xArm 轻量化改造任务书

状态：模式准备、反馈起点、jog 频率及上下游清理已实施，离线验证通过，尚未进行真机验收。

审查基线及实施起点：`a37d8fe11aa3d7116a85bcaa9fcc568efbb36e7e`。每次准备显式发送模式命令，不依赖异步 `api.mode` 缓存；共享函数处理实测起点的 float32 舍入；jog 频率只按有效采样周期准入。公共目标频率校验也复用 `sampling_clock_ns()`，无法转换为有限正整数周期的频率报 `ValueError`，实际发送手部目标仍限制 83 Hz。

沿 driver、Owner、共享通道、Robot、运行入口和脚本检查生命周期；沿轨迹、Policy、Mapper、observation 和 Raw 检查数据含义；沿 jog、Teleop、Policy、replay、录制和导出检查采样工具调用。未发现相关 SHA-256、旧格式 reader 或兼容迁移分支。删除非法反馈测试的重复准入断言，保留真实 Ruckig、Policy 一致性及外部目标严格性覆盖。

保留独立 arm HOME 等待时的手部真实反馈与故障服务，不写手部目标，继续承担已有手部停止责任；idle 按原 TTL 决定 hold/passive。静态碰撞参数和 TCP load 仅在本连接会话首次完整准备成功后复用；每次仍设置对应模式加速度及最终 State 0。hand 禁用时跳过无关预检，EEF/jog 使用固定手姿态时仍校验其范围。

验证环境：现有 `/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -B`。针对性回归后，完整 `-m pytest -p no:cacheprovider tests -q`，782 项通过，包含真实 Ruckig 0.17.3 上下边界初始化和步进。全项目 Ruff F（`--no-cache`）、26 个相关实现及脚本的 Mypy（`--check-untyped-defs --no-incremental --cache-dir=/dev/null --follow-imports=silent --ignore-missing-imports`）、四个运行脚本的 `--help`、Policy `--print-config` 和 `git diff --check` 通过。

将 Mypy 扩展到采样工具的 recorder、Raw、export 三个调用模块时，`data/raw.py` 有 7 项既有资源可空及赋值类型诊断；在 `/tmp` 中取 HEAD 源码，使用相同参数逐条对照，内容和行号完全一致，无新增。这部分扩展检查未通过，未添加类型忽略或额外运行时兜底。

使用现有 setuptools 在 `/tmp` 中构建 wheel，80 个 Python 文件与 3 份标定 JSON 均与当前源码逐字节一致。构建不证明可选依赖或设备环境可用；未连接设备，直接 HOME 路径/加速度、实际 100 Hz、运动跟随与物理停止仍未验证。

日期：2026-11-04。以下保留当前有效取舍，后置事项未实施。

本文决定本轮 xArm 改造的目标和取舍，替代历史简化任务书中与本文冲突的 xArm 约定。其他模块以当前源码、最新任务要求及 [XHand 任务书](xhand_simplification_taskbook.md) 为准。历史测试、README 和旧任务书中的“必须”均可随本轮明确改变的行为修订，不能成为保留冗余机制的理由。

## 1. 目标与范围

让个人研究者能直接理解并修改：连接、准备、更新目标、停止、回位、关闭。减少隐含依赖、重复确认和无实际调用者的协议防御；保留运动算法、真实设备边界和数据含义。

成功标准：

- 一条 Mode 1 执行链：一个 SDK Owner 进程、一个最新目标槽、一个 Ruckig 轨迹对象。
- 普通关节执行和 arm HOME 不依赖桌面标定、碰撞网格或 XHand 几何姿态。
- 连接、运动准备和回位职责清楚；无在途设备请求的普通取消可以回待命。
- 停止后旧目标不会重新执行；错误不会因清理成功而被抹掉。
- 当前 Teleop、jog、Policy 的 sync/async/RTC、replay 及录制导出仍有明确可用的入口。
- 减少需要维护的状态与分支，不以文件数或代码行数作为验收指标。

范围以 `robot/xarm7/` 为中心，必要时修改 `robot/system.py`、`runtime/home.py`、动作映射和 Teleop/Policy 调用方。原地收敛实现；不做目录大搬迁，不建立通用机器人框架，不新增兼容旧 API 的外壳。

最新 XHand 提交已取消启动自动 HOME/清零，调整了准备顺序、首帧限速、CRC 与触觉处理。本轮不得恢复旧行为。writer 前取得所需初值、writer 后取得新关节反馈、streaming 准备后的最终观测各有实际用途，不得因看起来重复就合并掉。

## 2. 最终取舍

| 对象 | 本轮决定 | 保留的语义或代价 |
| --- | --- | --- |
| Mode 1 + Ruckig | 保留为唯一流式执行实现 | 保持已有速度、加速度、jerk 和 hold 制动行为 |
| Owner 进程、最新目标邮箱、快照短锁 | 保留 | 隔离 SDK I/O，避免旧动作排队；不改为普通 FIFO |
| `reply_pending` | 删除 | 当前只有一个父控制流程顺序调用；在途异常仍关闭 Owner，下一次请求已被 closing 拦截 |
| 失败退出时 `ready.set()` | 删除 | ready 只表示连接初始化成功；失败由既有 fault、进程状态及取消检查发现 |
| `Robot.stop()` 重复 `arm.request_stop()` | 删除重复调用及其异常编排 | 函数开头已经设置同一个共享停止 Event |
| 连接时完整停稳并进入 Mode 1，prepare 再重复 | 合并到唯一运动准备入口 | 连接只读；静态参数仅在本 SDK 会话首次运动准备中配置，运动前仍确认新鲜实测停稳 |
| arm HOME 的 planner、桌面间隙和任务 workspace 门禁 | 从基础回位链彻底移除 | 改为操作者确认空间后的直接关节回位，失去原离散几何预检 |
| HOME 的空路径/两端点数组、`strict=True` | 删除 | 一个 HOME 请求和一个有界到位判断；已经到位时可省略移动调用 |
| arm HOME 强制先 hand HOME | 解除底层耦合 | 整机回位可由运行入口显式组合，独立 arm HOME 不改变手姿态 |
| 在线 joint 模式强制加载组合碰撞模型 | 移除 | 基础 joint 映射不再执行 EEF workspace 或端点碰撞筛选；EEF/jog 保留所用几何能力 |
| `ArmTrajectory.step()/commit()` | 合并轨迹推进接口 | 未发送候选或 SDK 失败后丢弃该轨迹，不新增重试/回滚路径 |
| 完整轨迹位置极值检查 | 保留现有实现 | 下一采样点合法不等于能在限位内完成制动 |
| 目标期限、反馈时间、最终发送检查 | 保留，去除同阶段同事实的重复 | 计算后的年龄检查与输入检查有不同职责 |
| SDK 返回类型穷举 | 简化 | 保留非零返回码、反馈 shape/finite、设备轴数等实际边界 |
| 无在途请求的用户取消一律退出程序 | 放宽 | 停止和资源收尾成功后回待命；不自动重新激活 |
| 在途取消后恢复同一 Owner | 本轮不实现 | 避免重新建设晚到回复和恢复协议；结果不明时结束设备会话 |
| IK 搜索、Mode 6、报告反馈替换 | 后置实验 | 不把未经验证的算法降级或新增后端塞进结构简化 |
| Teleop 连续 hold 和 Raw 新命令类型 | 本轮不新增 | 普通目标拒绝仍结束本段并允许重新 B；jog 已有 hold 保留 |
| 自动强杀/重启 SDK Owner | 本轮不新增 | 明确 close 等待有界不等于脚本退出或物理停止有界 |

## 3. 目标结构与必要边界

沿用现有文件分工即可：`driver.py` 负责 SDK；`worker.py` 负责执行循环和低频运动操作；`process.py` 负责父端调用；`protocol.py` 负责小型共享通道；`trajectory.py` 负责 Ruckig。文件和类的数量本身不是问题。

上层只需清楚地使用连接、准备、提交目标、停止、arm HOME 和关闭这些操作。SDK 仍由 Owner 独占；几何计算、模型推理、录制和手部 I/O 不进入伺服循环。保留普通 Python 函数和现有数据结构，不新增 `MotionBackend`、RPC 框架、全局状态枚举或诊断总线。

以下事实继续分别表达，不压成一个布尔值：停止请求、Owner 完成停止处理、关闭请求、运行故障。`stopped` 不表示实测物理停稳；`is_connected` 不承诺已经完成运动准备。用清楚命名和调用契约表达，无须新增一套 Connected/Prepared/Streaming 状态系统。

保留一个本地 `cancel_epoch` 和当前必要的短锁。键盘回调仍并发撤销操作，本轮不重写输入线程模型，不能孤立删锁。保留恢复前停止确认、清旧目标、在短临界区检查取消后才清 stop 的顺序。

## 4. 实施任务

### A. 删除确定冗余，收敛错误传播

主要文件：`robot/xarm7/process.py`、`worker.py`、`protocol.py`、`robot/system.py`。

1. 删除 `reply_pending` 字段、赋值、guard 及依赖它的实现断言。明确生命周期请求由一个父控制流程顺序调用；不增加并发操作能力。
2. 删除 Worker `finally` 中的 `ready.set()`。保留正常初始化完成时的设置，以及父端不超过 50 ms 的故障/退出/取消检查。
3. `Robot.stop()` 开头设置共享 stop 后，不再调用只会设置同一 Event 的 `arm.request_stop()`；保留手部停止和 arm 停止结果等待。低层 `ArmProcess.close()` 独立使用时仍须请求停止。
4. 将 Owner 错误收敛为一个首错传播位置，后续停止/断连失败保留完整日志；没有前置错误时，清理失败必须成为主错误。可由此删除第二块 `close_error` 共享字符串及重复读取/拼装，不丢失停止失败或清理失败导致的失败结果。
5. 保留逐个尝试资源收尾的语义，一个设备关闭失败不能阻止另一个设备关闭。不建立通用错误收集框架，不为每个内部函数复制 `KeyboardInterrupt` 分支；中断由确有停止/清理职责的边界处理。
6. 主循环显式处理父管道 EOF，进入停止和关闭流程；不要把正常的父端消失统一包装成“控制器错误”。保留 stopped 状态下的有界 `pipe.poll(delay)`，它避免已设置的 stop Event 导致空转。

验收：启动失败不能发送目标；失败不再设置成功 ready；停止请求仍到达同一 Owner；晚到故障可被父端发现；清理后错误不会伪装为成功。

### B. 连接和运动准备只各做一件事

主要文件：`robot/xarm7/driver.py`、`worker.py`、`process.py` 及运行入口。

- 连接阶段完成 SDK 创建、轴数/控制器错误检查及 Owner 的一次有效状态读取，直接连接保留只读用途。静态碰撞灵敏度和 TCP 负载放在本 SDK 会话首次实际运动准备；完整准备及最终取消检查成功后才记录完成，新连接清零该事实。删除为了连接而执行的完整 HOME_DWELL 停稳流程，以及马上又会被 prepare 撤销的 Mode 1 准备。
- 进入流式控制时，集中完成停止/清目标、必要使能与 Mode 1 设置、最终 State 0、新鲜实测停稳及初始轨迹状态建立。每次准备都发送 `set_mode(mode)` 并检查应答，后续准备和停稳检查仍须成功；`api.mode` 是异步报告缓存，不能跳过命令或作为 servo 门禁，成功应答也不是实测到位。配置与状态设置按 SDK 所需顺序执行，最后一次影响可运动状态的配置之后才完成 State 0。
- 只取消对同一个准备阶段的重复证明。writer 初始化后重新取样、模式切换后确认状态、长 IK 之后重查来源年龄，不是可以机械删除的重复。
- HOME 的到位与普通开始前停稳共用小型等待逻辑；不新增等待策略类。保留总期限、反馈来源时间、位置/速度条件和连续停稳，不将 SDK 返回当作实测完成。
- `check_sdk_return_code` 使用已知 SDK 合约判断非零码；反馈可直接解包成 q/v，再检查 `(7,)` 和有限值，不维护容器类型白名单。
- 超时仍按低频操作与流式 I/O 区分。仅在完整梳理阶段切换后才将设置移到阶段入口，确保停止和断连不会继承短超时；若没有净简化，保留当前小段 try/finally。`set_timeout()` 不是网络查询，不以它作为性能瓶颈。

保留现有 100 Hz 伺服、25 Hz 反馈等默认值，本轮不凭静态分析调高频率或放宽反馈 TTL。

### C. 直接 arm HOME，去掉路径协议和隐含手部动作

主要文件：`runtime/home.py`、`robot/system.py`、`robot/xarm7/{process,protocol,worker,driver}.py`、HOME 调用方及配置。

最终基础 arm HOME 流程：

1. 校验配置 HOME 的七维有限值和关节范围；锁存当前操作的取消上下文及一次绝对期限。
2. 停止现有流式目标并清除旧参考，取得新鲜、停稳的机械臂状态。
3. 若已在到位容差内，省略移动；否则使用 Mode 0 提交一次配置 HOME 目标。
4. 两种情况都使用同一个位置、速度和连续停稳判断，受同一个总期限限制；取消仍可中断软件等待。
5. 回位结束保持停止，由后续明确开始操作进入 streaming。

删除 `HomePlanner`、`HomePathRejected`、空/两端点 `waypoints` 协议、canonical 末点重复比对、规划起点偏移检查以及 `strict=True` 专用分支。基础 HOME 不再加载桌面标定、扫描任务 workspace 或要求起点已离桌 2 cm，也不再为了规划等待手部姿态。

这是主动改变默认回位契约：操作者负责确认直接回位空间；不提供自动绕障或碰撞保证。不要保留 strict/relaxed 两套 HOME，不要在预检失败后静默换成直线回位，也不新增每次 HOME 的软件确认弹窗。

`home_arm()` 和 `home_hand()` 独立可调用。arm HOME 等待期间复用现有 `service_feedback()` 服务已连接手部的真实反馈与故障，不写手部目标、不等待触觉初值；hand 禁用时不产生手部 I/O。保留已有手部停止责任，不通过改时间戳、TTL 或 hold/passive/CRC 规则维持抓持。现有 H/R 若继续表示整机回位，在对应运行入口明确按实验既有顺序调用手部和机械臂函数，并在帮助中说明；持物回机械臂应调用 arm HOME，不能隐式改变手姿态。无需新增快捷键或工具脚本来完成这次解耦。

按引用清理仅服务 HOME 的桌面碰撞盒、软间隙和距离计算代码。`table_plane_path`、平面系数及其加载/归档若仍被点云处理使用，必须保留；不要连带删除共享标定能力。XHand 已实现的显式 HOME、有限 CRC 补发和到位判断按最新实现保留。

为 HOME 提供一个独立且单位清楚的加速度参数，默认沿用当前 HOME 实际使用值，避免同时改变运动参数。若保留全局 `set_joint_maxacc`，在 Mode 0/1 准备时分别使用相应模式参数，最后完成 State 0，避免旧全局上限残留形成隐藏耦合。不要未经验证宣称这个全局 setter 在 Mode 1 无效。

验收：缺桌面标定仍可进行基础 arm HOME；不加载几何库；独立 arm HOME 不发送手部目标；已到位与移动后的到位采用相同条件；短时不达标重新等待而不是立即永久故障；超时和真实设备错误仍失败。

### D. 让 joint 执行成为轻路径

主要文件：`control/actions.py`、`constraints.py`、`runtime/preflight.py`、Policy 配置准入及调用方。

- `ActionMapper.for_mode("joint")` 不创建 `MotionConstraints`/`CollisionModel`，不加载 URDF、mesh、Pinocchio 或 hppfcl。关节范围直接来自 ArmConfig；joint 模式不要求有效的 EEF workspace 或静态碰撞场景配置。
- 保留当前关节等价角选择和已定义的意图投影；基础设备发送仍拒绝非法命令。明确区分“Mapper 将意图投影成目标”和“设备 API 接受确定目标”，本轮不顺手改变 Policy 越界投影规则或 Raw 动作含义。
- EEF/VR/jog 继续使用其需要的 IK、FK、workspace 和端点几何检查。`MotionConstraints.rejection()` 有 jog 调用方，不是死代码。几何能力留在这些上层用途，不新建约束插件体系或通用开关矩阵。
- 无手部 I/O 不等于没有安装手。保留使用组合几何时对固定手姿态的明确约定，EEF/jog 入口用 `validate_home_pose()` 校验所需范围及姿态；hand 禁用时基础 joint 与 arm HOME 不依赖手部范围或 HOME 配置，启用手部时保留设备配置检查。
- 去掉仅由软件结构造成的“目标发布频率必须小于等于 arm.servo_hz”硬错误，保留正频率及真实设备限制。jog 周期只发送 arm 目标或 hold，以 `sampling_clock_ns()` 校验自身频率和有效周期；启用手部不限制 100 Hz jog。Teleop、Policy、replay 周期实际发送手部目标，仍限制 83 Hz；默认运行频率和独立 30 Hz 手部反馈不变。预检通过不保证实测频率。邮箱可覆盖中间目标，提交次数不等于 SDK 执行次数。

验收：仅关节控制在未安装可选几何依赖时可以初始化、校验和提交 fake 目标；同一环境的 EEF 功能缺依赖时给出真实错误；joint 路径不再做 EEF workspace/碰撞筛选，README 明确说明失去的能力。

### E. 简化轨迹接口，保留实际运动保障

主要文件：`robot/xarm7/trajectory.py`、`worker.py`、相关测试。

- 将 `step()` 计算与内部参考推进收敛到一个清楚的接口，删除公开 `commit(q)` 及“未提交恢复”解释。现有所有未发送分支都会锁存 stop 并丢弃 trajectory，SDK 失败也终止 Owner；不得同时新增跳过失败帧后复用该对象的行为。
- 保留 `np.clip(q)` 对计算舍入的归一，并使下一步内部位置与返回位置一致。此处简化不扩展外部目标的数值容差，也不删除速度、加速度、jerk 连续性。
- 保留完整轨迹与制动段的关节位置极值检查。当前固定的 Ruckig 0.17.3 官方极值实现遗漏一类恒加速度段内极值，不能直接用官方 extrema 或下一采样点检查替代。现有这段小型计算比新增边界监督框架更合适。
- 初次轨迹从新鲜实测状态建立：`arm_reference_from_feedback()` 复用严格目标校验，用内部 `1e-6 rad` 反馈舍入容差准入并生成新的、归一到配置边界的规划起点；Policy 初始姿态用同一函数准入但返回原 observation。驱动、共享状态、observation、Raw 的原始反馈和时间戳不变，外部目标仍严格使用原限位；轨迹计算的 `1e-12 rad` 舍入容差保留独立语义。保留 hold 制动和 SDK state 检查；Owner 只在准备成功后建立参考，停止或失败时清除，不增加本地模式状态。不要每周期把规划状态重置成实测位置。
- `VRController.commit()` 和手部参考时间更新另有控制含义，不能因同名而一起删除。

### F. 只放宽能低成本放宽的取消

主要文件：`teleop/loop.py`、`deployment/rollout.py`、`deployment/session.py`、`runtime/home.py`、`robot/xarm7/process.py`。

| 情形 | 最终行为 |
| --- | --- |
| 正常取消发生在设备请求发出前，已有停止已确认且无故障 | 结束本次操作、清理本次资源、回待命 |
| prepare 已完成，取消发生在 writer/观测等父端准备阶段 | 请求停止并完成收尾，回待命，不自动开始 |
| 初值等待超时但设备健康 | 沿用最新 XHand 改造：本段未开始，可稍后再 B |
| HOME/prepare 请求已经在途且取消导致结果未正常收尾 | 结束本次设备会话，不恢复旧 Owner |
| 停止未确认、设备故障、writer/必要资源失败 | 按真实错误收尾，不因“用户取消”掩盖故障 |

删除无条件把所有准备期取消转换成 `request_quit()` 的处理。先核实当前阶段和资源结果，不靠 `OperationCancelled` 类型或“尚未 send”单一事实推断可复用：之前的停止本身也可能未完成。

保留普通操作取消与 Q/Esc 的区别；后两者仍按退出/急停意图处理。迟到的初始化或 writer 完成不能清除 stop，也不能自动重启。本轮不建立取消确认消息、请求代际、自动重连或恢复管理器。

### G. 小幅诊断与同步文档

- `command_issue()` 已有停止原因，记录一次具体原因即可；不要只设置 stop 丢掉原因，不为此增加共享 Ring Buffer。
- 性能验证只需记录一小段实际发送间隔、SDK 读/写耗时。没有实测瓶颈前，不改用报告回调、不增加主动查询线程、不新增 TCP 30000。
- 当前 real 报告没有可靠的错误码来源，`get_joint_states()` 成功不能替代控制器错误检查；不要直接删除 `get_err_warn_code()`。
- 反馈 TTL 不等于读错误容忍。本轮保留 xArm 读失败的现有故障策略，不套用 XHand CRC 规则，也不添加通用重试。
- README 同步说明直接 HOME、joint 几何筛选移除、取消范围、连接与准备区别、目标覆盖语义。必要时在历史任务书顶部指向本文，不维护两份相互冲突的现行契约。
- 明确限制：非 daemon Owner 在 SDK 卡死时可能让解释器退出继续等待；`close()` 的有界 join 不能证明程序退出或设备物理停止。当前改造不新增 daemon 隐式强杀或自动 terminate/restart；如要改变回收政策，另行定义独立设备停止与进程回收的关系。

## 5. 明确后置的事项

这些不是本轮完成前置条件，不为它们新增占位接口、参数或框架：

1. **IK 搜索收缩。** previous/current 两种初值可能足够，但当前零空间搜索有实际求解收益。先用已有 EEF 轨迹比较成功率、关节跳变和耗时，再决定是否删除候选评分、SVD 和额外种子；本轮先保留算法。
2. **Mode 6。** 只在受控真机比较后决定是否替换唯一默认实现；不同时长期维护两个后端。Mode 1 下不能直接用低频策略目标替代平滑参考。
3. **报告反馈。** 只有同步查询确实影响控制时序时，才研究替换现有来源；不默认加 effort、30000 或运输误差监督。
4. **Teleop 连续 hold。** 当前普通 IK/几何拒绝结束本段后可再次 B。Raw 的 hold 无位置目标会成为 NaN，现有导出拒绝；手部参考时钟也需正确处理。未经统一数据语义设计，不只改一个 return、不把实测位置伪装成已提交动作、不建立录制/非录制两套长期控制路径。
5. **SDK Owner 强制回收。** 若将来要求脚本必定退出，再明确超时回收政策；强杀不能记作设备停止成功，也不能复用受影响的 IPC。

## 6. 验证与实施顺序

按以下四个可独立审查的批次完成，每批同步修改调用方、相关测试与文档；不留下过渡兼容层：

1. A + B + E：Owner、启动准备、错误传播与轨迹接口。
2. C：直接 arm HOME、手臂/手部解耦、删除路径协议和专属依赖。
3. D：joint 轻路径与按用途进行配置验证。
4. F + G：取消体验、具体停止原因和最终文档收敛。

复用现有项目环境和测试工具，不为文档中的命令另建依赖环境。沿实际改动运行 `test_arm_recovery.py`、`test_home_snapshot.py`、`test_control_progress.py`、`test_replay.py`；涉及手部共享入口、首帧或录制时补跑对应 `test_xhand.py`、`test_hand_retargeting.py`、`test_recording_export.py`、`test_input_resources.py`。通过后只因新改动或未解决问题扩大测试范围。

测试关注以下可观察行为，不要求旧字段、旧类或旧握手步骤存在：

| 验收点 | 必须观察到的结果 |
| --- | --- |
| 启动失败、停止、目标或反馈过期 | 不再发送运动目标，具体原因可见 |
| 停止后的晚到发布与再次准备 | 旧目标不会在新一段被执行 |
| 轨迹变化/hold/近限位制动 | 既有动态约束和极值保护仍成立；发送失败后不复用轨迹 |
| HOME 无标定、已到位、未到位和取消 | 无几何依赖；已到位与移动后同一规则；真实未到位不能成功 |
| 独立 arm HOME | 不发送手部目标、不等待手部几何姿态；共享故障仍正常传播 |
| 取消边界 | 无在途健康操作可回待命；在途不明或真实失败不被假装恢复 |
| joint 与 EEF | joint 不构造几何模型；EEF/jog 原有功能可用 |
| 最新 XHand 与录制契约 | 首帧、CRC、触觉来源、固定 Raw/Canonical 字段及动作含义不被回退 |

删除或改写锁定 `reply_pending`、失败 ready、取消必 quit、固定 join 次数、空路径 strict 和公开 commit 调用顺序的测试。保留少量真实 Ruckig 行为回归及能覆盖已知极值缺陷的针对性用例；不为减少测试数删除真实故障案例。

修正 `test_home_sdk_return_does_not_mean_measured_arrival`：原用例只给 20 ms，却要求 300 ms 停稳，成功情形也会超时。使用受控时钟和足够期限，对照“持续到位可完成”与“未到位不可完成”。Fake SDK/线程替代同步原语的测试只证明逻辑，不能声称证明了 100 Hz、原生调用可中断或物理停止上限。

工具约定沿用 AGENTS.md：pytest 使用 `-p no:cacheprovider`，Ruff 使用 `--no-cache`；需要 Mypy 时禁用增量和缓存。不生成 `.pytest_cache`、`.ruff_cache`、`.mypy_cache`，不靠 ignore、假返回或放宽真实边界让检查通过。

本轮实施默认只做离线验证。真机验证单列：直接 HOME 的实际路径和加速度、重复开始/停止、目标覆盖、实际发送间隔。必须有明确设备操作授权后执行；未测量的性能改善不写成已证实结果。

## 7. 交付要求

- 完成上述必做批次，给出实际删除的机制、保留原因和行为变化；后置事项不混入完成清单。
- 列出修改文件、实际运行的检查及结果、未验证项；不要沿用其他任务书的历史通过数量。
- 检查 diff，清理无消费者的代码、导入、配置和测试；不留下过渡 API、同功能双实现或新增通用管理层。
- 将本文状态更新为实际完成情况，保持一份当前任务说明；不再追加冗长实施流水账。

## 8. 审查依据

- [本轮修复基线 bbbbe896](https://github.com/haoyangzhanglab/dexmani_real/tree/bbbbe8965c2f0a7b5894ab3ecf824fc47dbff7ac)；[两轮历史审查基线 c4ded28](https://github.com/haoyangzhanglab/dexmani_real/tree/c4ded28b1078adefbd017f4c546b6a132a91d29b)。
- [PF-DAG 的 xArm 循环](https://github.com/XiaohanLei/PF-DAG/blob/b877129754267c3d9c564abcb19960bcf215e12f/pf_dag/robots/xarm_robot.py)：借鉴最新目标与直接循环，不照搬重复查询、自动清错或无界等待。
- [PI-R2 的窄驱动](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/robots/xarm_sdk.py)：借鉴简短接口，不照搬忽略返回码和吞错。
- [UFACTORY LeRobot 实现](https://github.com/xArm-Developer/lerobot_robot_ufactory/blob/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82/src/lerobot_robot_ufactory/robots/uf_robot/uf_robot.py)：参考控制器规划，不引入隐式首帧切模式或额外报告后端。
- [UFACTORY Servo Mode 指南](https://docs.supportarticle.ufactory.cc/support_articles/developer/ufactory-servo-mode-guide.html)：Mode 1 的平滑周期参考与 Mode 6/7 控制器规划语义。
- [Ruckig 0.17.3 极值实现](https://github.com/pantor/ruckig/blob/v0.17.3/include/ruckig/profile.hpp#L418-L438)：完整轨迹极值检查不能被当前官方 API 简单等价替代。
- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)：进程 join、非 daemon 退出等待与终止语义。
