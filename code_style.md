# DexMani Real Code Style

本规范服务于一个博士生长期维护的个人真实机器人科研项目。项目的核心任务很窄：

1. 稳定、可追溯地采集少量真实机器人数据；
2. 把训练好的模型接入同一控制边界，安全地做真实机器人测试；
3. 在实验失败时，能够快速判断问题来自传感器、模型、控制、硬件还是数据。

代码不需要表现得像通用机器人平台，也不需要为未知的团队规模和硬件组合提前设计。
它应该像一本放在桌面上的参考实现：入口清楚、数据流直接、领域文件具体，隔几个月
重新打开仍能迅速找到主路径。

风格上参考 [ManiUniCon](https://github.com/Universal-Control/ManiUniCon) 的直接组织方式：
可执行入口、配置、robot、policy、sensor 和数据工具彼此可见。这里借鉴的是清晰度，
不是照搬它为多机器人准备的 Hydra、基类或动态实例化机制。

本文中“必须”和“禁止”是安全或合同约束；“优先”和“推荐”是默认选择，有更清晰的
具体理由时可以偏离，但应让理由在代码或评审中可见。

## 1. 优先级

发生取舍时，按以下顺序判断：

```text
真实机器人安全与结果正确
→ 主数据流清晰
→ 容易调试和复现实验
→ 容易修改当前实验
→ 复用与扩展
```

几个直接推论：

- 清楚的重复代码，可能优于过早的通用框架。
- 一个具体函数，可能优于 factory + registry + adapter。
- 为当前两条主路径服务，比支持假想的十种机器人更重要。
- “能运行”不够；时间、单位、坐标系、数据有效性和失败原因必须可判断。

## 2. 可读性的验收标准

第一次阅读一个功能时，应能在少量跳转内回答：

1. 从哪个入口启动？
2. 配置在哪里解析，最终值是什么？
3. observation 从哪里产生，shape、dtype、单位和坐标系是什么？
4. action 在哪里计算、校验和发布？
5. 哪个进程拥有硬件 SDK？
6. 数据写到哪里，失败时保留什么？

如果必须经过 `manager → service → controller → adapter → implementation` 才能回答，
说明抽象层次过深。主路径应尽量接近：

```text
entry point
→ resolve config
→ construct owned resources
→ read observation
→ compute proposal
→ validate command
→ publish side effect
→ record result
```

### 2.1 实现一个功能的顺序

新功能先定义数据和边界，再决定函数和类：

1. 写清输入、输出、shape、dtype、单位、frame 和失败语义；
2. 把变换、计算和校验写成纯函数；
3. 为硬件、进程、文件 transaction 或 model resource 选定唯一 owner；
4. 用窄的 lifecycle class 管理 owner，用可顺读的 function 组合 workflow；
5. 先验证纯计算和失败分支，再考虑进程、模型和硬件集成。

如果这个顺序进行不下去，通常是 boundary 或 ownership 还没有说清，而不是
缺少一个新的抽象层。

## 3. 项目结构

按稳定的领域责任组织，而不是按技术模式组织：

- `config/`：canonical defaults、运行时快照和标定读取。
- `sensor/`：设备输入、点云和时钟映射。
- `robot/`：硬件 driver、worker 与 SDK 边界。
- `planning/`：FK、IK、collision 和 path 等纯计算优先的能力。
- `teleop/`：操作者输入到 action proposal 的映射与采集决策；`teleop/control_loop/` 收拢因果实时控制算法，`teleop/retargeting/` 收拢手部 retargeting backend。
- `control/`：跨控制方式共用的 action contract、safety boundary、publication 与 homing。
- `replay/`：物理回放的加载、调度、捕获、评估与 session。
- `calibration/`：相机、桌面等标定算法与 side-effect lifecycle。
- `deployment/`：模型 observation、inference、flat Prediction IPC 和调度；`deployment/inference/` 拥有模型 adapter 与 inference child。
- `recording/`：raw episode transaction、schema、写入与读取；`recording/storage/` 拥有 persisted artifact 实现。
- `dataset/`：离线清洗和训练数据导出。
- `runtime/`：进程生命周期、supervisor 与结构化退出状态。
- `ipc/`：跨进程 typed channels、wire schema、ring buffer 与 causal read。
- `examples/`：薄入口和自包含的诊断/可视化/离线分析程序。

不要轻易新增顶层包。新模块应先回答“它属于哪个现有领域”。

`utils/` 只放真正跨领域的稳定 helper。与 point cloud、recording 或 planning 明确相关的
函数应留在各自领域，不要因为暂时不知道放哪里就创建新的 `utils.py`。

## 4. 文件与模块

一个文件应有一个主要责任，但不设机械行数上限。完整、连续的算法留在一个文件中，
通常比拆成许多一跳一跳的小文件更易读；包含多个独立 owner 或 side effect 的文件则应拆分。

推荐的文件顺序：

```python
"""一句话说明模块负责什么，以及它不负责什么。"""

from __future__ import annotations

# standard library
# third-party
# project

logger = get_logger(__name__)

# module constants
# small dataclasses / enums / protocols
# pure helpers
# primary public class or functions
# private side-effect helpers
```

这只是阅读顺序，不要求空章节或装饰性分隔线。

拆文件的理由应该是责任不同，例如 camera capture 与 camera calibration；“超过 300 行”
本身不是理由。反过来，一个函数同时做 argument parsing、IK、硬件发送和 HDF5 写入，
即使只有 80 行也应该拆。

## 5. 入口和脚本

`examples/` 应该“无聊而明显”：

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = resolve_experiment_config(yaml_path=args.config)
    return run_experiment(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
```

入口可以负责：

- 参数解析和用户确认；
- 构造或解析配置；
- 调用一个明确的 domain lifecycle；
- 打印面向操作者的结果。

入口不应拥有通用的 geometry、retarget、safety、recording 或 device protocol 逻辑。
诊断、可视化与离线分析程序可以保留在 `examples/`，但文件名和顶部 docstring 必须
说明是否连接硬件、是否打开 GUI、是否写标定。

## 6. 命名

遵守 Python 基本约定：函数、变量和模块用 `snake_case`，类用 `PascalCase`，真正的
模块常量用 `UPPER_SNAKE_CASE`，私有实现用前导 `_`。

名称优先表达领域语义：

```python
read_arm_state()
compute_target_eef_pose()
validate_joint_command()
publish_record_sample()
```

避免无法定位责任的名称：

```python
process()
handle()
update()
do_work()
Manager
Handler
Data
Utils
```

只有当局部上下文完全明确时才使用短名称。`q` 可在三行公式中表示关节向量，跨越函数
边界时应写成 `arm_qpos_rad` 或同等明确的名字。

### 6.1 单位、坐标系和时间

物理量有歧义时把单位写进名字：

```python
timeout_s
timestamp_ns
control_hz
distance_m
joint_velocity_rad_s
acceleration_deg_s2
```

坐标系写进 pose、point 和 transform 名称：

```python
points_camera
points_world
eef_pose_base
T_world_camera
T_eef_camera
```

本项目的 transform 命名采用 `T_target_source`，表示把 source frame 中的数据变换到
target frame。Quaternion 顺序不允许靠猜，名称或边界合同中写明 `wxyz` / `xyzw`。

elapsed time、deadline、heartbeat 和 freshness 使用 monotonic clock；wall clock 只用于
文件名、面向人的时间和外部时间戳。变量名用 `_monotonic_ns`、`_wall_time_ns` 等区分。

## 7. 函数和数据流

函数是主要的阅读单元。函数名、签名、返回值和失败方式共同构成合同；调用者
不应该阅读函数体才知道它是否会发送指令、修改输入或访问磁盘。

### 7.1 名称就是合同

使用稳定的动词区分语义：

- `compute_` / `transform_` / `convert_`：纯计算，不改外部状态；
- `build_`：在内存中构造新值，不打开设备或写文件；
- `read_`：从设备或 shared state 取一次值，必须说明缺失、超时和 blocking 语义；
- `load_`：把 storage/checkpoint 中的内容带入内存，让可能昂贵的 IO 或资源创建可见；
- `connect_` / `start_` / `send_` / `publish_` / `save_` / `stop_` / `close_`：有可见 side effect；
- `validate_`：不修改输入，合法时返回 `None`，不合法时抛明确异常；
- `check_`：不修改输入，当不通过是预期运行分支时返回 typed result；
- `clamp_` / `normalize_`：显式转换并返回新值，不假装成 validation；
- `is_` / `has_` / `can_`：无 side effect 的布尔判断；
- `require_`：返回必需值，不存在或状态不允许时抛异常；
- `try_`：失败是预期分支，并在返回类型中明确表达。

特别避免下面这种“验证实际上是修改”的写法：

```python
def validate_action(action: EpisodeAction) -> bool:
    action.arm_qpos_rad = np.clip(action.arm_qpos_rad, LOWER_RAD, UPPER_RAD)
    return True
```

判断和修正是两种不同的 safety policy，要拆开：

```python
def validate_arm_target(
    arm_qpos_rad: np.ndarray,
    *,
    limits: ArmJointLimits,
) -> None:
    if arm_qpos_rad.shape != limits.shape:
        raise ValueError(
            f"Expected arm target shape {limits.shape}, got {arm_qpos_rad.shape}"
        )
    if not np.all(np.isfinite(arm_qpos_rad)):
        raise ValueError("Arm target contains non-finite values")
    if np.any(
        (arm_qpos_rad < limits.lower_rad)
        | (arm_qpos_rad > limits.upper_rad)
    ):
        raise ValueError("Arm target exceeds joint limits")


def clamp_arm_target(
    arm_qpos_rad: np.ndarray,
    *,
    limits: ArmJointLimits,
) -> np.ndarray:
    if arm_qpos_rad.shape != limits.shape:
        raise ValueError(
            f"Expected arm target shape {limits.shape}, got {arm_qpos_rad.shape}"
        )
    if not np.all(np.isfinite(arm_qpos_rad)):
        raise ValueError("Arm target contains non-finite values")
    return np.clip(arm_qpos_rad, limits.lower_rad, limits.upper_rad)
```

`ArmJointLimits` 在自己的创建边界已 copy 并冻结 lower/upper array，同时保证 shape 一致、
值有限且逐元素满足 `lower_rad <= upper_rad`。函数不重复验证这个 canonical invariant，但仍验证
自己接收的输入。

### 7.2 函数签名

签名应当让主数据和策略参数分开：

```python
def resample_joint_plan(
    arm_qpos_rad: np.ndarray,
    timestamps_monotonic_ns: np.ndarray,
    *,
    target_hz: float,
    max_gap_s: float,
) -> JointPlan:
    ...
```

具体规则：

- 主要 domain input 放在前面，限制、模式和 policy modifier 优先写成 keyword-only；
- 单位、frame 和时钟语义有歧义时写进参数名；
- 多个字段始终成组出现且共享同一 invariant 时，优先引入窄 dataclass；
- 禁止 mutable default；`None` 只在真正表示“缺失、不覆盖或未启动”时使用；
- 不用 `record=False`、`blocking=False`、`safe=True` 等多个布尔值组合出不同 workflow；
- 返回类型保持稳定，不在 `None`、`False` 和实际结果对象之间切换；
- 默认不修改 caller 拥有的 array/list/dataclass；确需原地修改时在名称中写 `_inplace`。

参数列表过长时，先问这些值是否属于一个真实 domain contract；不要为了缩短签名
就传入全局 config 或一个无语义的 `context` 对象。

### 7.3 函数体与返回值

一个函数尽量只处于一个语义层级。先做廉价的 guard，用 early return 保持正常路径左对齐，
再按数据流顺序调用下一层具体函数：

```python
def control_tick(snapshot: ObservationSnapshot) -> PublishResult:
    if not snapshot.is_fresh:
        return PublishResult(published=False, reason="stale_observation")

    proposal = compute_action_proposal(snapshot)
    candidate = build_action_candidate(proposal, snapshot)
    validate_action_candidate(candidate)
    return publish_action_candidate(candidate)
```

一个函数尽量能在一屏中理解，但不设机械行数上限。不要为了缩短行数把三行连续计算
拆成无语义的 `_step_1()` 和 `_step_2()`。应抽取的是独立语义，不是视觉长度。

返回值遵守下列区分：

- 预期的运行分支，如 stale、gated 或 queue full，使用 enum 或 typed result；
- 外部边界不符合合同，如 shape、dtype、frame 或 lifecycle 错误，抛出具体异常；
- predicate 只返回 `bool` 且不记录日志；需要 reason 时返回显式结果；
- 避免“记一条 error 然后返回 `False`”，这会让调用者不知道是否已经处理失败。

### 7.4 `run()` 和高频循环

`run()` 和 `*_loop()` 允许比普通 helper 更长，因为它们需要展示 lifecycle 和时序；但主体应当
像目录一样可以顺读：

```python
def run_policy_loop(
    observation_reader: PolicyObservationReader,
    policy_runtime: PolicyRuntime,
    publish_prediction: Callable[[np.ndarray], bool],
    stop_event: Event,
    *,
    read_timeout_s: float,
) -> None:
    while not stop_event.is_set():
        observation = observation_reader.read_fresh(timeout_s=read_timeout_s)
        flat_actions = policy_runtime.predict(observation)
        if not publish_prediction(flat_actions):
            return
```

上例的 loop 只借用已 ready 的组件，不拥有它们，因此不负责启动或关闭。资源由创建它的
lifecycle function 统一启动，并在 `finally` 或等价的 `ExitStack` 中按相反顺序关闭。
一个函数不要只关闭某个
传入资源，形成“半拥有”合同。

如果 loop 内同时展开 reset、clock sync、interpolation、action scheduling、recording 和 recovery，
应按真实语义抽出 `_handle_reset_request()`、`_read_fresh_snapshot()`、
`_select_due_action()` 和 `_record_control_sample()`。不要用 `_process()` 或 `_handle()` 隐藏这些差异。
键盘、设备和 sensor callback 也应保持窄：解码事件、更新明确状态或入队后立即返回，
不在 callback 中直接做 model inference、阻塞 IO 或发送 robot command。

不要在同一函数中展开 SDK packet、坐标转换、模型推理、安全检查、HDF5 写入和 UI。
geometry、filter、validation、mapping 和 metrics 优先写成纯函数，并移出资源 owner class。

### 7.5 抽取 helper 的边界

不为一行调用增加 pass-through wrapper，除非 wrapper 增加了校验、转换、同步或 ownership。
不隐藏昂贵操作；函数名或调用位置应让 inference、collision check 和 disk IO 可见。

局部重复可以接受。当相同语义已经稳定地出现两到三次，并且修复时必须同步修改，
再抽取共享 helper。不要仅因为代码形状相似就合并不同的安全语义。

## 8. 数据结构与类型

用 dataclass 表达小型、固定的 domain contract，例如 config、observation、result、status。
跨进程固定布局使用 centralized NumPy dtype；外部模型边界可以使用明确约定的 mapping。

`@dataclass(frozen=True)` 只防止 field 被重新赋值，不会自动冻结其中的 NumPy array。
需要 immutable snapshot 时，在创建边界明确 copy 或设置 read-only；高频路径不为了
表面不可变而无条件复制大数组，但必须写清 buffer owner 和有效期。

边界数据必须能看出：

- shape，例如 arm `(7,)`、hand `(12,)`；
- dtype；
- 单位；
- coordinate frame；
- timestamp 语义；
- valid/fresh 状态；
- 缺失值是否允许。

不要让一个 `dict[str, Any]` 穿过多个核心模块。字典适合 YAML、JSON、第三方 SDK 和
模型 adapter 的边缘；进入内部稳定数据流后，应转为 typed structure 或已定义 schema。

类型标注优先覆盖 public function、process boundary、storage boundary 和容易混淆的返回值。
不要求为了 type checker 给每个显然的局部变量加注解。

## 9. 类、状态与生命周期

类用来表达“谁拥有什么，当前处于什么状态，允许做什么”。它不是整理函数的默认文件夹。

### 9.1 先决定是否需要类

| 问题 | 优先形式 |
| --- | --- |
| 无状态的 geometry、conversion、validation | module-level 纯函数 |
| 一组有名字、有类型的固定数据 | dataclass；值语义明确时优先 `frozen=True` |
| 需要缓存且缓存语义稳定 | 小型 stateful class |
| 拥有 device、thread、process、file transaction 或 model runtime | lifecycle class |
| 只是按顺序组合多个 owner，自身没有长期状态 | 显式 lifecycle function |

顶层 workflow 不必为了“统一”包成 `System` 或 `Manager`。一个带 `try/finally` 的
`run_data_collection()` 往往比无持久状态的 `DataCollectionManager` 更直接。

### 9.2 类名与单一责任

每个重要类都应能用一句话说清：

```text
<Class> owns <resource/state>, converts <input> to <output>, and does not own <adjacent resource>.
```

例如：`ArmWorker` 拥有 arm SDK 和 command loop，它消费已验证的 joint command，不拥有
model runtime。`EpisodeWriter` 拥有一次 episode transaction，它不决定当前样本是否值得记录。

优先具体名称：`SpaceMouseReader`、`PolicyRuntime`、`EpisodeWriter`、`ArmCommandLoop`。
避免 `Manager`、`Handler`、`Processor`、`System` 或 `Utils` 这类不说明 owner 的名称；除非该词
在领域内有精确、已约定的含义。

### 9.3 构造函数

`__init__` 只做廉价、确定、易回退的事：

- 保存窄而 typed 的 config 和 dependency；
- 校验本地参数，例如频率为正、shape 匹配；
- 初始化 lifecycle state、counter 和 `None` resource placeholder；
- 建立不访问外部系统的小型内存对象。

不在 `__init__` 中：

- 连接机器人、相机或 HID；
- 启动 shared-memory manager、thread、process 或 keyboard listener；
- 改变硬件 mode/state 或安装 signal handler；
- 加载大型 model/checkpoint；
- 用隐藏 fallback 吞掉 dependency 或 config 错误。

构造成功只表示“对象可以被配置”，不表示外部资源已就绪。这使得对象可以在无硬件的
单元测试中创建，也使 partial startup failure 容易清理。

### 9.4 公开表面和方法顺序

公开方法只保留直接 domain operation，并用动词说明是否有 side effect。推荐的阅读顺序是：

```text
__init__
廉价 property
connect/start/load
公开 domain operation
stop/close
私有 state transition / boundary helper
```

具体规则：

- property 必须廉价、无 IO、不抛出设备异常；`camera.frames` 不应暗中抓取一帧；
- 公开方法尽量少，不用大量 `_get_x()` / `_set_x()` 包装普通属性；
- 公开方法在边界检查 lifecycle precondition，不让“未 connect 就 send”变成 SDK 内部偶发错误；
- 不直接暴露内部可变 buffer/list；返回 immutable snapshot、明确的 copy 或受约束的 read-only view；
- 只在跨多次调用确实需要保留时才写 `self._state`，不把单次调用的临时值堆到对象上；
- 只访问参数的 helper 通常是 module-level 纯函数，不是为了归类而写的 `@staticmethod`；
- `@classmethod` 只用于有真实语义的备选构造，例如 `Calibration.from_yaml()`；
- private 方法也用领域动词，不用 `_do_work()`、`_process_data()` 代替责任。

### 9.5 状态和 lifecycle

推荐的可见生命周期：

```text
construct → acquire (connect/start/load) → operate → release (stop/close)
```

用一个 enum 或一个权威 state field 表达 lifecycle，不用 `is_started`、`is_connected`、
`has_failed`、`is_stopping` 等多个可能矛盾的布尔值。缓存数据与其 source timestamp、freshness 和
generation 一起保存，不只保存“最新值”。

`connect()` / `load()` 和同步 `start()` 必须在完全 ready 后才返回，否则抛异常并保留可诊断的原因。
必须异步启动的 process/thread 把合同拆成 `start()` 和有界的 `wait_until_ready(timeout_s=...)`；
任何 operate method 都不得在 readiness 确认前被调用。重复调用是幂等还是 lifecycle error 也要在
合同中固定，不要由底层 SDK 偶然决定。
一个 lifecycle method 在抛异常前，必须释放它尚未发布给 owner 的局部资源，或者把资源
保存到 `close()` 能够访问的权威 state 中。

一个资源只有一个 owner。SDK 在 owning process 内构造并留在该 process；不把 live SDK object
作为构造参数穿过 multiprocessing boundary。`close()` 应幂等、有界，并能清理只成功一部分的
startup。不依赖 `__del__` 关闭硬件；如果 context manager 能显著让 ownership 更清楚，才实现它。

同一 workflow 拥有多个资源时，每构造一个 owner 就在启动前注册它的 cleanup。`ExitStack` 在这里是
有意义的，因为它使 partial startup cleanup 和逆序关闭同时可见：

```python
def run_policy_deployment(
    config: PolicyWorkflowConfig,
    publish_prediction: Callable[[np.ndarray], bool],
    stop_event: Event,
) -> None:
    with ExitStack() as cleanup:
        observation_reader = PolicyObservationReader(config.observation)
        cleanup.callback(observation_reader.close)
        observation_reader.start()
        observation_reader.wait_until_ready(timeout_s=config.startup_timeout_s)

        policy_runtime = PolicyRuntime(config.policy)
        cleanup.callback(policy_runtime.close)
        policy_runtime.load()

        run_policy_loop(
            observation_reader,
            policy_runtime,
            publish_prediction,
            stop_event,
            read_timeout_s=config.read_timeout_s,
        )
```

这些 `close()` 在对应 `start()` / `load()` 前就注册，因此启动到一半抛异常也会被清理。
这也要求 `close()` 能处理尚未启动和部分启动的状态。

一个可接受的 model runtime 可以写成：

```python
class PolicyRuntime:
    """Own one loaded PyTorch policy; do not publish robot commands."""

    def __init__(self, config: PolicyModelConfig) -> None:
        self._config = config
        self._backend: TorchPolicyBackend | None = None

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> None:
        if self._backend is not None:
            return
        self._backend = TorchPolicyBackend.from_checkpoint(self._config)

    def predict(self, observation: PolicyObservation) -> np.ndarray:
        backend = self._require_backend()
        return decode_joint_action(backend.predict(observation), config=self._config.action)

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()

    def _require_backend(self) -> TorchPolicyBackend:
        if self._backend is None:
            raise RuntimeError("Policy runtime is not loaded")
        return self._backend
```

这个类只拥有 model lifecycle 和 inference boundary。observation 采集、action scheduling、robot safety
和 recording 留给各自 owner，不逐渐塞进一个巨大 `run()`。

### 9.6 继承与组合

ManiUniCon 同时支持多个 robot、policy 和 sensor 实现，因此它的 `RobotInterface`、`BasePolicy`
和 `BaseSensor` 有明确上下文。DexMani Real 只在当前确有多个实现，或真实/模拟边界
需要可替换时，才引入 ABC 或 Protocol。

其他情况优先组合：

- 一个 concrete owner 持有它所需的窄 dependency；
- 共享算法提取为纯函数，不塞进巨大 base class；
- adapter 必须做真实的 shape/unit/frame/schema 转换，不只转发方法；
- 不为了“以后可能换机器人”预先建立 factory、registry 或 plugin。

### 9.7 对 ManiUniCon 的具体取舍

ManiUniCon 的 `RobotControlSystem`、`Robot`、`KeyboardPolicy`、`SpaceMousePolicy` 和
`RobotInterface` 展示了几个值得保留的特点：具体的领域类名、顶层可见的组合关系、
明确的 `connect/start/run/stop/close` 动词，以及 policy 附近独立的数学 helper。

DexMani Real 会在此基础上收紧约束：

- `__init__` 不启动 shared-memory manager、进程、监听线程，也不打开 HID 或硬件；
- `run()` 展示 orchestration，不同时展开 reset、调度、控制、录制和 recovery 的全部细节；
- `validate_*` 只校验并报错，不边裁剪输入边返回成功；
- 不用 `except ...: pass` 或无声沿用旧状态掩盖设备故障；
- 稳定的内部边界使用 typed config/result，不让深层 `dict[str, Any]` 成为默认接口。

ManiUniCon 是可读性参考，不是逐行模板。参考实现与本规范冲突时，以本项目的
安全边界和明确 lifecycle 为准。

## 10. 配置与实验复现

每个默认值只有一个 canonical 定义。当前运行时遵循：

```text
CLI override > YAML experiment file > config/defaults.py
```

具体原则：

- hardware address、limit、rate 和 workspace 不在 worker 中另设 fallback；
- CLI 未提供的参数用 `None` 表示“不覆盖”，不要复制默认值；
- component 接收它拥有的窄配置，而不是随处传递巨大配置对象；
- 简单 derived value 在 owner 附近计算，不再成为独立配置项；
- calibration 是带 provenance 的外部事实，不与普通超参数混放；
- 每次采集或部署保存/记录 resolved config identity 与关键资源 hash。

只有会被实验修改、需要记录或影响行为的值才应配置化。内部循环里不会调整的实现细节，
用靠近 owner 的常量即可。

## 11. 机器人和并发代码

真实机器人代码比一般 research script 更严格：

- SDK 只存在于 owning worker/driver，不跨 multiprocessing boundary；
- import 不连接设备、不启动线程、不改变 robot state；
- parent process 只组织 lifecycle，不直接绕过 worker 发 SDK command；
- actuator command 经过现有 safety/lifecycle boundary；
- worker 在 SDK 调用前再次检查 state、generation、shape、finite、limit 和 freshness；
- startup、readiness、shutdown 和 queue wait 都必须有界；
- 异常必须能被 supervisor 或操作者看见。

控制循环保持窄且按固定顺序阅读：

```text
read → check freshness/state → compute → validate → publish → rate control
```

文件 IO、可视化、长日志、用户交互和阻塞网络请求移出关键循环。高频诊断使用计数器、
周期汇总或 throttled warning，不逐帧打印。

## 12. 模型部署代码

模型输出是 proposal，不是机器人命令。部署边界保持：

```text
typed observation (causal T-step multimodal history)
→ dexmani_policy public runtime (load/predict/reset_episode)
→ Real NumPy adapter
→ flat joint/EE prediction
→ inference-to-executor transport (one latest prediction + endpoint index + EE→IK)
→ shared safety gate (limits + delta + collision)
→ actuator IPC
```

要求：

- checkpoint/Hydra/EMA/normalizer/Torch 只由 `dexmani_policy.deployment` public API 拥有；
- device 与 experiment 是 session 输入；PolicySpec 是 model shape、modality、action 与 dt 的只读合同；
- Real-owned observation freshness、ACK 与 watchdog timing 只从 resolved runtime 的 `policy` 段读取；调度模式与 episode action-step 上限由 deployment 配置拥有，inference cadence 来自 PolicySpec；
- Real 校验固定硬件、字段 shape/dtype/语义、控制周期与 IPC capacity，不解析 checkpoint 内部结构；
- inference/no-grad、normalization/denormalization 与 model preprocessing 均由 Policy runtime 拥有；
- EE action、joint action、degrees/radians 不做静默猜测或自动兼容；
- 不允许模型代码直接访问 hardware SDK 或绕过 policy executor 或 shared safety gate；
- `FakePolicyRuntime` 保持确定性，用于验证 observation → flat prediction 链路而非模拟真实性能。

## 13. 数据采集和离线处理

raw episode 是实验事实，应尽量不可变、可审计：

- schema、field 语义和 alignment 只有一个定义；
- recorder 序列化已选择的 fixed-grid sample，不重新做控制决策；
- robot、camera 和 action 的 source timestamp/freshness 不被悄悄丢弃；
- episode 写入先完成 sidecar 校验，再原子发布；
- 失败或中断不伪装成完整 episode；
- raw → processed → training format 的转换留在离线工具；
- 离线清洗不替操作者猜测 task success。

数据格式变更必须沿 writer → representation → reader → processing → model consumer 全链检查。
不要为了当前 notebook 方便，直接改变 persisted field 的既有含义。

## 14. 错误、日志和诊断

外部输入、硬件、IPC、模型和存储边界使用明确异常或 typed result。内部 programmer
invariant 才使用 `assert`。

禁止：

```python
try:
    ...
except Exception:
    pass
```

在 process/lifecycle 顶层可以捕获宽异常，但必须保留 traceback/context、置 fault 状态并
触发有界清理。降级行为必须明确且安全，不能用旧值或零值假装一切正常。

日志应帮助回答“哪个 owner、什么状态、哪个 sequence/generation、失败值是多少”。
不要打印整幅图像、大数组或每帧重复消息。操作者提示与开发诊断分开表达。

## 15. 注释、docstring 和 import

注释解释代码本身表达不了的内容：安全原因、坐标系、单位、时序假设、硬件怪癖和算法
选择。不要逐行翻译代码，也不要保留整段注释掉的旧实现；Git 已保存历史。

Public API 或关键 boundary 的 docstring 说明非显然合同：输入/输出、shape、dtype、单位、
frame、lifecycle 和失败语义。简单函数用一句话即可，不为完整而写长篇参数复述。

Import 分为 standard library、third-party、project 三组：

```python
import time
from pathlib import Path

import numpy as np

from dexmani_real.control.safety_gate import SafetyGate
```

禁止 wildcard import。可选或重量级依赖只有在确实需要延迟加载、隔离硬件/torch 或支持
离线工具时才放进局部 import；不要用局部 import 掩盖循环依赖。

## 16. 格式化和检查

使用 Black-compatible 的 Python 布局和 isort-style import 分组，不手工制造个人对齐格式。
格式化只作用于本次修改的文件，不借机重排整个仓库。类型检查和 lint 优先解决
修改路径上的真实问题，不要求为了
一次小实验清空所有历史 warning。

验证按风险递增，不对每次修改机械执行同一套命令：

| 修改类型 | 最小验证 |
| --- | --- |
| 纯文档 | 标题/围栏结构、本地链接、尾随空白、`git diff --check` |
| Python 语法或 import 修改 | 修改路径的 compile/import check |
| pure function、schema、reader、adapter | 小型确定性测试，覆盖边界和失败分支 |
| lifecycle、IPC、recording | fake/mock 下的 startup、normal path、timeout、partial failure 和 shutdown |
| 硬件行为 | 只在用户明确请求后执行，记录设备、模式和实际现象 |

通用的廉价检查：

```bash
git diff --check
git diff --stat
git status --short
```

Python 修改的基础语法检查：

```bash
python -m compileall -q dexmani_real examples
```

example script 不是自动化测试，不能为了“验证”就批量执行。没有实际运行硬件时，
明确报告“未做硬件验证”，不把 compile、mock 或 simulation 描述成真机验证。

## 17. 科研代码的取舍

探索阶段允许快速，但提交到主代码路径前需要完成最小收口：

- 临时常量进入配置或明确的 owner constant；
- 调试 print 删除或转为有意义、可限频的日志；
- scratch/notebook 逻辑若成为 workflow，就移到 package + thin entry；
- random seed、checkpoint、config 与输出目录可追溯；
- 失败路径不会留下仍在运动的硬件或看似完整的数据；
- 删除 dead code、复制版本和无 owner 的临时文件。

不要为了论文 deadline 构建大框架，也不要把一次性脚本直接变成长期运行时。最好的折中
通常是：先写一个具体、端到端、可验证的 vertical slice，实验稳定后再提取真实重复。

## 18. 提交前自查

### 18.1 范围和合同

- 主路径能否从入口顺着读到 side effect？
- 新名称是否说明单位、frame 和时间语义？
- 是否新增了第二份 default、schema 或 safety rule？
- 是否只修改了当前任务需要的文件和边界？

### 18.2 函数和类

- 函数名是否如实表达 side effect？返回类型与失败合同是否稳定？
- `validate_*` 是否修改了输入？裁剪或归一化是否用独立动词显式表达？
- class 是否真的拥有状态或资源？wrapper 是否增加了语义？
- `__init__` 是否意外连接设备、启动线程/进程或加载模型？
- `run()` 和 loop 是否可以像目录一样顺读？还是混合了多个独立责任？

### 18.3 安全和数据

- control loop 中是否混入 IO、GUI 或高频日志？
- model proposal 是否仍经过统一安全边界？
- raw 数据语义和 alignment 是否保持不变？
- import 是否新增硬件 side effect？
- failure 是否可见、fail closed 且能有界退出？

### 18.4 验证和交付

- 能否用更直接的实现达到同样的正确性？
- 是否检查了 normal path 以及至少一个相关 failure path？
- 最终 diff 是否包含调试代码、无关格式化或重复逻辑？
- 离线验证和未做的硬件验证是否记录清楚？

如果一个更简单的实现同样安全、正确并容易复现，选择更简单的实现。
