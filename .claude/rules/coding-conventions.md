# 编码约定

## 命名规范

| 规则 | 示例 |
|------|------|
| 类名 PascalCase | `XArm7`, `TeleopController` |
| 函数/方法 snake_case | `send_action()`, `get_state()` |
| 变量 snake_case | `connected_flag`, `last_qpos_cmd` |
| 常量 UPPER_SNAKE_CASE | `JOINT_NAMES`, `OPERATOR2MANO_RIGHT` |
| 配置类 `XxxConfig` 后缀 | `XHandConfig`, `PipelineConfig` |
| 私有方法加前导下划线 | `_tick()`, `_apply_retargeted_angles()` |
| 公有属性不加前导下划线 | `connected_flag` 而非 `_connected_flag` |

> **ref:** P1 LeFranX 和 ManiUniCon 均严格使用 snake_case；P2 BunnyVisionPro 和 Open-Teach 同。P2 Open-Teach 存在部分 camelCase 类名，本项目不采纳。

## 方法命名统一

| 统一名称 | 禁止使用 |
|---------|---------|
| `get_state()` | `get_obs()`, `get_observation()` |
| `send_action()` | `move()`, `control_*()`, `set_*()` |
| `connect()` / `disconnect()` | `start()` / `stop()`（传感器也统一用 connect/disconnect） |

> **ref:** ManiUniCon 的 `RobotInterface` ABC 统一了 `connect/disconnect/get_state/send_action` 接口；LeFranX 使用 `get_observation` 命名，本项目不采纳。

## 配置管理

- 所有配置使用 `@dataclass` 定义（不使用 Hydra、OmegaConf、draccus）
- 可变默认值（`np.ndarray`、`list`、`dict`）必须使用 `field(default_factory=...)`
- 每个模块独立配置类，物理量单位在字段名或注释中显式标注
- 需要运行时验证的配置在 `__post_init__` 中实现（替代 Pydantic 的自动验证）

```python
@dataclass
class DeviceConfig:
    dt: float = 1.0 / 50.0
    home_qpos: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float64))
    qpos_min: np.ndarray = field(
        default_factory=lambda: np.full(7, -np.pi))
    qpos_max: np.ndarray = field(
        default_factory=lambda: np.full(7, np.pi))

    def __post_init__(self):
        assert self.qpos_min.shape == self.qpos_max.shape
        assert self.dt > 0
```

> **ref:** P1 LeFranX 使用 `@dataclass` + `register_subclass` 工厂模式；P1 ManiUniCon 使用 Hydra YAML（本项目不采纳）。P2 Open-Teach 使用模块级常量（仅适用于简单场景）。

## 模块结构

每个模块文件提供简单 `example()` 函数，不默认提供 argparse CLI：

```python
def example():
    config = DeviceConfig(...)
    device = Device(config)
    device.connect()
    ...

if __name__ == "__main__":
    example()
```

批量参数管理放 `scripts/` 下，不在模块文件中内嵌 CLI。

> **ref:** P1 ManiUniCon 和 LeFranX 均使用 `scripts/` 目录管理批量脚本。

## 物理量标注

所有物理量在注释中显式标注单位：

```python
qpos: np.ndarray      # rad
qvel: np.ndarray      # rad/s
tau: np.ndarray       # N·m
eef_pos: np.ndarray   # m
current: np.ndarray   # mA
temperature: np.ndarray  # °C
tactile: np.ndarray   # N
timestamp: float      # seconds
```

## 导入规范

- 核心模块（robot/sensor/controller）不强依赖可视化、训练框架
- cv2/open3d 放 viewer 或函数内部局部 import
- torch/wandb 只在 deploy/ 代码中使用
- 硬件驱动只依赖硬件 SDK + numpy + 标准库

```python
def visualize(...):
    import cv2  # 局部 import
    ...
```
