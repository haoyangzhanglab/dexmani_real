# 参考优先开发协议

## 核心原则

开发新功能/模块前，必须先按 P1→P2→P3 优先级检索参考项目，理解设计意图和边界条件，再结合本项目实际情况决定采纳方式。**禁止不参考现有实现就凭空编写代码。**

## 优先级体系

| 优先级 | 项目 | 角色 |
|--------|------|------|
| **P1** | LeFranX + ManiUniCon | 决定架构设计和接口风格 |
| **P2** | BunnyVisionPro + Open-Teach + DexUMI | 决定硬件控制代码写法（xArm7 + XHand + Quest VR）和策略部署架构 |
| **P3** | Bidex_Manus_Teleop | 手部数据手套和 IK retargeting 补充 |

P2 内部优先级：**BunnyVisionPro > DexUMI > Open-Teach**（BunnyVisionPro 硬件最接近 xArm7；DexUMI XHand 封装和数据 pipeline 最完整；Open-Teach 补充抽象模式）。

## 检索流程

```
1. 查 CLAUDE.md Section 0.5.3 映射表 → 列出相关参考文件
2. 必须 Read P1 参考文件（LeFranX + ManiUniCon 各至少一个）
3. P1 未覆盖 → Read P2 参考（BunnyVisionPro 优先）
4. P2 未覆盖 → Read P3 参考（仅手部/IK 场景）
5. 无参考 → 标记「无参考实现」，按本项目接口规范自行设计
6. 在实现注释中标注参考来源：
   "# ref: [P1] LeFranX arm_ik_processor.py L120-150"
   "# ref: [P2] BunnyVisionPro xarm7_ability.py L80-100"
```

## Fact-Check 规则

1. **必须 Read 实际代码**，不基于文件名猜测行为
2. **验证关键声明**：README/注释可能与代码不一致
   - 例：Bidex README 声称用 "SDLS" IK，代码实际使用 `p.IK_DLS`
3. **检查硬件兼容性**：
   - Franka → xArm7: 力矩控制 vs 位置控制，运动学模型不同
   - ManiUniCon xArm6 → xArm7: SDK API 可能有版本差异
   - BunnyVisionPro XArm7: 同型号，可直接参考硬件调用
   - Open-Teach Allegro → XHand: 手型结构不同，仅参考滤波/限位模式
   - DexUMI UR5 → xArm7: 运动学模型不同（6 vs 7 DOF），仅参考 Server/Client 架构
   - DexUMI iPhone ARKit → Quest VR: 手腕追踪方式不同，仅参考坐标变换链设计
   - DexUMI 外骨骼编码器 → VR hand tracking: 手指追踪源不同，仅参考校准模型加载模式
   - DexUMI Zarr → HDF5: 存储格式不同，仅参考 RecorderManager 编排模式
   - DexUMI ZMQ IPC → Shared Memory: 通信机制不同，仅参考架构分离模式
4. **检查依赖兼容性**：是否引入不采纳框架（ROS、Hydra 等）
5. 多参考库实现冲突时：P1 > P2 > P3；同优先级按硬件接近度选择

## 适用性判断

| 判断 | 含义 | 注释格式 |
|------|------|---------|
| **Adopt** | 可直接采纳核心逻辑 | `# ref: [P2] BunnyVisionPro xarm7_ability.py L120-150` |
| **Adapt** | 需适配硬件/依赖差异 | `# ref: [P2] Open-Teach allegro_retargeters.py — 仅参考滤波模式` |
| **Skip** | 不适用 | 记录原因，不写入代码注释 |

## 不采纳清单

以下技术已确认不适用，避免重复评估：

| 不采纳 | 来源 | 原因 |
|--------|------|------|
| libfranka + Ruckig C++ server | LeFranX | xArm7 内置伺服控制 |
| geofik + Brent 解析式 IK | LeFranX | 已有 MPlib 数值 IK + 碰撞检测 |
| LeRobot Parquet / draccus CLI | LeFranX | 使用 HDF5 + @dataclass |
| Hydra 配置管理 | Open-Teach / ManiUniCon | 使用 @dataclass |
| ROS/ROS2 通信层 | Open-Teach / Bidex | 使用 multiprocessing + shared memory |
| Vision Pro 专用 API | BunnyVisionPro | 本项目使用 Quest 3 |
| Allegro Hand 专用 retargeting | Open-Teach | 手型差异 |
| Manus Core C++ SDK 直连 | Bidex | 不依赖 Manus 数据手套 |
| Pydantic 模型验证 | ManiUniCon | 使用 @dataclass + `__post_init__` |
| LeFranX dict 风格 action | LeFranX | 使用结构化的 RobotAction dataclass |
| Zarr 存储格式 | DexUMI | 本项目使用 HDF5 |
| ZMQ IPC 通信 | DexUMI | 本项目使用 multiprocessing + shared memory |
| UR5 RTDE servoL 控制 | DexUMI | xArm7 使用 set_servo_angle_j() |
| `DexterousHand` ABC 接口（write/send 分离）| DexUMI | 本项目统一 send_action(np.ndarray) → bool |
| iPhone ARKit 手腕追踪 | DexUMI | 本项目使用 Quest 3 VR |
| 外骨骼编码器手指追踪 | DexUMI | 本项目使用 VR hand tracking + dex-retargeting |
| `ExoDexterousHand.predict_motor_value()` | DexUMI | 外骨骼专用，VR 遥操作不需要 |
| `Recorder` Protocol 类 | DexUMI | 参考接口设计思想，本项目使用具体类 |

## SDK 参考资源（P0 — 开发前必查）

> 详细 API 签名、集成模式和已知陷阱见 `.claude/rules/sdk-dependencies.md`。

| SDK | 版本 | 来源 | 用途 |
|-----|------|------|------|
| **xarm-python-sdk** | 1.18.x | [GitHub](https://github.com/xArm-Developer/xArm-Python-SDK) / PyPI | xArm7 硬件控制 |
| **xhand_controller** | 1.1.8 | 本地 wheel `/home/zhy/Documents/硬件/Xhand/SDK/Python/` | XHand 硬件控制 |
| **hand-tracking-sdk** | latest | [GitHub](https://github.com/wengmister/hand-tracking-sdk) / PyPI | Quest VR 手部追踪 |
| **dex_retargeting** | latest | [GitHub](https://github.com/wengmister/vr-dex-retargeting) / PyPI | 手部重定向 (DexPilot) |
| **mplib** | 0.2.1 | [Docs](https://motion-planning-lib.readthedocs.io/latest/) / PyPI | 运动规划 (IK/路径/碰撞) |
| **sapien** | 3.0.3 | [Docs](https://sapien.ucsd.edu/) / PyPI | 物理仿真 (SimRobotInterface) |

**SDK 参考协议：**
- 修改 hardware driver（robot/*.py）→ 先查 `sdk-dependencies.md` 对应章节
- 遇到 SDK 版本不兼容 → 更新 `sdk-dependencies.md` 的「已知陷阱」
- 新增外部 SDK 依赖 → 在 `sdk-dependencies.md` 追加完整章节（API + 陷阱 + 版本）
- 与代码参考库冲突时：SDK 文档 > 代码参考（SDK 是 ground truth）

## 快速检索工具

```bash
# 按模块查映射表
python .claude/skills/ref-check/ref-search.py <module>

# 动态关键词搜索
python .claude/skills/ref-check/ref-search.py --keyword "<关键词>" -p <最高优先级>

# 验证路径有效性
python .claude/skills/ref-check/ref-search.py <module> --check-exists
```
