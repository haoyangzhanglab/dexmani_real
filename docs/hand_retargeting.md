# XHand 手部重定向

这次实现以 Sharpa 的掌坐标与固定尺度标定、Wuji 的手形/指尖/末节方向联合优化为主线，针对本仓库的右手 XHand、HTS 21 点和 16 Hz 控制链重新实现。它是几何重定向，不输出接触力，也不需要学习模型或物体网格。

## 为什么替换原来的封装

基线提交为 `16918e51e87850ea9dacb86748931c8d7493f404`。原实现直接调用外部 `DexPilotOptimizer.get_objective_function()`，仓库没有固定该依赖版本；默认全局倍率 1.15、额外小指倍率 1.15，并用滤波前的优化解推进下一帧初值。腕部状态等提交成功后才更新，手部求解和滤波状态则在候选计算时已经更新。

新实现把目标函数与解析梯度放在同一处，取消外部 DexPilot/Torch 路径及其离散投影状态；让手部和腕部共用现有提交边界。保留现有运行限位、ActionMapper 准入、反馈时效、取消/急停及发送期限。这里没有复现部署工作站所安装的旧 DexPilot 版本，因此不能把离线消融称为对旧版本的实测性能比较。

## 参考来源与取舍

| 来源 | 本实现采用 | 没有直接移植的部分 |
| --- | --- | --- |
| [Sharpa vision SDK](https://github.com/sharpa-robotics/sharpa-vision-retargeting-sdk/tree/73ac80e980b980b11d8853b874f8efea79095add) | 逐帧掌坐标对齐、会话内固定人体尺度 | 平手角度修正、Sharpa 的 22 关节/关键点映射；本输入已有米制 3D 骨架 |
| [Wuji retargeting](https://github.com/wuji-technology/wuji-retargeting/tree/531f6ed4250b475d2e9231f54e988fc9b1c5b4ea) | 手形、指尖位置、末节方向联合目标，解析梯度，有界 SLSQP | 20 自由度映射、厘米制权重、硬编码分段比例 |
| [AKO](https://github.com/Mingrui-Yu/retargeting/tree/3846d3fa207165bb0d498145aac8b885a28ea923) | 按捏合意图连续调整几何目标与权重 | 联合机械臂/腕部优化；本仓库已经有独立的腕部锚点和机械臂控制 |
| [AnyDexRetarget](https://github.com/qqsq12321/AnyDexRetarget/tree/9e49c03123bd9ebd99e4e1a04e893a1caa7bd9b3) | 显式对指关系、函数值和梯度一致的正则 | 单个 argmax 对指伙伴切换；这里同时支持四个连续激活目标 |
| [Smooth Operator](https://github.com/mimicrobotics/mimic_retargeter_lab/tree/4bf36ddc3516c3bdf651f857ce39db87673af585) | 预热后记录 p50/p95/p99、做目标函数消融 | GPU 批量采样；本控制链采用已有 CPU FK 与 NLopt |

GeoRT/AnyDexRT 的学习映射需要新的训练/标定证据；TopoRetarget/SPIDER 需要物体或动力学输入；ReForce 类方法需要可解释的接触力信号。当前 Raw 的触觉字段标注为 SDK 原生未知物理标度，不能直接当成标定力加入损失。这些方向不进入本次默认在线链路。

上述来源用于算法设计与审计，本次未 vendor 第三方优化器源码。

## 坐标、标定与 XHand 的可表达几何

输入沿用接收端转换后的右手 FLU、米制、21 点顺序。先减 wrist，再用 wrist→middle MCP 和 index MCP→pinky MCP 建立正交掌系。轴为 +X 掌侧、+Y 拇指侧、+Z 指向中指；只使用 proper rotation，不使用坐标反射。平移或整体旋转不会改变输出目标。

启动后采集 25 个新鲜样本；ring 编号和来源接收时间都必须前进，反复读取一个缓存帧不增加计数。标定取每段指骨长度和掌宽的中位数，默认相对 MAD 小于 10%，单帧骨长偏差不超过 35%。无需平手，因为使用骨链弧长，而不是会随握拳缩短的 MCP→TIP 直线长度。退化掌面、非有限输入和 2–180 mm 范围外的骨段明确拒绝。此校验用于发现单位和明显跟踪错误，不是完整的人体测量或遮挡置信度模型。

标定在一次进程会话中固定；暂停/继续仅重置实测关节初值。更换操作者需重启会话重新标定。实际标定写入 Raw 的 `meta.hand_retargeting_calibration_json`，包括方法版本、样本数、15 段人体骨长和掌宽。

| 手指 | 人手对应点 | XHand 拟合点 | 尺度处理 |
| --- | --- | --- | --- |
| 拇指 | MCP(2)、IP(3)、TIP(4)；CMC(1) 为基点 | thumb_rota_link1 / link2 / tip | 三段分别按中立 URDF 长度与人体标定骨长比缩放 |
| 食指 | MCP(5)、PIP(6)、TIP(8) | index_rota_link1 / link2 / tip | MCP→PIP 单段；PIP→DIP→TIP 合为末段 |
| 中/环/小指 | 对应 MCP、PIP、TIP | 对应 link1 / link2 / tip | 同食指；不再额外逐点放大小指 |

非拇指 MCP 以机器人中立模型位置为基点，拟合 PIP/TIP；不会假设它们具有 Wuji 那样的完整独立侧摆和 DIP 自由度。末节方向使用人手 DIP→TIP，机器人使用最后刚体的 link2→tip。位置与方向可能互相冲突，优化器通过权重折中。中立模型的零关节只用于 FK 取几何，不下发零位命令。

## 联合目标与导数

记机器人三点为 $x_{f,k}(q)$，几何目标为 $t_{f,k}$，末节单位方向为 $u_f(q)$，人体末节方向为 $u_f^h$。位置残差除以 $s=0.10$ m，径向 Huber 阈值为 $\delta=0.02/s$；内部统一米与弧度，不混用厘米权重。

对四个非拇指，先用固定掌宽比例归一化人手对指距离 $d_i$，再定义

$$
z_i=\operatorname{clip}\left(\frac{d_{\mathrm{start}}-d_i}{d_{\mathrm{start}}-d_{\mathrm{full}}},0,1\right),\qquad
a_i=z_i^2(3-2z_i),\qquad
g_i=(1-a_i)d_i+a_i d_{\mathrm{gap}}.
$$

默认 start/full/gap 分别为 50/20/12 mm。这里的 gap 是两个 URDF tip 原点之间的目标距离，不能解释为物体厚度、皮肤表面间隙或实际接触力。距离和激活在两端一阶连续；人手两个 tip 重合时仍有限，不需要归一化零向量。

总目标为

$$
E(q)=
\frac{1}{5}\sum_{f,k}w_{f,k}\rho_\delta\left(\frac{x_{f,k}(q)-t_{f,k}}{s}\right)
+\frac{w_{\mathrm{dir}}}{10}\sum_f\|u_f(q)-u_f^h\|^2
+\frac{w_{\mathrm{pinch}}}{4}\sum_i a_i^2
\rho_\delta\left(\frac{\sqrt{\|x_{i,\mathrm{tip}}-x_{0,\mathrm{tip}}\|^2+10^{-12}}-g_i}{s}\right)
+\frac{\kappa}{2\Delta t}\|q-q_{\mathrm{sent}}\|^2.
$$

末项是近端阻尼，按 $1/\Delta t$ 缩放，函数值和梯度都包含；不把它宣传为实测速度惩罚。实际关节目标变化率另由硬边界约束。近指节形状权重为 $w_{\mathrm{shape}}(1-0.8a_f)$，指尖权重固定；拇指的激活为 $1-\prod_i(1-a_i)$，避免非光滑的伙伴 argmax。默认 shape/tip/direction/pinch 权重为 1/2/0.08/12。

位置雅可比由 Pinocchio 计算并转换到 SDK 列顺序。方向导数使用
$\partial(v/\|v\|)/\partial q=(I-uu^\top)J_v/\|v\|$，对指项使用两个 tip 雅可比的差。一次目标函数评估共用一遍 FK/关节雅可比；不经过 Torch 自动微分。新增数值测试直接使用仓库真实 URDF，覆盖全部 12 个 SDK 关节。

## 求解与提交状态

1. 初值、阻尼中心和滤波参考统一为上一次成功提交的手命令；新段用实测手关节初始化到运行限位内。
2. 最终命令的速率约束为 $|q_{\mathrm{cmd}}-q_{\mathrm{sent}}|\le v_{\max}\Delta t$。优化器搜索的是滤波前候选，因此搜索盒为运行限位与 $[q_{\mathrm{sent}}-v_{\max}\Delta t/\alpha,\ q_{\mathrm{sent}}+v_{\max}\Delta t/\alpha]$ 的交集，默认 $v_{\max}=4$ rad/s；这样提高采样率不会把可达到的最大命令速率额外缩小为 $\alpha v_{\max}$。
3. SLSQP 默认最多 50 次评估，时间预算 25 ms。NLopt 时间限制是软预算，单次评估及系统调度可越过它；原控制链的绝对提交期限仍负责拒绝过期命令。
4. 只接受正状态、有限、满足边界且损失不劣于初值的候选。预算耗尽但候选满足这些条件时可使用，诊断中标记 `budget_limited`，不能称为已收敛。负状态、roundoff 失败和非法候选返回具体拒绝原因，不回填旧目标伪装成功。
5. 单层时间滤波 $\alpha=1-e^{-\Delta t/\tau}$，默认 $\tau=0.04$ s。这是对目标序列的滤波；整个非线性优化并不因此获得严格的跨频率等价性。
6. `VRController.commit()` 只在既有发送调用成功之后接收实际提交结果。预计算、机械臂拒绝、取消或发送失败均不推进手部状态。重复读取同一 VR ring 帧只复用候选，不再求解或叠加滤波。

滤波是运行限位中两个点的凸组合，因此保持关节位置范围；命令步长由 $\alpha(v_{\max}\Delta t/\alpha)=v_{\max}\Delta t$ 保证。它不保证滤波后仍是最优解，也不保证电机已达到命令。碰撞仍由现有组合模型准入检查；本优化器没有自碰撞或物体接触约束，不能凭几何捏合损失断言可直接完成真实抓取。

## 配置迁移

`dexpilot_retargeting` 和 `DexPilotConfig` 已删除，不保留静默兼容分支。现有 YAML 中的旧节需要删除，按需改为：

```yaml
hand_retargeting:
  smoothing_tau_s: 0.04
  max_joint_speed_rad_s: 4.0
  shape_weight: 1.0
  tip_weight: 2.0
  direction_weight: 0.08
  pinch_weight: 12.0
  pinch_start_m: 0.05
  pinch_full_m: 0.02
  pinch_gap_m: 0.012
  max_evaluations: 50
  max_solve_time_s: 0.025
```

全局 scaling、小指倍率被实际骨长标定替代，没有逐项等价迁移。旧 project/escape 是离散投影的迟滞阈值，与新连续 start/full 的含义不同。仅就滤波而言，16 Hz 下旧 alpha=0.5 对应 $\tau\approx0.09017$ s；改为该值也不会恢复旧算法输出。新默认权重、速率、gap 都是待真实录制和真机检验的工程起点。

## 离线验证与复现

在已有实验环境中运行；配置打印不加载数值后端或设备：

```bash
python -m unittest discover -s tests -p 'test_hand_*.py' -v
python scripts/collect_teleop.py --print-config
python scripts/run_policy.py --print-config
python scripts/tools/evaluate_hand_retargeting.py --synthetic --ablations
python scripts/tools/evaluate_hand_retargeting.py --episode episodes/TASK/EPISODE --config local.yaml --ablations
```

本次 41 项测试通过、0 跳过；数值验证环境为 Python 3.12、NumPy 1.26.4、Pinocchio 2.7.0、NLopt 2.7.1。FK、方向/位置/对指/阻尼的完整梯度、可达目标、SDK 顺序、旋转/人体尺度不变性、限位、失败及提交状态均使用实际实现验证。启动标定测试使用时钟、VR、设备替身，且替换不参与测试的 Ruckig 导入；不是整机运行测试。本环境 Ruckig 0.17.3 的源包构建受 scikit-build-core 配置错误阻塞，未修改项目版本以掩盖该缺口。

合成评估为固定 320 帧、16 Hz、真实 URDF 生成的骨长恒定序列。所有消融使用同一输入、模型、标定、关节范围、时间滤波和初始化。下表为一次离线运行；位置误差针对本算法构造的目标，不是人工真值，也不代表任务成功率：

| 配置 | 指尖目标误差均值 mm | 活跃对指距离误差均值 mm | 完成帧数 |
| --- | ---: | ---: | ---: |
| 完整目标 | 3.07 | 1.36 | 320/320 |
| 去掉近指节手形 | 3.10 | 1.32 | 320/320 |
| 去掉末节方向 | 3.22 | 1.29 | 320/320 |
| 去掉对指距离 | 2.45 | 6.78 | 320/320 |

完整目标在该次共享 CPU 环境运行的重定向耗时 p50/p95/p99 为 4.77/8.28/11.87 ms，最大 15.26 ms；不包含机器人 I/O、机械臂 IK 和传感延迟。后台负载和调度会造成更长尾延迟，此前消融运行已出现超过 25 ms 的例子，不能据此宣称硬实时。上述结果体现捏合与位置模仿的折中，不能证明全面优于原 DexPilot、Sharpa 或 Wuji。

评估器仅重放 Raw VR 字段，不连接设备；首次拒绝即停止此段，不跳过坏帧继续虚构执行。新 Raw 优先恢复保存的标定；旧 Raw 用前 25 个有效且来源时间递增的样本离线估计，报告中明确标记，两者不是同一标定实验。参数使用当前配置或显式 `--config`，不会偷偷从旧 runtime JSON 猜测新参数；比较前应核对实际运行限位和控制周期。评估输出包含使用的配置、标定、预算停止次数和延迟分位数。

下一步应以同一操作者录制的张合手、单指屈曲、拇食/拇中/拇环/拇小指捏合及多指包络序列复核，关注末节朝向、无法表达的侧摆、跟踪遮挡、暂停恢复和目标拒绝率。实际 XHand 指尖表面偏移、触觉标定、抓取成功率、执行速度及端到端时延尚未验证；本次没有连接或发送任何硬件动作。
