# Examples 实现审查 — 2026-07-18

多智能体审查：15 个按文件审查者 + 5 个横向审查者（16Hz 一致性 / 安全架构 / 三胞胎漂移 / real-sim 漂移 / 录制正确性），全部非 info 发现由独立对抗验证者逐项复核（须亲自追溯库代码方可确认）。原始发现 153 项 → 反驳 19 项 → 确认 100 项 + 34 项 info 观察。严重性以验证者校正后为准。基于工作区状态（含 collection-hardening 未提交修改）。

**统计**: critical 1 / major 35 / minor 64 / info 34（反驳 19）


## CRITICAL

### `examples/real/replay_traj.py:683` — 回放无起点对齐：首帧目标直接甩给 ArmInnerLoop，臂可能以最高 90°/s 无检查扫向轨迹起点
- 来源: replay_traj | 验证置信: high

run() 在进入循环前直接把 traj.action_arm_joint[0] 通过 set_target 发给 ArmInnerLoop，全程没有比较当前 arm_qpos 与轨迹起点的偏差、没有确认提示、也没有用已创建的 planner 规划一条到起点的安全路径。ArmInnerLoop 的每步 delta clamp (max_joint_delta=0.3) 只在 _last_sent_target 非 None 时生效——首次发送前该值恒为 None（启动期 target=None 走 continue 分支，不经过 _hold_position），因此首发目标不受任何幅度限制。Mode 6 固件会在 0.4s 软启动斜坡后以 joint_max_speed=90°/s、acc=500°/s² 直接规划到该目标，且 collision_sensitivity=1（最低）。若臂停在 home 而轨迹起点在桌面另一侧，臂会高速大幅扫过工作区（桌上有物体/相机支架时即为硬件危险）。回放脚本必须先做起点对齐（阈值检查 + planner 规划慢速趋近或拒绝执行）。

<details><summary>验证记录</summary>

逐一核实发现中的所有代码引用，全部确认存在且逻辑链成立：

1. 起点对齐缺失：replay_traj.py:683-684 在进入主循环前直接将 traj.action_arm_joint[0] 通过 set_target 发给 ArmInnerLoop，其上方 (lines 671-675) 虽然读取了当前 arm_qpos，但仅用于初始化 state 变量——从未与 action_arm_joint[0] 做偏差比较，也无任何确认提示。setup() (lines 612-645) 只做 connect + preflight_check + 启动 inner loop，preflight_check 仅验证连接/响应性，不做位置校验。planner 在 line 617 创建、line 618 用于构造 RobotInterface，在 run() 的 line 666 仅做 assert，全程未用其 plan_path() / solve_teleop_ik() 规划从当前位姿到轨迹起点的安全趋近路径。

2. 首发绕过 delta clamp：inner_loop.py:155 _last_sent_target = None；start() (line 209-216) 不设此值；_run() 启动阶段 (lines 280-307) 仅读取初始关节角写到 _arm_qpos，不调用 _send_target；主循环中 target=None 时走 line 334 continue，同样不到达 _send_target。因此当 replay_traj.py 首次 set_target 后，inner loop 处理目标时 line 503 条件 self._last_sent_target is not None 为 False，delta clamp 被完全跳过，目标角度无截断直达固件。

3. 速度斜坡后全速运动：inner_loop.py:511-519，_ramp_step 初始化为 0 (line 156)，首帧 speed = speed_ramp_min + 0 = 0.2 rad/s (约11.5度/s)，20 帧 (0.4s) 后 ramp 结束，speed = joint_max_speed = 1.5708 rad/s (90度/s)，acc = 8.7266 rad/s^2 (500度/s^2)。斜坡仅提供短暂的慢启动，不阻止臂最终以全速冲向轨迹起点。

4. 碰撞灵敏度最低：inner_loop.py:263 arm.set_collision_sensitivity(1)。xArm SDK 文档 (set_collision_sensitivity) 确认 sensitivity 范围 0-5，0=关闭碰撞检测，1-5=递增灵敏度。级别 1 是启用碰撞检测的最低灵敏度，需要更大的碰撞力才触发停止，减速保护最弱。

5. 无 validate_action 保护：replay_traj.py line 764 直接将 arm_cmd 通过 self._arm_inner.set_target(arm_cmd) 发送，完全绕过了 RobotInterface.validate_action() 的 pre-send gate (workspace clamp, torque gate, temperature gate 等)。hand 命令走的是 self.robot.send_action(action) (line 778)，但 arm 命令不受此保护。

综上，臂从任意初始位置（如 home）到轨迹起点之间无任何代码级位置校验、无 planner 规划的慢速安全趋近、无 delta clamp 的首发保护，仅靠 0.4s 速度斜坡做短暂缓冲。若臂停在 home 而轨迹起点在桌面另一侧，臂会在 0.4s 后以 90度/s、碰撞灵敏度最低的状态扫过工作区。这是真实存在的硬件安全缺陷，评级 critical 恰当。
</details>


## MAJOR

### `dexmani_real/recording/episode_recorder.py:615` — stop_episode 异步化打破 RecordingSession.stop 阻塞合同，进程退出时 HDF5 断尾/数据损坏
- 来源: x-recording | 验证置信: high | 原评级: critical

EpisodeRecorder.stop_episode() 在 line 615-623 创建 daemon 线程执行 _stop_episode_impl（flush/gzip/close），立即返回 path。但 RecordingSession.stop() docstring (recording_session.py:70-71) 承诺 'the file is fully written before return'。用户 S 保存后立即 Q 退出，daemon 被 kill，HDF5 缺 /meta 尾部属性（schema_version/num_frames/success）且未 close()，下游工具无法打开。Discard 路径在无相机帧时文件尚未创建，unlink 空操作后 daemon 事后创建文件，导致应丢弃的 episode 泄露到磁盘。

<details><summary>验证记录</summary>

核实属实，但 critical 评级过高，应为 major。

已确认的事实：(1) episode_recorder.py:598-623 的 stop_episode() 确实创建 daemon 线程后立即返回 path；/meta 尾部属性（schema_version/num_frames/success/duration/fps/min_frames_met）仅在 _stop_episode_impl 的 686-697 行写入，close() 在 700 行，均在 daemon 线程中。(2) recording_session.py:70-71 docstring 承诺 'the file is fully written before return'，但 _handle_stop (103-120) 在 daemon 刚 spawn 后即 done.set()——合同确实被打破。(3) 全代码库只有 start_episode (episode_recorder.py:134-136) join 了 _stop_thread；controller._shutdown() (controller.py:814-821) 只 join session 线程，vr_teleop_shm.py main() 在 controller.run() 返回后即 vr_receiver.stop()+robot.disconnect() 退出，退出路径无人 join daemon。(4) tests/test_episode_recorder_hz.py:62-63 在 stop_episode() 后手动 rec._stop_thread.join(timeout=10.0) 才读文件——作者自己知道调用方必须 join 才能拿到完整文件，而 RecordingSession 恰恰没做。(5) Discard 泄露路径已逐行确认：主生产入口 vr_teleop_shm.py 未传 camera_process（构造调用无此参数，cfg 无 multi_camera_configs），_ensure_hdf5 仅由 flush 触发；flush_interval=10s（160 帧@16Hz），故 <10s 的 episode 在 discard_episode() unlink (collection_loop.py:161-164) 时文件尚不存在，unlink 空操作后 cam-writer 的 final flush / daemon 的 _flush_buffered() → _ensure_hdf5() (episode_recorder.py:557-558→376-382) 在 unlink 之后创建文件——被丢弃的短 episode（D 键、录制中 Q、VR 断连自动丢弃 controller.py:289、shutdown:819）大概率泄露 .h5 到磁盘。

降级为 major 的理由：(a) 异步 stop 是 HEAD 已有的有意设计（git diff 确认本分支只改了 control_hz 参数化），bug 实质是 RecordingSession 的阻塞合同未同步更新为 join；(b) S→Q 杀死 daemon 的窗口在无相机生产入口很窄：daemon 工作量约 1MB gzip（<0.5s），而退出 teardown（arm_inner.stop + vr_receiver.stop + robot.disconnect）耗时相当或更长，且 HDF5 C 库 atexit 常能兜底 close——典型后果是缺尾部 meta 属性（DataValidator 可检出）而非'无法打开'，发现里该表述过重；带相机的 record_plus 入口不走 RecordingSession，且退出前有交互式 post-loop 提示（vr_teleop_arm_only_record_plus.py:898-905）给 daemon 留出时间；(c) 泄露的 discard 文件带 success=False + min_frames_met=False meta（daemon 写入），按 CLAUDE.md 下游以 meta 属性过滤，污染训练集的风险有限，主要是磁盘垃圾+discard 语义失效；(d) validate_on_stop 默认 False（collection_config.py:34），验证器与 daemon 并发读写的竞态是潜伏问题非生产现役；(e) start_episode 已 join，无 episode 重叠损坏。综上：合同违背+退出竞态+discard 泄露均属实且需修（退出路径 join _stop_thread、discard 改为 join 后 unlink），但'保存数据高概率损坏'不成立，评 major。
</details>

### `examples/real/calibrate_camera.py:506` — 算法比选与质量门槛均不防 NaN：NaN std 可锁死 best 并绕过写盘门槛，把 NaN 外参写入 cameras.json
- 来源: calib_cam | 验证置信: high

calibrate_and_select 中 std_mm=float(errors_mm.std())：若第一个不抛 cv2.error 的算法（退化采样——正是文件头警告的'纯平移采样'操作失误——下 calibrateHandEye 可能输出非有限/退化的 T）产出含 NaN 的残差，则 best 先被 (NaN, ...) 占据，之后所有有限 std 的比较 `std_mm < best[0]` 因 finite<NaN 恒为 False 而永远无法取代它。随后 861-862 行 `pos_bad = std_mm > 5.0`、`rot_bad = std_deg > 3.0` 对 NaN 均为 False → 双门槛'通过' → 874 行 save_cameras_json 把含 NaN 的 T_world_camera 写入生产配置 cameras.json（旧文件已被 rename 走）。读取端 dexmani_real/config/camera_calib.py 的 _pose_to_matrix 对 position/orientation 无任何有限性校验，scipy from_quat 对 NaN 四元数静默产出 NaN 矩阵 → 之后每条录制 episode 的 camera_T_world_camera 与点云世界变换全部被污染。全流程对 T_candidate 没有任何 np.isfinite 检查。缓解因素：终端会打印出 NaN 数值、旧文件有备份，但依赖操作者肉眼发现。

<details><summary>验证记录</summary>

代码结构性漏洞全部经独立验证确认。

已确认的缺陷:
1. calibrate_camera.py:506 — if best is None or std_mm < best[0]: 当std_mm为NaN且best非None时，NaN小于finite恒为False，之后有限std永远无法替换已占位的NaN best。
2. calibrate_camera.py:861-862 — pos_bad = std_mm > 5.0 / rot_bad = std_deg > 3.0: NaN参与比较恒为False，双重质量门槛被静默绕过。
3. calibrate_camera.py:558-560 — json_path.rename(backup)在save_cameras_json内发生在任何T矩阵校验之前: 若后续R.from_matrix(line 524)因NaN旋转矩阵抛出LinAlgError: SVD did not converge(已实证)，旧cameras.json已被rename走，新文件未写入导致文件丢失。若仅position含NaN而rotation有限，则json.dump以allow_nan=True写入NaN token，经json.load读回为float NaN，而后进入_pose_to_matrix和h5py链路污染后续所有录制episode的camera_T_world_camera。
4. camera_calib.py:57-74 _pose_to_matrix — 对position/orientation无任何np.isfinite()或NaN校验，R.from_quat对NaN四元数产生NaN矩阵，下游无阻传播。

实证验证(conda real_robot环境, OpenCV 4.9.0, scipy 1.11.x):
- 构造纯平移退化数据(10样本, rotation均为np.eye(3)): cv2.calibrateHandEye的PARK方法返回NaN rotation+有限translation, DANIILIDIS返回NaN rotation+NaN translation, 均未抛出cv2.error。
- scipy R.from_matrix对全NaN矩阵抛出LinAlgError: SVD did not converge。
- json.dump默认allow_nan=True将NaN序列化为NaN字符串。

与发现描述的差异:
发现声称第一个不抛cv2.error的算法产出NaN时锁定best。当前dict顺序为TSAI->PARK->HORAUD->ANDREFF->DANIILIDIS，在测试的纯平移退化场景下TSAI始终收敛且返回有限值(虽然结果完全错误，位置分量达10^14米量级)，因此NaN lock在当前排序下未实际触发。相反TSAI的错误残差被compute_marker_consistency以std_mm=25.7mm>5.0正确拦截。但这取决于dict顺序：任何重排序、算法新增、或TSAI同样产生NaN的退化模式都会暴露NaN锁。

结论: 代码缺乏防御纵深(无np.isfinite().all()检查)，NaN可通过比选、质量门、写盘三道防线而不被拦截。虽然当前dict顺序提供了偶然缓解，但这是脆弱的；结构性修复应在calibrate_and_select返回前和save_cameras_json写盘前加入np.isfinite检查。

严重性确认为major: 旧文件有.bak备份可手工恢复，终端会打印NaN数值供肉眼发现，但依赖操作者察觉而非程序防御。
</details>

### `examples/real/calibrate_l515_depth.py:226` — 重连失败分支打印 "aborting phase 1 with partial data" 后必然崩溃，partial data 实际全部丢失
- 来源: calib_l515 | 验证置信: high

漂移循环里两次连续失败触发重连：224-228 行 disconnect() 后若 connect() 返回 False 则 break。但 break 只跳出漂移循环，控制流继续走到 261-262 行的 sigma 采集——此时 camera.pipeline 必为 None（disconnect 置 None，connect 所有失败路径都保证 pipeline 已撤销），capture_burst 第 99-100 行立即 raise RuntimeError("Camera pipeline is unavailable.")，无人捕获，叠加上一条发现（JSON 只在最后写），已采集的漂移 burst 全部丢失。该分支的行为与打印的"保留部分数据"意图确定性矛盾（100% 触发，非概率性）。应在 break 前落盘或让该函数带着已有 result 返回并跳过 sigma 采集。

<details><summary>验证记录</summary>

逐行追溯确认：

1) calibrate_l515_depth.py:224 camera.disconnect() → realsense.py:454-461 将 self.pipeline 置 None（无分支例外）。
2) calibrate_l515_depth.py:225 camera.connect() 返回 False → realsense.py:205-242 的五条 return False 路径（233/238/241）均保证 self.pipeline is None：
   - 218-219 行：pipeline 非 None 时 connect() 必定返回 True，所以 False 返回意味着 pipeline 已为 None。
   - _open_pipeline (244-278)：_start_pipeline 之后的所有失败路径（264/270/276）显式调用 disconnect() 置 pipeline=None；device discovery 失败（255-257）从未启动 pipeline，pipeline 保持 None。
   - _hardware_reset_and_wait (295-321)：307 行显式调用 disconnect()。
3) calibrate_l515_depth.py:228 break 跳出 for 循环后，控制流落至 238 行（result 字典构造 + 漂移后分析），然后到 261-262 行调用 capture_burst。
4) capture_burst:98-100 行 pipeline = camera.pipeline; if pipeline is None: raise RuntimeError("Camera pipeline is unavailable.")。
5) 该 RuntimeError 无人捕获——run_drift_phase (201 行起) 内部无 try/except，main() (362-375 行) 的 try/finally 只保证 camera.disconnect() 但不捕获异常。脚本级崩溃，377-379 行的 json.dump 永远不执行。
6) run_drift_phase:238 行构造的 result 字典（含已采集的 bursts 列表）和 main:366 行的 results["device"] 全部随进程崩溃丢失。

这与 226 行 print("reconnect failed — aborting phase 1 with partial data.") 声称保留部分数据的意图确定性矛盾——代码路径 100% 触发 crash，不存在概率性。
</details>

### `examples/real/calibrate_l515_depth.py:262` — sigma 采集与 A/B 阶段无异常保护，且 JSON 只在最后写盘——任何中途异常丢弃整场 25 分钟标定数据
- 来源: calib_l515 | 验证置信: high

run_drift_phase 的漂移循环专门为 L515 停流写了 try/except RuntimeError + 重连逻辑（211-230 行），但紧随其后的 sigma 采集（262 行 capture_burst(camera, args.sigma_frames, ...)）和 main 里的 run_ab_phase（372 行）完全无保护。capture_burst 内 pipeline.wait_for_frames(5000) 超时即抛 RuntimeError；已知 L515 直连根口也会中途无声停流（本仓库已记录该故障模式，脚本自己的漂移循环重连逻辑就是为此而写）。而 json.dump 在 377-379 行、位于 try/finally 之外的函数末尾——异常沿 run_drift_phase→main 传播后 finally 只做 disconnect，结果字典（含已跑完 25 分钟、且要求冷机 30 分钟才能重跑的漂移曲线）从未落盘。run_ab_phase 的 finally（306-307 行）在停流场景还会因 camera.pipeline.wait_for_frames(5000) 再次超时抛出新异常掩盖原始异常，结局相同。建议：sigma/AB 阶段包 try/except，并在每阶段结束后立即增量写 JSON。

<details><summary>验证记录</summary>

已验证全部 5 个独立子主张，均在代码中确认：

1. **sigma 采集无 try/except**：calibrate_l515_depth.py:262 `sigma_burst = capture_burst(...)` 在 for 循环（211-230 行）结束后、`return result`（276 行）之前，漂移循环的 try/except RuntimeError 只覆盖 216-230 行，不覆盖此行。

2. **capture_burst 会抛 RuntimeError**：三处路径确认 — (a) pipeline 为 None 时 99-100 行直接 raise RuntimeError（disconnect 在 realsense.py:461 将 pipeline 置 None）；(b) 102/114 行 `pipeline.wait_for_frames(5000)` 超时即抛 RuntimeError（同一调用在 realsense.py:424-426 被显式捕获，证实超时确会抛此异常）；(c) 152-153 行无可用深度帧时 raise。

3. **json.dump 不可达**：main() 的 362-375 行是 try/finally，finally 仅做 `camera.disconnect()`。异常从 run_drift_phase 或 run_ab_phase 抛出后，finally 执行完毕异常继续上抛，377-379 行的 `json.dump(results, ...)` 永远执行不到。所有结果字典（含 25 分钟漂移曲线的 bursts 列表 + total_drift_mm/t_settle_min/canary_mm 等汇总）丢弃。

4. **run_ab_phase finally 可二次抛异常**：306-307 行 `for _ in range(10): camera.pipeline.wait_for_frames(5000)` 在 finally 块中，停流超时会抛新 RuntimeError 掩盖原始异常（Python finally 抛异常的标准行为）。

5. **断连重连失败路径同样暴露**：漂移循环 222-227 行 reconnect 失败 break 后，camera.pipeline 为 None，262 行 sigma 采集必然在 99-100 行抛异常。

无任何缓解因素：无中间文件落盘、main() 无 except 子句、无析构函数副作用写 JSON。一个停流异常即可丢弃 25 分钟 + 要求冷机 30 分钟才能重跑的全部数据。
</details>

### `examples/real/replay_traj.py:638` — --arm-ip 未传入 ArmInnerLoop：非默认 IP 时内环连向硬编码的 192.168.1.111，最坏情况驱动另一台臂
- 来源: replay_traj | 验证置信: high

setup() 创建 `ArmInnerLoop(cfg=self._inner_cfg)` 未传 ip，ArmInnerLoop.__init__ 的默认 ip='192.168.1.111'（inner_loop.py:134-139）。CLI 明确提供 --arm-ip（line 888-891），传非默认值时：RobotInterface（connect/preflight/急停）连向指定 IP，而实际发运动命令的内环连向 .111——若 .111 恰好有另一台臂，回放命令会驱动错误的机器人而急停按钮打在另一台上；若 .111 无设备，则内环 error_state → 4 帧后误触发急停，功能完全不可用。_do_return_home 里重建的 `ArmInnerLoop(cfg=_INNER_CFG)`（line 555）同样问题；`_make_planner(arm_ip)` 的 arm_ip 参数也从未使用（line 503-507）。

<details><summary>验证记录</summary>


**确认属实，证据如下：**

1. `replay_traj.py:638` — `self._arm_inner = ArmInnerLoop(cfg=self._inner_cfg)` 未传 `ip=self.arm_ip`。`TrajectoryReplayer.__init__` 已在 line 588 正确存储 `self.arm_ip = arm_ip`，但 `setup()` 在创建 `ArmInnerLoop` 时遗漏了该参数。`ArmInnerLoop.__init__` 的默认值为 `ip="192.168.1.111"`（`inner_loop.py:134`），且 `ArmInnerLoopConfig`（`inner_loop.py:47-96`）不包含 IP 字段——IP 是 `ArmInnerLoop` 构造函数的独立参数。对照 `_make_robot`（line 526-537）：该函数正确将 `arm_ip` 传入 `XArm7Config(ip=arm_ip)` → `RobotInterface`，因此 RobotInterface 连向正确 IP 而内环连向硬编码 IP。

2. `replay_traj.py:555` — `_do_return_home` 中 `new_inner = ArmInnerLoop(cfg=_INNER_CFG)` 同样未传 IP。`_do_return_home` 的签名甚至没有 `arm_ip` 参数（line 540-544），`_INNER_CFG` 仅是一个 `ArmInnerLoopConfig()` 实例（line 75），不含 IP。不过实际影响有限：该路径仅在 post-loop "press H to return home"（line 1006-1014）触发，此时主回放循环已结束，`shutdown()` 已在 line 994 调用，新创建的内环在 `r.disconnect()` 后立即被丢弃，不会实际发送任何运动命令。

3. `replay_traj.py:503-507` — `_make_planner(arm_ip: str)` 接受 `arm_ip` 参数但函数体内从未使用。`XArm7MotionPlanner` 构造不接收 IP 参数，`arm_ip` 在此处确实是死参数。

**后果验证（非默认 IP 场景）：**

- `_run()` 中 `XArmAPI(self._ip)`（`inner_loop.py:251`）使用默认 `"192.168.1.111"`。
  - 若 `.111` 无设备：抛出异常 → `_error_state = True` → 线程立即返回 → `_ready_event` 永不被 set → `wait_ready` 超时打印"timed out, falling back to direct read"（line 643）→ `run()` 中 `get_state()` 返回 `error_state=True`（line 724-725）→ 4 帧后 `error_count > 3` 触发 `_emergency_stop()`（line 728-731）→ 回放完全不可用。
  - 若 `.111` 恰好有另一台臂：内环正常启动并以 `set_servo_angle` 驱动该臂（`inner_loop.py:342`），而 RobotInterface 的急停打在正确 IP 的臂上——这是真正的危险场景。

- `_emergency_stop()`（line 810-818）调用 `self.robot.emergency_stop()`，`self.robot` 通过 `_make_robot` 使用了正确的 `arm_ip`，因此急停打在正确的机器人上，但内环发令的对象是另一台（`.111` 那台）。

**严重性评估：** finding 标记为 major 是恰当的。最坏情况（驱动另一台臂）可升级为 critical，但该场景需要 `.111` 恰好有另一台 xArm，依赖于特定的网络拓扑巧合。主要影响是非默认 IP 时回放功能完全不可用，符合 major 定义。

</details>

### `examples/real/replay_traj.py:684` — 回放启动时把 HDF5 首帧关节目标直发内环——无接近性检查、无规划、无操作员确认
- 来源: x-safety | 验证置信: high

run() 在进入主循环前直接把 episode 第 0 帧关节位姿发给 ArmInnerLoop，臂会立即从当前任意位姿沿未经碰撞检查的关节直线插值扫向录制起点（固件 Mode 6 限速 90°/s + soft ramp，但完全没有 plan_path/碰撞检查，也没有『臂即将移动到起始位姿』的确认提示或起点接近性校验）。若回放时臂不在录制起点附近（如用 --max-frames 截断后半段、或臂停在别处），这段过渡可能扫过桌面/障碍。CLAUDE.md 要求复位类动作『经规划或限速』，此处仅有限速无规划。对比 RobotInterface.return_to_home()（interface.py:275）是 Tier1 plan_path + Tier2 碰撞检查插值，本文件的归位反而走了规划路径，唯独 jump-to-start 没有。

<details><summary>验证记录</summary>

逐项核实：

1) **replay_traj.py:671-684** — run() 在 672 行读取当前 arm_qpos（`arm_qpos, error_state, _ = self._arm_inner.get_state()`），675 行构造完整 RobotState，但 **683-684 行直接 `first_cmd = self.traj.action_arm_joint[0].copy(); self._arm_inner.set_target(first_cmd)`**。当前位姿与 first_cmd 之间无任何距离比较，读取到的 arm_qpos **仅用于传入 get_state() 构造 state，从未与 first_cmd 做接近性校验**。无操作员确认提示。

2) **inner_loop.py:169-175** — set_target() 只是 Lock 保护的纯赋值，无安全检查。

3) **inner_loop.py:488-529** — _send_target() 首次调用时 `_last_sent_target is None`（line 503），**delta clamp 被完全跳过**。后续内环循环发送同一静态目标时 delta=0，clamp 也不生效。唯一约束是 speed ramp（line 511-518，从 0.2 rad/s ~11°/s 起，0.4s 内逐步升至 90°/s）和固件限速/限加速（90°/s, 500°/s²）。**无碰撞检测**——固件 Mode 6 仅做关节空间插值。

4) **interface.py:275-344** — return_to_home() 使用 Tier1 plan_path（完整碰撞检测）+ Tier2 关节插值碰撞检测，与本文件 jump-to-start 形成明确对比，不对称性属实。

5) **load_trajectory:157** — `action_arm_joint[:T]` 始终从索引 0 截取，`--max-frames` 只影响帧数不改变起始帧，发现的“截断后半段”描述有微误，但不影响核心结论（臂停在别处即触发问题）。

结论：核心发现成立——回放启动时将首帧关节目标直发 ArmInnerLoop，无接近性检查、无碰撞规划、无操作员确认。臂会从未知位置沿固件 Mode 6 关节空间插值扫向录制起点，仅有限速无规划，与 CLAUDE.md 要求（复位类动作经规划或限速）形成安全缺口。
</details>

### `examples/real/replay_traj.py:764` — 全程未调用 validate_action：力矩/温度门、臂关节限位裁剪、工作区检查在回放路径全部缺失
- 来源: replay_traj | 验证置信: high

回放循环每帧直接 `self._arm_inner.set_target(arm_cmd)` + `self.robot.send_action(action)`，从未调用 robot/validate.py 的 validate_action()，也从未调用 _arm_inner.get_dynamics() 获取 tau/temps。这意味着：(1) 力矩门与温度门完全失效——回放场景恰恰是环境可能与录制时不同（物体被移动）的场景，撞击时只剩灵敏度最低(=1)的固件碰撞检测兜底；(2) 臂命令没有 qpos_min_soft/max_soft 软限位裁剪（ArmInnerLoop._send_target 只做 delta clamp 不做限位 clip），越界命令会直接触发固件 reduced-mode 故障而急停。CLAUDE.md 反模式明列 'Ignoring validate_action() → always call before send_action()'，生产控制器每帧都调用。

<details><summary>验证记录</summary>

逐项核实结果：

**核心事实确认：**
1. `replay_traj.py:764` `self._arm_inner.set_target(arm_cmd)` + `:778` `self.robot.send_action(action)` 之间无任何 `validate_action()` 调用。
2. `replay_traj.py:724` 仅调用 `self._arm_inner.get_state()`，从未调用 `_arm_inner.get_dynamics()`。对照生产控制器 `controller.py:308` 显式调用 `get_dynamics()` 获取 tau/temps。
3. `replay_traj.py:733` `self.robot.get_state(arm_qpos=arm_qpos)` 未传入 `arm_qvel`/`arm_tau`，导致 `interface.py:181-182` 将 `arm_tau` 设为 `nan_array(7)`（全 NaN）。即使补上 `validate_action()` 调用，若无 `get_dynamics()` 数据，力矩门（`validate.py:54-58` 要求 `np.all(np.isfinite(tau))`）和温度门（`validate.py:62-67`）也均为 no-op。
4. 臂关节软限位裁剪缺失：`validate.py:84-86` 的 `np.clip(action.arm_qpos_cmd, arm_lo, arm_hi)` 在回放路径无对应。`inner_loop.py:498-506` `_send_target` 仅做 delta clamp，不做绝对限位 clip。
5. 碰撞灵敏度：`xarm7.py:73` 默认值=1，`inner_loop.py:263` 启动时设 `arm.set_collision_sensitivity(1)`。回放路径未覆盖此值，碰撞保护仅靠最低灵敏度固件检测。

**发现中的过度陈述：**
- "越界命令会直接触发固件 reduced-mode 故障而急停" — 过度。HDF5 命令来自已验证的录制会话，本身应在限位内；Mode 6 固件轨迹规划对硬件限位内目标正常处理。软限位（硬件限位内缩 2.5°，`xarm7.py:33`）缺失是安全裕度问题而非直接故障源。
- 手部限位裁剪缺失 — `XHand.send_action`（`xhand.py:520`）自带 `_limit_joint_range`（`xhand.py:805` `np.clip(qpos, qpos_min, qpos_max)`），手部命令已有独立保护，非完全缺失。

**结论：** 发现核心属实。力矩/温度门缺失是最重要的现实风险——回放场景下环境可能变化，仅靠 sensitivity=1 固件碰撞检测兜底。严重性维持 major。
</details>

### `examples/real/replay_traj.py:776` — --no-hand 或臂-only episode 时仍每帧向已连接的手发送 zeros(12) 命令，手会被物理驱动到近零姿态
- 来源: replay_traj | 验证置信: high

循环中 else 分支构造 `RobotAction(hand_qpos_cmd=np.zeros(12))` 并无条件 `robot.send_action(action)`。RobotInterface.send_action 只做手命令转发（interface.py:241 `self.hand.send_action(action.hand_qpos_cmd)`），XHand.send_action 在手已连接时会真实下发：zeros 被 clip 到 qpos_min（thumb_j2 min=10°、index/middle/ring/little_j2 min=5°、thumb_j1=0° 而非 home 的 45°），即手被以 E3 delta clip 0.3 rad/次的速率驱动到握零姿态。这直接违反 `--no-hand` 的语义（help: 'Skip hand commands even if hand data is present'），也让臂-only 回放意外动手。正确做法：hand 不参与时完全不调用 send_action（手未连接时无害，因 XHand.send_action 直接 return False，但该分支同样多余）。

<details><summary>验证记录</summary>

逐文件逐行追溯确认：

1. replay_traj.py:587 — no_hand=True 由 --no-hand 传入。
2. replay_traj.py:621 — self.robot.connect() 无条件同时连接臂和手（interface.py:103-114），无论 --no-hand。
3. replay_traj.py:629 — _hand_available = bool(result.get("hand")) and not self.no_hand 得 False。
4. replay_traj.py:714 — 因 _hand_available=False，hand_cmd 保持 None。
5. replay_traj.py:768 — guard 失败，进入 else。
6. replay_traj.py:773-777 — 构造 RobotAction(hand_qpos_cmd=np.zeros(12, dtype=np.float64))。
7. replay_traj.py:778 — self.robot.send_action(action) 无条件调用。
8. interface.py:241 — 无条件 self.hand.send_action(action.hand_qpos_cmd)。
9. xhand.py:516-517 — 手已连接时 guard 通过（control/hand_command 非 None）。
10. xhand.py:519-520 — _limit_joint_range(zeros(12)) 被 clip 到 qpos_min（xhand.py:117-137: thumb_j1=0deg, thumb_j2=10deg, index/middle/ring/little_j2=5deg）。
11. xhand.py:526-530 — delta clip，max_delta_rad=0.3（xhand.py:207）。
12. xhand.py:537-538 — write_command_positions + control.send_command 真实下发硬件。

手已连接+--no-hand 时，手被物理驱动到 qpos_min 近零姿态（home_qpos 的 thumb_j1=45deg 被驱至 0deg），每步最多 0.3 rad。完全违反 --no-hand 的语义（help: "Skip hand commands even if hand data is present"）。严重性 major 合理：用户明确要求不动手，代码却动手。若手未连接则无害（xhand.py:516 返回 False），但代码仍每帧进入无用的 send 路径。
</details>

### `examples/real/replay_traj.py:806` — run() 的 finally 块中 return 吞掉一切在途异常（含 KeyboardInterrupt），故障回放被伪装成正常完成
- 来源: replay_traj | 验证置信: high

finally 里的 `return self._recorder.to_dict()`（line 806）会丢弃任何正在传播的异常：Ctrl-C 时 KeyboardInterrupt 被静默吞掉，main 的 `except KeyboardInterrupt`（line 991）永远不会命中，'Interrupted by user' 不打印，流程按'回放成功'继续计算 metrics 并保存；更严重的是循环内任何未预期的运行时异常（如 send_action/get_state 抛出未被内层 except 覆盖的错误）也被吞掉，无任何 traceback，部分数据被当成完整回放写入 metrics.json。真机操作脚本必须区分正常完成/中断/崩溃三种结束方式。修法：把 return 移到 try/finally 之外（finally 只做 kb.stop + hold）。

<details><summary>验证记录</summary>

核实确认该发现完全成立。关键代码位置：
1. replay_traj.py:693-806 — run() 的 try/finally 结构中，finally 块（line 795-806）以 `return self._recorder.to_dict()` 结尾。Python 语义明确规定：finally 中的 return/break/continue 会丢弃 try 块中正在传播的一切异常。KeyboardInterrupt 继承自 BaseException 而非 Exception，循环内的 `except Exception`（line 734）不会捕获它。
2. replay_traj.py:764,778,695 — 循环内 set_target、send_action、rate_mgr.wait 等调用均无 try/except 保护，任何未预期异常同样被 finally 吞掉。
3. replay_traj.py:808 — `return None` 在 finally 之后但因 finally 在所有非 dry-run 路径均 return，此行实际不可达（死代码佐证）。
4. replay_traj.py:991-992 — main() 的 `except KeyboardInterrupt: print('\nInterrupted by user')` 对 run() 内发生的 Ctrl-C 永远不会命中，是死代码。
5. keyboard.py:37-44 — KeyboardHandler 的 _KEY_MAP 不包含 Ctrl-C 对应的 KeyCode(char='\x03')，因此 Ctrl-C 不会被转成 ControlSignal，SIGINT 仍送达主线程。
实际后果：Ctrl-C 中断后无"中断"提示，部分数据被当成完整回放计算指标并保存。
</details>

### `examples/real/test_motion_planning_real.py:164` — 真机执行路径无任何桌面/工作区/环境防护：低位 waypoint + ±30° 随机旋转可能使指尖撞桌
- 来源: mp_real | 验证置信: high | 原评级: critical

create_planner() 构造 XArm7PlannerConfig 时既不传 collision=CollisionConfig(...) 也不传 workspace_bounds，导致三层 Z 向防护全部失效：(1) planner.desk_safety=None（planner.py:119-127 仅在 config.collision 非 None 时构造 FingertipDeskSafety，_check_desk_safety 于 planner.py:572-573 直接跳过）；(2) workspace_safety=None（planner.py:91-95，_check_workspace_bounds 于 planner.py:542-543 跳过）；(3) 桌面 FCL 障碍从未注册——add_table 只在 RobotInterface 里调用（interface.py:74-84），本脚本绕过 RobotInterface，collision_model._obstacle_names 为空，check_env_collision 无条件返回 False（collision_model.py:471-472），行 546-548 打印的"碰撞检测验证"里 env 检查是空转。而 Test 4 真机执行的目标由 run_waypoint_test 行 521 传入 rng，叠加最大 30° 随机旋转（RANDOM_ROT_DEG），SAFE_WAYPOINTS 含 z=0.13/0.15 m 低位点；生产配置（vr_teleop_shm.py:102-106）表明桌面在 z=0、指尖在 home 手姿下低于 EEF 0.076 m、安全裕量 0.03 m——z=0.13 处 home 姿态余量仅 0.054 m，30° 不利倾转即可逼近/突破 0.03 m 裕量；且 RRT 中间构型的 EEF/指尖高度完全不受检（只查自碰撞），路径可下探到端点 z 以下。固件侧唯一兜底是 collision_sensitivity=1（xarm7.py:73，最不灵敏）。同一 PlanningProfile 还把 max_waypoint_delta_deg=360、max_ik_delta_deg=(180,)*7 放开，进一步扩大低位大摆幅路径的搜索空间。建议与生产入口对齐传入 CollisionConfig（+可选 workspace_bounds），或至少去掉真机执行目标上的随机旋转。

<details><summary>验证记录</summary>

核实确认三项防护全部失效：(1) desk_safety=None 因 test_motion_planning_real.py:164-185 创建 XArm7PlannerConfig 时未传 collision=CollisionConfig(...)，planner.py:119-127 仅在 config.collision 非 None 时构造 FingertipDeskSafety，_check_desk_safety 在 planner.py:571-573 直接跳过；(2) workspace_safety=None 同理，planner.py:91-95 仅在 config.workspace_bounds 非 None 时构造，_check_workspace_bounds 于 planner.py:541-543 跳过；(3) env_collision 空转：collision_model.py:237 _obstacle_names 初始化为空 set，add_table 仅在 RobotInterface.__init__ (interface.py:74-84) 调用，本脚本绕过 RobotInterface 直接使用 XArm7，故 check_env_collision (collision_model.py:471-472) 因 _obstacle_names 为空无条件返回 False。几何风险真实：SAFE_WAYPOINTS 含 z=0.13/0.15m 低位点，test 行 521 叠加 RANDOM_ROT_DEG=30deg 随机单轴旋转，home 手姿指尖低于 EEF 0.076m，宽松的 max_waypoint_delta_deg=360 和 max_ik_delta_deg=(180,)*7 使 RRT 可生成低位大幅摆幅路径；唯一兜底为 xarm7.py:73 collision_sensitivity=1（最不灵敏）。从 critical 降为 major 理由：该脚本为人工监督测试而非生产入口，操作员在场；桌面为平面，最坏情况为轻触触发固件碰撞停机而非灾难性损坏。
</details>

### `examples/real/test_motion_planning_real.py:814` — Test 5 碰撞-hold 判定字符串不匹配（self_collision vs self-collision），主路径永远无法确认 C3 修复生效
- 来源: mp_real | 验证置信: high

test_teleop_ik_collision 用 `"self_collision" in (r.reason or "")` 判定碰撞被 hold，但被测的 command_from_target_qpos 碰撞门（即本测试 docstring 行 780-781 点名要验证的 C3 修复路径）返回的 reason 是连字符形式：ik.py _check_teleop_collision_gate 返回 "IK result in self-collision ({info.summary}), holding."（ik.py:369-376），"self_collision"（下划线）不是它的子串。下划线形式只出现在两级 IK 全失败后的诊断摘要 "Diff IK [self_collision]: ..."（ik.py:285-287, 345-351），而 solve_position_ik 不做碰撞过滤（ik.py:131-227，只滤 pose_err/jump/elbow_flip），碰撞候选照常返回并在碰撞门被 hold（连字符 reason）——即最典型的 hold 结果恰好匹配不上。由于 held 结果 success=False，行 819-821 的 `elif not r.success: pass` 把真正的 hold 静默吞掉，测试落入 "not_applicable" 并打印"✅ available (no colliding target found)"，且行 835 的 ok 判定不含 collision_held——回归验证被字符串失配悄悄架空。应改为匹配 "self-collision" 或直接检查 r.held + report["collision"]。

<details><summary>验证记录</summary>

已通过完整代码追溯确认。

**碰撞门产出的 reason 字符串**（ik.py:374）：
```python
f"IK result in self-collision ({info.summary}), holding."
```
使用连字符 "self-collision"。

**测试检查的字符串**（test_motion_planning_real.py:814）：
```python
if r.held and "self_collision" in (r.reason or ""):
```
使用下划线 "self_collision"。

**主路径调用链**：solve()(ik.py:65) → solve_differential_ik(ik.py:83) → command_from_target_qpos(ik.py:601) → _check_teleop_collision_gate(ik.py:433) → 返回 IKResult(success=False, held=True, reason="IK result in self-collision (...), holding.")(ik.py:466-477)。diff IK 失败后走 position IK fallback(ik.py:102-115)，solve_position_ik(ik.py:131-225)不做碰撞过滤（仅滤 pose_err/jump/elbow_flip），碰撞候选照常返回，再次经 command_from_target_qpos 的同一碰撞门被 hold，reason 仍是连字符形式。

**唯一能匹配的路径**：两级 IK 全失败后的诊断摘要(ik.py:117-126)。_build_ik_diagnostic(ik.py:255-351)在第286-287行将 reason_lower 中的 "self-collision"（连字符）归类为 category="self_collision"（下划线），然后摘要拼入 `f"Diff IK [self_collision]: ..."`(ik.py:347)，此时下划线形式才出现。但此路径要求 solve_position_ik 返回 None(ik.py:225)，对于从有效关节构型 FK 导出的可达目标 EEF 而言不常见。

**后果确认**：测试行819-821的 `elif not r.success: pass` 会静默吞掉主路径的 held 结果，collision_detected 保持 False，落入行825-827的 "not_applicable" 分支并打印 "available (no colliding target found)"。行835的 ok 判定不含 collision_held 条件，测试在未实际验证 C3 hold 行为的情况下"通过"。
</details>

### `examples/real/test_motion_planning_real.py:865` — 绕过 RobotInterface 直接实例化 XArm7 驱动真实运动，且全程无键盘急停路径
- 来源: x-safety | 验证置信: high

该脚本直接 `arm = XArm7(arm_config)` 并用 arm.send_action()（execute_path_on_arm:476 以 30Hz 执行稠密路径）、arm.reset()、incremental_motion_check（408 行给全部 7 关节 +2°）驱动真实硬件——不是纯读/标定脚本，不符合例外条件，绕过了 RobotInterface 的 workspace/validate 体系（自带 plan_path + 碰撞校验部分缓解）。更重要的是全文件无 KeyboardHandler/ESC：执行 10 个 waypoint 期间唯一的中止手段是 Ctrl-C，而 Ctrl-C 后 finally(972) 只调 arm.disconnect()（xarm7.py:135-138 仅关连接）不调 arm.stop()（set_state(4)），臂会继续完成固件中最后一条指令。任务要求『急停路径必须可达』在此入口不成立。

<details><summary>验证记录</summary>

所有核心发现均已通过代码逐层追溯确认：

1. 绕过 RobotInterface 直接实例化 XArm7 -- 确认。
   - test_motion_planning_real.py:865 arm = XArm7(arm_config)
   - 导入自 dexmani_real.robot.xarm7 (line 39)，非 RobotInterface
   - CLAUDE.md 明确将此列为 Anti-pattern: "Hardware access: ONLY via RobotInterface; never call XArm7/XHand directly"
   - 但需注意: 这是一个开发测试脚本(非生产入口)，且自带多层安全(plan_path 碰撞检测, 1deg插值, preflight_check, arm.is_error() 逐路点检查)

2. 全程无键盘急停路径 -- 确认。
   - grep 全文无 KeyboardHandler/keyboard/pynput/ESC/signal/SIGINT 匹配
   - main() 的 try/finally 块 (line 871-973) 无 except 子句，仅 finally 做 cleanup

3. disconnect() 不调用 stop()，stop() 在任何 Ctrl-C 退出路径均不被调用 -- 确认。
   - xarm7.py:135-138 disconnect() 仅调 self.arm.disconnect() + connected_flag = False
   - SDK 层 xarm_api.py:786-790 disconnect() 仅调 self._arm.disconnect()
   - SDK 底层 base.py:1131-1159 仅关闭 TCP stream (self._stream.close())，无 set_state(4) 调用
   - xarm7.py:185-190 stop() 调 self.arm.set_state(4) (急停状态)，但 main() 的 finally 块 (line 972-973) 只调 arm.disconnect()，不调 arm.stop()
   - stop() 仅在 _fallback_reset() (line 685) 内部恢复路径被调用，非用户中断退出路径

4. 关于"臂会继续完成固件中最后一条指令" -- 需细化。
   - send_action (xarm7.py:235-281) 使用 Mode 1 (servo) + set_servo_angle_j
   - SDK 文档明确: "execute only the last instruction" -- Mode 1 是流式伺服，每个新指令替换前一个，不存在指令队列
   - 路径已被 interpolate_waypoints 以 1deg 步长插值 (INTERP_MAX_STEP_RAD = deg2rad(1.0), line 121)
   - Ctrl-C 断开连接后，臂最多继续追踪最后一次 set_servo_angle_j 的目标(距离上一个路点 max 1deg 关节运动)
   - xArm 固件通常有通信看门狗(~1-2s 超时)，之后会自停
   - 但关键在于: 代码层面没有任何显式 stop 指令，依赖的是固件看门狗的未文档化行为，不是可靠的工程保障

综上所述，发现的核心结构性问题属实(绕过 RobotInterface, 缺失显式急停路径, disconnect 不调 stop)，但"完成最后一条指令"的风险描述过度 -- Mode 1 伺服模式不存在指令缓冲队列，实际无控运动最大幅度为 1deg 关节偏差。严重性维持 major，考虑到(a)项目自身将直接调用 XArm7 列为反模式，(b)真实硬件运动时缺失显式急停是工程缺陷而非纯风格问题。
</details>

### `examples/real/test_pointcloud_stream.py:118` — 退出判据缺失第三条通过标准：点云路径完全失效（标定解析失败、全程空云）时汇总仍打印 PASS/PASS
- 来源: pointcloud | 验证置信: high

docstring（行 9-12）声明三条通过标准，第三条是 '2048 valid points inside the workspace crop box'，但退出汇总（行 111-123）只检查帧率 >=28Hz 和延迟 p95<80ms，从不校验 pointcloud_valid / 有效点数。已核对 camera_process.py:_build_processor（行 427-453）：cameras.json 缺序列号条目或 eye-in-hand 时子进程静默禁用点云，rgb/depth 照常采集，SHM 槽持续发全零 + pc_num_points=0 → 消费端 frame['pointcloud'] 为全零数组、pointcloud_valid=False（layouts.py:173-175）。此时帧率和延迟均正常，测试打印两行 PASS——这个以'验证生产点云路径端到端'为唯一目的的冒烟测试，在点云路径彻底死亡时给出绿灯。逐秒打印里的 'pc valid=False' 依赖人工注意，不进入判据。

<details><summary>验证记录</summary>

经过完整代码链路追溯，该发现准确描述了问题：

1. **Docstring 声明的第三条通过标准未被实现**：`examples/real/test_pointcloud_stream.py` 第 9-12 行 docstring 明确列出三条 pass criteria，其中第三条为 "2048 valid points inside the workspace crop box"，但退出汇总（第 118-123 行）仅检查 `rate >= 28 Hz` 和 `latency < 80 ms`，完全未涉及 pointcloud_valid 或有效点数。

2. **点云路径静默死亡时仍打印 PASS/PASS**：当 cameras.json 缺少序列号条目或配置为 eye-in-hand 时，`dexmani_real/sensor/camera_process.py:_build_processor`（第 448-453 行）捕获 KeyError/ValueError 后 return None，子进程继续以 30Hz 采集 rgb/depth 并向 SHM 写入全零 pointcloud + pc_num_points=0。消费端 `dexmani_real/shm/layouts.py:bytes_to_camera_frame`（第 173-175 行）据此设置 `frame["pointcloud_valid"] = False`。此时帧率和延迟均正常，汇总打印两行 PASS——该冒烟测试的唯一目的是"验证生产点云路径端到端"（第 3 行），却在点云路径彻底死亡时给出绿灯。

3. **逐秒打印不足以替代判据**：测试第 91 行的逐秒输出虽打印 `pc valid=False`，但该信息未被聚合到退出汇总中。滚动输出依赖人工注意，不与自动化 CI 或批处理流程兼容。

4. **无上游兜底**：`CameraProcess.start()`（第 96-113 行）仅启动子进程并返回 True，不检查 `_build_processor` 是否成功；`CameraProcess` 类未暴露任何 `pointcloud_enabled` 属性供消费端查询运行时状态。

结论：缺陷确认存在，产生所述后果。
</details>

### `examples/real/vr_teleop_arm_only.py:551` — 录制的 /arm_qvel、/arm_tau 全为 NaN：get_state(arm_qpos=...) 未配套传入 get_dynamics()
- 来源: arm_only | 验证置信: high

主循环用 arm_inner.get_state() 取 qpos 后调 robot.get_state(arm_qpos=arm_qpos)。RobotInterface.get_state 在传入 arm_qpos 时会跳过 SDK 读取，arm_qvel/arm_tau 若未显式传入则填 nan_array(7)。本脚本从不调用 arm_inner.get_dynamics()，因此每个 episode 的 /arm_qvel(T,7) 和 /arm_tau(T,7) 数据集全部是 NaN——schema 字段存在但内容无效，下游依赖动力学流（力矩分析、异常检测）的消费者拿到空数据。16Hz 版 record_plus 已修（get_dynamics → get_state 三参）。

<details><summary>验证记录</summary>

逐层核实确认问题真实存在：

1. **数据源缺失**：`vr_teleop_arm_only.py` L540 仅调用 `arm_inner.get_state()` 取 `arm_qpos`，该脚本全文无任何 `get_dynamics()` 调用（grep 零结果）。

2. **NaN 填充逻辑**：`interface.py` L180-182 明确设计——当 `arm_qpos` 传入但 `arm_qvel`/`arm_tau` 为 None 时填充 `nan_array(7)`。docstring L175-178 记载了此行为。

3. **无保护写入 HDF5**：`episode_recorder.py` L310-311 直接写 `state.arm_qvel` 和 `state.arm_tau` 到 HDF5，无 NaN 检查或兜底逻辑。主循环 L728-729 `recorder.add_frame(state, action, vr_frame)` 将含 NaN 的 state 送入录制器。

4. **对照修复版已修**：`vr_teleop_arm_only_record_plus.py` L666-667 正确调用 `arm_inner.get_dynamics()` 并三参传入 `robot.get_state(arm_qpos=, arm_qvel=, arm_tau=)`。

5. **无替代路径**：脚本内所有 `robot.get_state()` 调用（L297、L452、L551）均仅传 `arm_qpos=`，无参调用仅在 L569 错误恢复分支（不走录制流程）。

后果：每个 episode 的 `/arm_qvel(T,7)` 和 `/arm_tau(T,7)` 数据集全为 NaN，schema 字段存在但内容无效，下游力矩分析、异常检测等消费者拿到空数据。CLAUDE.md 将该脚本列为正式入口点，实际用户会受影响。严重性 major 合理——数据静默损坏而非崩溃，但 arm_only 是文档化入口点。
</details>

### `examples/real/vr_teleop_arm_only.py:566` — A 对 C22 错误恢复无上限计数器，持久性硬件故障可导致无限循环
- 来源: x-triplet | 验证置信: high

arm_only.py 在检测到 C22 (C31/C32 自碰撞) 错误时，仅 clear_error + continue，无任何计数器或上限。若 C22 持续出现（例如真实的机械卡死），A 将在 50Hz 下无限循环清错。record.py (L642-645) 和 record_plus.py (L687-690) 均有 recover_count 计数器，连续超过 5 次后触发 _emergency_stop 退出。B/C 是对的，A 应补齐上限。

<details><summary>验证记录</summary>

发现属实，且实际后果比转述更严重。核实链条：(1) examples/real/vr_teleop_arm_only.py L566-574 确如发现所述——C22 分支仅 clear_error() + error_count=0 + continue，无恢复计数器；而同仓库 record.py L399/L641-645 与 record_plus.py L420/L686-690 均有 recover_count > 5 → _emergency_stop() 兜底，证明团队已认定该场景需要上限。(2) 关键的是外环 L542-549 的 error_state → error_count>3 急停兜底对 C22 不生效：dexmani_real/robot/inner_loop.py L105-108 定义 _RECOVERABLE_ERRORS={22,24}，L374-387 与 L546-563 对 C22 明确"Do NOT set error_state=True"，注释写明恢复责任完全交给外环（"The outer loop will detect the error via robot.arm.is_error(), clear the latch"）——而 arm_only.py 的外环恰恰没有上限。(3) 后果成立：dexmani_real/robot/xarm7/xarm7.py L154-183 的 clear_error() 每次执行 motion_enable(True)+set_state(0)，docstring 明确 C31/C32 碰撞后 motion_enable 会重新使能电机；持久性机械卡死时该循环以 50Hz 反复重新使能电机压向障碍物。clear_error 的 post-check（L174-180）失败返回 False，但 arm_only.py L568 忽略返回值，下一轮 error_code==22 仍走同一分支，无任何自动退出路径。(4) 频率核实：arm_only.py L141 CTRL_DT=0.02 即 50Hz，发现表述正确（该入口未迁移 16Hz）。唯一缓解：循环顶部 kb.poll() 仍响应 ESC 可人工急停，且 arm_only.py 非主生产入口（主入口为 vr_teleop_shm.py），但人工干预依赖不构成反驳。作为真实硬件入口的安全兜底缺失，major 评级恰当。
</details>

### `examples/real/vr_teleop_arm_only.py:566` — A 对 C24 速度超限错误无恢复处理，直接触发急停
- 来源: x-triplet | 验证置信: high

arm_only.py L566 仅检查 C22 错误码 (`arm_code == 22`)。当 xArm 固件返回 C24（速度超限，通常是 IK 尖峰导致的运动中低速上限，见 CLAUDE.md 记忆 "C24 Ramp Reset Mid-motion"），A 会走到 L575-576 的通用错误分支，直接调用 _emergency_stop()——而 C24 是可恢复的。record.py (L639) 和 record_plus.py (L684) 均通过 `arm_code in (22, 24)` 将 C24 纳入可恢复处理。B/C 是对的。

<details><summary>验证记录</summary>

发现属实，我逐层核实了完整调用链。(1) examples/real/vr_teleop_arm_only.py L562-577（工作区与 HEAD 一致）：`robot.arm.is_error()` 为真时仅 `arm_code == 22 or sdk_code == 22` 走清错+保持位置分支，C24 落入 L575-576 通用分支直接 `_emergency_stop()`。(2) 该路径对 C24 确实可达：dexmani_real/robot/inner_loop.py L105-108 定义 `_RECOVERABLE_ERRORS = frozenset({22, 24})`，两处检测点（L372-387、L546-563）对 C24 刻意不设 `_error_state=True` 只 hold 位置，L550-551 注释明确契约"The outer loop will detect the error via robot.arm.is_error(), clear the latch, and supply a fresh IK solution"——因此 arm_only.py L542 的 error_state 检查挡不住 C24，必然到达 L562；xarm7.py L143-152 `is_error()` 在 `arm.error_code != 0` 时返回 True。即内环把 C24 当可恢复错误放行等外环清错，外环却将其升级为急停，违反库层契约。(3) 后果核实：`_emergency_stop()`（arm_only.py L406-420）丢弃进行中 episode（`stop_episode(success=False)`）、停内环、`running=False` 整个程序退出。(4) 对照核实：record.py L639、record_plus.py L684 均为 `arm_code in (22, 24)` 且打印标注 24="速度超限"，配 recover_count≤5 上限；git log -S 显示 (22,24) 是最新提交 5632784(0716) 加入的——在 C24 根因修复后团队仍保留该恢复路径，说明 C24 仍被视为现实可发生事件。(5) 排除"死代码"反驳：arm_only.py 是 CLAUDE.md 列出的正式入口点，且在 0715/0716 两次提交中仍被修改。唯一减轻因素是 C24 根因（ramp 重置）已修、发生率降低，且急停方向 fail-safe 无物理危险，但"可恢复错误→整个会话终止+episode 数据丢弃"在活跃维护的入口点上与显式库契约相悖，major 评级恰当。附注：修复时建议同时补 recover_count 上限（arm_only.py 现有 C22 分支也缺此上限，连续错误会无限清错重试）。
</details>

### `examples/real/vr_teleop_arm_only.py:648` — C 暂停期间不 add_frame：'录制继续'提示为假，恢复时栅格用恢复后首帧批量回填暂停窗口并消耗 90s 墙钟预算
- 来源: arm_only | 验证置信: high

C 键提示『录制继续』（L475），但 hold 分支（L639-648）在 add_frame（L729）之前 continue，暂停期间一帧都不会写入。TimestampAlignedBuffer 按 (ts-anchor)/dt 分配槽位并用『下一到达样本』回填全部缺口（n_repeats 回填），因此恢复后第一帧会把整个暂停窗口填成同一份恢复时刻数据（vr_wrist 为恢复后新基准的 wrist，timestamp 全部为同一个值）；同时缓冲容量 max_frames+100 是墙钟预算（4500@50Hz≈90s），暂停时长直接吃掉录制额度——暂停 60s 的 episode 只剩 ~30s 有效数据即触发 max_frames 自动停止。要么暂停期间继续 add_frame（held 帧），要么提示改为『录制暂停』并停表。

<details><summary>验证记录</summary>

逐条核实，全部成立：

1) C 暂停期间不 add_frame：vr_teleop_arm_only.py:639-648 当 teleop_active=False 时无条件执行 continue（L648），跳过 L729 的唯一 add_frame 调用。grep 确认该文件中 add_frame 仅出现在 L729。

2) "录制继续"提示为假：L472-475 PAUSE 处理只翻转 teleop_active，不动 recording_active（B 设 True @L499，仅 S/Q/ESC 清除）。teleop_active=False 时 recording_active 仍是 True，故打印"录制继续"——但循环体 L639 的 continue 保证没有任何帧被写入。

3) 恢复后批量回填：timestamp_buffer.py:59-67 get_accumulate_timestamp_idxs 中 n_repeats = max(0, global_idx - next_global_idx + 1) 将所有缺口槽位指向同一个 local_idx（恢复帧）。:172-177 将该帧的 data 和 timestamp 写入全部回填槽。50Hz 下暂停 60s → global_idx≈3000，回填~3000 个槽为恢复瞬间的同一份数据。

4) 暂停时长吃掉录制额度：episode_recorder.py:337 self._frame_count = self._buffer.size（含回填槽）；:254 用 _frame_count >= max_frames 判定停止（无独立墙钟计时）。max_frames=4500 @50Hz=90s 栅格预算；暂停 60s 回填消耗 3000 槽，只剩 1500 槽≈30s 有效数据即触发 :731-734 自动停止。

5) 无任何缓解：EpisodeRecorder.stop_episode (:598) 直接 flush 无 gap 探测；RecordingSession/CollectionLoop 本脚本未使用（grep 零引用）。
</details>

### `examples/real/vr_teleop_arm_only.py:725` — 全程未调用 validate_action()，力矩/温度安全门整条链路缺失
- 来源: arm_only | 验证置信: high

CLAUDE.md 明确要求 send_action() 前必须调用 validate_action()（反模式清单：'Ignoring validate_action()'）。本脚本在 L711 arm_inner.set_target(arm_cmd) 和 L725 robot.send_action(action) 前没有任何 validate 调用；且从不调用 arm_inner.get_dynamics()，validate_action 的 torque gate（actual_arm_tau）与 temperature gate（actual_arm_temps）所需数据链路完全断开。脚本自建的检查只覆盖 error 门（L562 robot.arm.is_error()）与 workspace clamp（L673-675），持续卡阻/过热场景下无任何保护。作为真机录制入口而非标定脚本，不适用 info 降级。

<details><summary>验证记录</summary>

确认属实，逐层追溯验证：

1. vr_teleop_arm_only.py L711/L725 直发无 validate：L711 调用 arm_inner.set_target(arm_cmd) 前无任何 validate_action 调用，L725 调用 robot.send_action(action) 前同样无。脚本未 import validate_action（imports 仅含 preflight，见 L50-53）。

2. get_dynamics 数据链路完全断开：脚本全文无 get_dynamics 调用（grep 确认）。validate_action 的 torque gate（robot/validate.py:54-59 检查 abs(tau) > _ARM_TORQUE_LIMIT_NM）与 temperature gate（L62-67 检查 temps > 70C）均为可选参数，None 时跳过——但脚本根本不调 validate_action，check #3/#4 连跳过的机会都没有。

3. ArmInnerLoop 不自带力矩/温度保护：inner_loop.py 的 _run() 循环（L309-429）仅在 L411-419 读取 tau/temps 并存入共享变量供外环消费，_send_target()（L488-537）仅做 delta clamp + speed ramp，无任何主动力矩/温度安全逻辑。L150 docstring 明确写"consumed by the outer loop for recording + torque/temperature pre-send gates"。

4. robot.send_action() 无内部校验：interface.py:235-245 仅调用 self.hand.send_action()，无任何安全检查。

5. 对照 arm_only_record_plus.py:666：确实调了 get_dynamics()，但仅用于填充 arm_qvel/arm_tau 到 RobotState（用于录制），同样未调 validate_action。两个脚本均缺 validate gate。

6. 自建检查覆盖有限：L542 error_state 检查、L562 robot.arm.is_error()（仅 arm，非 robot.is_error() 全覆盖）、L673-675 workspace clamp——覆盖 validate_action 的 check #1(部分)、#6，缺失 check #2(连接)、#3(力矩)、#4(温度)、#7(arm joint-limit clip)、#8(hand joint-limit clip)。

7. CLAUDE.md 引用准确但属反模式提示：CLAUDE.md 将"Ignoring validate_action()"列为反模式（anti-pattern），虽非硬性强制要求，但该脚本是真机录制入口（CLAUDE.md entry points 表列出），缺失力矩/温度安全门是实质性安全缺陷，严重性 major 合理。

8. 实际后果可达：持续卡阻（push against obstacle）或长时间运行过热场景下，硬件层保护以外无软件层防护。IK 异常产生超限位 joint angle 时，无 joint-limit clipping 兜底，arm 可能触发 C22/C24 错误而非被软件层提前截断。
</details>

### `examples/real/vr_teleop_arm_only.py:725` — 5 个直控入口全部未调用 validate_action()——力矩门/温度门整体缺失
- 来源: x-safety | 验证置信: high

CLAUDE.md 反模式清单明确『Ignoring validate_action() → always call before send_action()』，安全架构第 1 层的力矩门/温度门只在 validate_action()（robot/validate.py:21，controller.py:346 有正确接线）中实现。但 vr_teleop_arm_only.py（711/725）、vr_teleop_arm_only_record.py（795/809）、vr_teleop_arm_only_record_plus.py（842/856）、keyboard_teleop_real.py（594/600）、replay_traj.py（764/778）都直接 arm_inner.set_target() + robot.send_action()，无一调用 validate_action。record_plus 更是在 666 行 `arm_qvel, arm_tau, _temps = arm_inner.get_dynamics()` 已经拿到了温度数据却以 `_temps` 丢弃——数据在手边但门没接。后果：过载/过热/碰撞卡阻时这些入口不会停发命令，只能等固件报错（C22/C24）兜底，且这些入口还对 C22/C24 做最多 5 次自动清错重试（record_plus.py:684-699）。

<details><summary>验证记录</summary>

全部核实，发现属实。

证据链：

1. validate_action 是力矩/温度门的唯一实现点 (validate.py:21-94)：
   - 力矩门：第54-59行，np.abs(tau) > _ARM_TORQUE_LIMIT_NM（按关节：J1-J2=50, J3-J5=30, J6-J7=20 Nm，定义在 types.py:19）
   - 温度门：第62-67行，temps > 70C
   - 两个门都是 opt-in（None 参数跳过），必须由调用方显式传入 actual_arm_tau 和 actual_arm_temps

2. 唯一正确调用点 (controller.py:346-352)：TeleopController 在 compute_action() 中调用 validate_action(robot, action, actual_arm_qpos=..., actual_arm_tau=state.arm_tau, actual_arm_temps=arm_temps)，接线完整。

3. ArmInnerLoop 明确不做力矩/温度门 (inner_loop.py:150-153)：
   - 第150行注释："consumed by the outer loop for recording + torque/temperature pre-send gates" — 声明这些数据供外层循环做闸门用
   - 第415-420行：读取温度并存入 self._arm_temps，但全文无任何温度阈值检查
   - _send_target()（第488-577行）：直发 arm.set_servo_angle()，仅做 delta clamp + 软启 + C22/C24 恢复，无力矩/温度检查

4. robot.send_action() 无任何验证 (interface.py:235-245)：仅是手部动作的薄封装（self.hand.send_action(action.hand_qpos_cmd)），完全不触及力矩/温度。

5. 5 个入口均绕过 validate_action（已逐文件核实）：
   - vr_teleop_arm_only.py:711,725 — arm_inner.set_target(arm_cmd) + robot.send_action(action)
   - vr_teleop_arm_only_record.py:795,809 — 同上模式
   - vr_teleop_arm_only_record_plus.py:842,856 — 同上模式；且第666行 arm_qvel, arm_tau, _temps = arm_inner.get_dynamics() 已获取温度数据却以 _temps（下划线前缀=约定丢弃）丢弃
   - keyboard_teleop_real.py:594,600 — 同上模式
   - replay_traj.py:764,778 — 同上模式

6. 全仓 grep 结果：validate_action 仅出现在 validate.py:21（定义）、controller.py:30（导入）、controller.py:346（调用）三处，证实无其他调用点。

实际后果分析：

- 力矩门缺失：过载/碰撞卡阻时，ArmInnerLoop 继续发 servo 命令，只能等固件电流极限触发 C22/C24。而 record_plus.py:684-699 对 C22/C24 做了最多 5 次自动清错重试（recover_count > 5 才急停），这会反复重送危险命令。
- 温度门缺失：长时间高负载操作时关节温度无软件级预警，仅靠固件热保护（阈值可能高于 70C 软件门）。
- replay_traj.py 是自主回放（无人类监督），缺失力矩/温度门的风险高于遥操作入口。
- CLAUDE.md 反模式清单明确写 "Ignoring validate_action() -> always call before send_action()"，安全架构第 1 层文档也明确列入力矩门和温度门。

严重性维持 major 的理由：5/5 个直控入口全部绕过安全架构第 1 层，且 ArmInnerLoop 代码注释明确指望外层做这些检查，说明这是设计意图而非可选优化。replay_traj.py 的自主回放特性进一步放大了风险。
</details>

### `examples/real/vr_teleop_arm_only.py:729` — add_frame 未传 signals：flag_ik_ok/flag_retarget_ok/flag_held 每帧恒 False，与事实相反
- 来源: arm_only | 验证置信: high

recorder.add_frame(state, action, vr_frame) 缺 signals 参数，EpisodeRecorder 对缺省 signals 取 sig.get(...) 默认 False 写入三个 flag 数据集。而本脚本只有 IK 成功的帧才会走到 add_frame（L701 IK 失败先 continue），所以录出的每一帧实际都 ik_ok=True 却被记为 flag_ik_ok=False。任何按 flag_ik_ok 过滤的下游流水线会整段丢弃这些 episode；flag_held 也无法反映 hold 状态。record_plus 已传 signals。

<details><summary>验证记录</summary>

独立验证确认了所有声称：

1. vr_teleop_arm_only.py:729 调用 recorder.add_frame(state, action, vr_frame) — 没有传入 signals= 参数。

2. episode_recorder.py:276: sig = signals or {} — 当 signals=None（默认值）时，sig 为空字典。

3. episode_recorder.py:320-322: 三个标志位均通过 sig.get("ik_ok", False) 等默认取值，在 sig 为空字典时全部得到 False。

4. vr_teleop_arm_only.py:701-703: if not ik_result.success or ik_result.qpos is None: ik_method = "fail"; continue — IK 失败的帧永远不会到达第 729 行的 add_frame。第 703 行到第 729 行之间不存在其他 continue（第 705-725 行是直线代码：赋值、RobotAction 构造、send_action）。因此，到达 add_frame 的每一帧都满足 ik_ok=True，但被记为 flag_ik_ok=False。

5. 同一个文件中还有其他在 add_frame 之前的 continue 路径：第 648 行（not teleop_active or vr_stale）和第 654 行（mapped is None）。所有存活到第 729 行的帧都通过了全部三个关卡（遥操作活跃 +映射成功 + IK 成功）。

6. vr_teleop_arm_only_record_plus.py:873-874 正确传入了 signals=sig: sig = {"ik_ok": ik_result.success and ik_result.qpos is not None, "retarget_ok": True, "held": False}。这证明是相对于其他同类脚本正确处理信号的疏忽性遗漏。

对 flag_held 的说明：巧合的是，该脚本中 held 的默认值 False 是正确的——被 hold 的帧会在第 648 行通过 continue 跳过。但这只是偶然正确，并非通过显式信号实现。对于 flag_retarget_ok：这是一个 arm-only 脚本，本不涉及手部重定向，但被硬编码为 False 作为元数据具有误导性（record_plus 将其设为 True）。

核心危害：依赖 flag_ik_ok 筛选有效帧的任何下游流水线都会静默丢弃由该脚本录制的所有 episode，造成完全数据丢失。该问题的严重性与声称一致（major）。
</details>

### `examples/real/vr_teleop_arm_only_record.py:809` — 跳过 validate_action() 预检安全门 — 缺少力矩/温度/关节限位检查
- 来源: arm_only_record | 验证置信: high | 原评级: critical

robot.send_action(action) 之前未调用 validate_action() (dexmani_real/robot/validate.py:21-94)。validate_action 执行: (1) 每关节力矩门限 (types.py:19, J1-J2=50Nm/J3-J5=30Nm/J6-J7=20Nm), (2) 每关节温度门限 70°C, (3) arm/hand 关节软限位 clip, (4) robot 级 is_error() (含 hand 错误)。TeleopController (controller.py:346-352) 通过 arm_inner.get_dynamics() 获取 torque/temps 并传入 validate_action; 本脚本从未调用 get_dynamics() — 热路上完全没有力矩和温度保护。

<details><summary>验证记录</summary>

属实但评级过高。核实确认：(1) vr_teleop_arm_only_record.py:809 确为 robot.send_action(action)，全文件 grep validate_action/get_dynamics 零匹配，脚本从未调用预检门；(2) RobotInterface.send_action (interface.py:235-245) 内部无任何校验，仅转发 hand 命令；(3) 脚本 623 行 robot.get_state(arm_qpos=arm_qpos) 经 interface.py:180-182 将 arm_tau 填为 NaN，且 validate.py:54-67 的力矩门(types.py:19: [50,50,30,30,30,20,20]Nm)与温度门(70°C)在本脚本热路径上完全缺失——即使补调 validate_action 也会因 NaN 静默失效，必须像 controller.py:308 那样先调 arm_inner.get_dynamics()；(4) 对照 controller.py:346-352 引用准确；(5) 后继版本 vr_teleop_arm_only_record_plus.py:666 虽取 get_dynamics 但仅用于录制(_temps 丢弃)，仍无 validate_action，缺陷延续。此外这直接违反 CLAUDE.md 明文反模式"Ignoring validate_action() → always call before send_action()"。但 critical 评级夸大了后果："热路上完全没有力矩和温度保护"仅对软件预警层成立——validate_action 的 8 项检查中多数已有等效兜底：错误门在脚本 612-656 行逐帧检查(inner-loop error_state 连续 3 次急停 + robot.arm.is_error() 硬错误急停)；workspace clamp 在 753-757 行 IK 前自做；hand 限位+delta clip 在 XHand.send_action 驱动内部强制执行(xhand.py:520,526-530)，且本脚本 hand 命令仅为保持当前位姿(799 行)；arm 限位由 IK 候选过滤(ik.py:271)+内环 0.3rad 步进钳位(inner_loop.py:502-506)+Mode 6 固件限速(90°/s)覆盖；固件碰撞检测已启用(inner_loop.py:263 set_collision_sensitivity(1))，C31 触发即内环停机→脚本急停，固件过温/过载硬保护经同一路径捕获。真正缺失的是力矩(50/30/20Nm)与 70°C 的软件早期预警门：持续抵压低于固件碰撞阈值时不会被主动拦截，只能靠固件硬保护或人工 ESC。人在环遥操+多层固件兜底下属安全层退化而非无保护，定为 major。
</details>

### `examples/real/vr_teleop_arm_only_record_plus.py:428` — 丢弃路径(D键/退出时)未删除 HDF5 文件，数据泄漏
- 来源: x-recording | 验证置信: high

_stop_recording(save=False) 仅调 recorder.stop_episode(success=False)，文件完整写出并留在磁盘。用户看到 '已丢弃' 但 episode_*.h5 文件存在且可读（仅 /meta success=False）。DataValidator.validate_directory 会 glob 到这些文件并纳入 batch 验证/导出，污染下游 pipeline。Contrast: CollectionLoop.discard_episode() (collection_loop.py:162) 正确调 h5_path.unlink()。同样影响 vr_teleop_arm_only_record.py:403。

<details><summary>验证记录</summary>

逐条核实成立。(1) 丢弃路径确认：examples/real/vr_teleop_arm_only_record_plus.py:425-436 的 _stop_recording(save=False) 仅调 recorder.stop_episode(success=False)，无任何 unlink；可达入口有四处——D 键 (567-572 行 ControlSignal.DISCARD)、Q 键录制中二次确认选 D 或 30s 超时 (522-529 行，打印"已丢弃"/"超时，默认丢弃")、finally 兜底 (884-885 行)、_emergency_stop (444-446 行)。grep 全文件仅 atexit.register(kb.stop)，无删除逻辑。(2) 库行为确认：dexmani_real/recording/episode_recorder.py:598-717 的 stop_episode/_stop_episode_impl 完整落盘——_flush_buffered() (655 行) 内部经 _ensure_hdf5() (558→376-382 行) 创建文件并写出全部缓冲流，相机队列 drain + forward-fill，meta 写 success=False (692 行) 后正常 close；success 参数只影响 /meta 属性，EpisodeRecorder 全类无删除 API。脚本头部文档第 28 行明确承诺 "D 丢弃录制 (Discard, 不保存)"，与实际"完整保存仅标 False"矛盾。(3) 下游污染确认：data_validator.py:273 validate_directory 用 data_dir.glob("episode_*.h5") 不区分 success，8 项检查均不含 success 过滤，被丢弃 episode 可 PASS；export_hdf5_to_zarr.py:688-690 --filter_success 默认 None，默认导出会把 success=False 的 episode 纳入 Zarr 训练集，过滤是 opt-in。(4) 对照确认：collection_loop.py:157-170 discard_episode() 确实 h5_path.unlink(missing_ok=True) + json unlink，且两脚本 grep 确认完全未用 CollectionLoop，说明库的丢弃语义就是删除文件，脚本未实现。(5) 姊妹脚本确认：vr_teleop_arm_only_record.py 的 _stop_recording 实际在 404-420 行（发现写 403，差一行不影响结论），该变体无 D 键但 Q (500 行) 与 finally (841 行) 同样走 save=False 泄漏路径。唯一不成立的窄边界：录制启动后 0 帧即丢弃时 _flush_buffered 早退且 HDF5 懒创建 (_file=None)，不产生文件——但两脚本都启动 CameraProcess，首个带相机帧的 add_frame 即触发 _ensure_hdf5 (356-357 行)，正常使用场景下文件必然存在。严重性维持 major 恰当：操作员被明确告知"不保存"的坏数据（故障/失败演示）完整留存，且默认导出/批量验证均会纳入；success=False 标记提供了 opt-in 过滤缓解，故不到 critical。
</details>

### `examples/real/vr_teleop_arm_only_record_plus.py:508` — record_plus 的 Q 确认对话框阻塞急停 (ESC) 最长 30 秒
- 来源: x-triplet | 验证置信: high

C 的 QUIT 处理在录制中引入 30 秒确认循环 (S 保存/D 丢弃)。该循环内 kb.poll 只检查 STOP 和 DISCARD，不检查 EMERGENCY_STOP。用户在该窗口中按 ESC 的信号被 kb.poll 消费后因无匹配分支而静默丢弃，导致软件急停在此期间无法触发。A 和 B 无此对话框，Q 直接退出。B 是对的（无对话框，简单直接）; C 引入的对话框需要在循环内同时处理 ESC。

<details><summary>验证记录</summary>

属实。逐层核实：(1) examples/real/vr_teleop_arm_only_record_plus.py L505-516 的确认循环 deadline=30s，循环内 kb.poll(timeout=0.1) 的返回值只与 ControlSignal.STOP/DISCARD 比较，无 EMERGENCY_STOP 分支。(2) dexmani_real/teleop/control/keyboard.py L171-176 证实 poll() 是"全量抽干"语义（list(self._buffer) 后 buffer.clear()），且 on_press (L109-121) 只入队不直接触发动作——因此对话框窗口内按 ESC 产生的 EMERGENCY_STOP 信号被取出后无分支匹配，被永久丢弃，对话框结束后也不会再被看到（L518-534 之后 running=False 直接走 L909-918 的常规 cleanup，不经过 _emergency_stop）。(3) 后果确认：窗口期内机械臂并非断电——dexmani_real/robot/inner_loop.py L319-330/L581-599 显示 50Hz 内环线程在主循环阻塞、目标超时 (0.2s) 后进入 hold 模式，持续以 set_servo_angle 主动发保持指令，臂全程伺服上电；且主循环内的力矩/温度等安全检查 (record_plus L660-702) 也随主循环一起被阻塞。此窗口内软件急停 (_emergency_stop, L438-452 → robot.emergency_stop()) 完全不可达，最长 30s。(4) 对照文件核实无误：vr_teleop_arm_only_record.py L498-502 与 vr_teleop_arm_only.py L437-441 的 Q 均为直接 _stop_recording(save=False); running=False，无对话框，ESC 分支始终在每轮 poll 中优先检查。缓解因素（窗口仅在录制中按 Q 后打开、臂处于 hold 不追踪 VR、硬件急停按钮仍可用、30s 有界）使其不到 critical，major 评级恰当。
</details>

### `examples/real/vr_teleop_arm_only_record_plus.py:604` — record_plus 在 start_episode 中遗漏 camera_K，导致生产录制的 zarr 导出丢失全部相机元数据
- 来源: x-triplet | 验证置信: high

record_plus 调用 recorder.start_episode() 时传了 depth_scale、calib、camera_name、record_config，但未传 camera_K。导致 /meta 缺少 camera_K 属性。同时 export_hdf5_to_zarr.py L104 访问 meta["camera_K"] 时抛出 KeyError，被外层 except 捕获后返回 None，连 T_world_camera/T_eef_camera 也一并丢弃。record.py 正确传入了 camera_K (L567)。

<details><summary>验证记录</summary>

我亲自核实了完整调用链，发现属实。(1) examples/real/vr_teleop_arm_only_record_plus.py L604-609：recorder.start_episode() 只传 depth_scale/calib/camera_name/record_config，确无 camera_K；且 recorder 是直接构造的 EpisodeRecorder（L343），无中间层补齐。(2) dexmani_real/recording/episode_recorder.py L120-131 签名中 camera_K 默认 None；L169-179 存入 _pending_meta；L223-225 仅在 camera_K is not None 时才写 meta.attrs["camera_K"]——无任何兜底。calib 分支（L211-221）只写 serial/type/T_world_camera/T_eef_camera，config/camera_calib.py L78-82 的 CameraCalibEntry 只含外参无内参；record_config 来自 pointcloud_meta → pointcloud_processor.py L52-63 的 to_meta_dict()，只有 pc_* 键。故 /meta 确实缺 camera_K。(3) record_plus L860/L874 每帧传 camera_frame=cam，rgb 数据集会创建，episode_recorder.py L695 使 has_camera=True，因此导出时不会在 has_camera 检查处提前返回。(4) tools/export_hdf5_to_zarr.py L91-108：_read_camera_meta 的 try 块 L104 访问 meta["camera_K"] 抛 KeyError，被 L107 except (KeyError, ValueError) 捕获返回 None（无告警）；L285-286 得到 camera_meta=None，L629 'if camera_meta is not None' 跳过，zarr 的 meta/camera 组（K、T_world_camera、serial、type、depth_scale）整体静默丢失——即已正确录入 h5 的外参也被连带丢弃。(5) 对照组属实：vr_teleop_arm_only_record.py L567 确实传了 camera_K=camera.camera_K。唯一小瑕疵：发现中说 T_eef_camera 也被丢弃，实际 _read_camera_meta 从不读 T_eef_camera（只读 L105 的 T_world_camera），但这不影响结论实质。后果评估：camera_K 来自 CameraProcess 的硬件实时读回（sensor/camera_process.py L175-182），h5 文件本身无法事后恢复内参；CLAUDE.md 的 schema 明确列出 camera_K 为 /meta 属性，record_plus 违反了文档 schema，且失败是静默的。缓解因素是在线点云已烘焙外参落盘，主 3D 数据路径不受影响，故维持 major 而非 critical。
</details>

### `examples/real/vr_teleop_arm_only_record_plus.py:604` — record_config 仅含 pc_meta，缺失控制参数元数据（ema_alpha_*、*_clip、*_mode）
- 来源: x-recording | 验证置信: high

recorder.start_episode() 的 record_config 仅设为 camera.pointcloud_meta (pc_* 键)，缺失: control_mode、arm_mode、hand_mode、arm_delta_clip、hand_delta_clip、hand_max_qvel_deg_s、hand_ema_alpha、hand_low_pass_alpha、ema_alpha_pos、ema_alpha_rot。实际生效的 EMA_ALPHA_POS=0.94/EMA_ALPHA_ROT=0.67 (line 181-182) 无法从 HDF5 /meta 提取，数据集不自描述，下游无法复现控制参数。同样影响 vr_teleop_arm_only_record.py:563-568。

<details><summary>验证记录</summary>

逐条核实属实。(1) record_plus.py:604-609 的 start_episode 调用确认 record_config=camera.pointcloud_meta（camera 为 None 时干脆为 None），而 pointcloud_meta 只返回 pc_* 八个键（camera_process.py:158-163 → pointcloud_processor.py:52-63 to_meta_dict）。(2) episode_recorder.py:169-179 将 record_config 存入 _pending_meta，_write_meta_attrs（episode_recorder.py:233-239）逐键写入 /meta attrs，注释明言这是"Collection-config snapshot (control mode, EMA alphas, delta clips) — essential for downstream reproducibility"，即契约由调用方供参，库不兜底——确认无任何其他代码路径写入 control_mode/ema_alpha_* 等 attrs。(3) 对照组准确：controller.py:604-632 的 _build_record_config 含全部 10 个控制参数，并在 645 行合并 pointcloud_meta；CLAUDE.md 的 HDF5 /meta 文档也列出这些 attrs，而 episode_recorder.py:689 无条件写 schema_version=7，故 record_plus 产出的文件自称 v7 却缺失文档化的 v7 控制参数。(4) 数值自行重算：tau_pos=-0.02/ln(0.4)=0.02183s → alpha@16Hz=1-exp(-0.0625/0.02183)=0.943；tau_rot=-0.02/ln(0.7)=0.0561s → 0.672，与发现所述 0.94/0.67 一致（signal_utils.py:12-48 定义确认为 y+=alpha*(x-y) 语义），且这两个值在 record_plus.py:791-796 真实作用于录制的 action 流。(5) record.py:563-568 同样只传 pointcloud_meta，其 EMA 硬编码 0.6/0.3（record.py:170-171）——两个脚本写入同一 episodes_arm/ 目录（record_plus.py:344 / record.py:330），混存的文件平滑系数不同却均无从 /meta 区分，"数据集不自描述、下游无法复现"的后果成立且对已采 episode 不可逆。(6) 无兜底：该脚本直用 EpisodeRecorder（record_plus.py:343，非 CollectionLoop，无 sidecar），TrajectoryLogger npz 不记 config。削弱因素有二：当前 tools/ 与 data_validator 无任何代码消费这些 attrs（grep 无命中），故无运行时故障；且此臂-only 脚本中 hand_low_pass_alpha（无 retargeter）等手部参数本就不生效，发现列举略有夸大——但 arm_mode=6、arm_delta_clip=0.3（ArmInnerLoopConfig，inner_loop.py:83）、ema_alpha_pos/rot 确实生效且缺失，脚本以 success=True 保存真实数据（record_plus.py:431），元数据缺口核心成立。考虑到该仓库以数据集为产品、分支即 collection-hardening、且缺口对已采数据永久不可补，major 评级恰当，不作降级。
</details>

### `examples/real/vr_teleop_arm_only_record_plus.py:789` — IK 失败/VR 过期时跳过 add_frame 导致背填槽位中 flag_ik_ok/flag_held 系统性伪造（数据污染）
- 来源: x-recording | 验证置信: high | 原评级: critical

IK 失败时 line 789 执行 continue 完全跳过 recorder.add_frame()，TimestampAlignedBuffer 的 back-fill 机制（timestamp_buffer.py:56-67 'back-filled by the NEXT arriving sample'）将下一帧成功的数据写入所有跳过槽位。这些槽位的 flag_ik_ok=True、flag_held=False，但实际 IK/VR 已失败。下游数据集消费者依 flag_ik_ok 筛选时将接受本应排除的帧，污染模仿学习训练数据。同样影响 vr_teleop_arm_only_record.py line 785。

<details><summary>验证记录</summary>

问题属实，我逐层核实了完整链路。(1) 跳帧机制：vr_teleop_arm_only_record_plus.py 当前工作树中 IK 失败 continue 在 832-834 行（发现引用的 789 行现为注释，行号漂移但机制无误），vr_stale/未激活 continue 在 768-777 行，mapped is None 在 781-783 行；三条路径均跳过唯一的 add_frame 调用点（872-874 行），且该调用点只在全成功路径可达，signals 恒为 {"ik_ok": True(表达式在此处必真), "retarget_ok": True, "held": False}——即该脚本写入 HDF5 的 flag_ik_ok 恒真、flag_held 恒假。vr_teleop_arm_only_record.py 785-787/825-827 行完全相同（该文件发现引用行号精确）。(2) 回填机制：timestamp_buffer.py:63-67 的 n_repeats = global_idx - next_global_idx + 1 配合 add() 172-177 行 self._data_buffer[key][global_idxs] = value，确认下一个到达样本的全部字段（含 flag_ik_ok=True、flag_held=False 及未来时刻的 state/action）写入所有被跳过的槽位；episode_recorder.py:320-322 确认 flags 完全来自调用方 signals，recorder 侧无 ik/held 兜底（仅 flag_camera_fresh 是 recorder 侧推导，278-297 行）。(3) 下游后果在仓库内真实存在：export_hdf5_to_zarr.py:127-130 与 167-169 用 /flag_held 计算 held_ratio 并按 --max_held_ratio 过滤 episode——这些脚本 held 恒假使该质量门系统性失效（VR 长时间过期的 episode 会被回填为全成功数据并通过过滤）；visualize_episode.py:121-126 将恒显示 100% ik 成功率；data_validator.py 不检查这些 flag，验证也不会捕获。(4) 非有意设计：生产路径 teleop/core/controller.py:389-401 每帧记录并如实写 held/ik_ok/retarget_ok，符合 CLAUDE.md schema 语义；且这两个脚本自己的 debug traj_logger（record_plus.py:817-830）如实记录 ik_ok=False，说明是 episode 录制路径的遗漏而非约定。反驳点均不成立：录制期间 IK 失败/VR 过期路径确实可达（recording 与 teleop 同启，IK 尖峰与 VR 卡顿为已知现象）。严重性修正为 major 而非 critical：无安全/崩溃/文件损坏；污染槽位是合法成功帧的复制（可通过 /timestamp 重复值离线检出，CLAUDE.md 已记载该约定）；主生产入口 vr_teleop_shm.py（TeleopController→CollectionLoop）不受影响，仅限两个 arm-only 采集脚本。但数据完整性问题真实且系统性：flag 在这两个脚本产出的数据集中完全退化为常量，专为剔除劣质 episode 设计的 held_ratio 门被静默绕过。
</details>

### `examples/real/vr_teleop_arm_only_record_plus.py:856` — 跳过 validate_action() 安全门：扭矩/温度/关节限位裁剪全部缺失
- 来源: record_plus | 验证置信: high

第 856 行 robot.send_action(action) 之前未调用 validate_action()。validate_action（robot/validate.py:21-94）执行 8 项预发送检查: SDK 错误态、臂连接、扭矩门（_ARM_TORQUE_LIMIT_NM）、温度门（70°C）、工作空间夹紧、臂关节软限位裁剪（qpos_min_soft/max_soft，缩进固件限界防止 reduced-mode 故障）、手关节限位裁剪。当前代码只做了 workspace clamp（line 802-804）和臂错误态检查（line 680），缺失扭矩/温度关断和关节限位软裁剪。对比: TeleopController（teleop/core/controller.py:346）每次都调用 validate_action。arm 命令经由 arm_inner.set_target（line 842）绕过 validate_action 中的 joint-limit 软裁剪，delta-clamp（inner_loop.py:503-506）是最后一道防线但不是替代。

<details><summary>验证记录</summary>

发现属实。我逐条尝试反驳均未成功：(1) 脚本内 grep 无任何 validate_action 调用；L856 robot.send_action(action) 与 L842 arm_inner.set_target(arm_cmd) 与引用一致。(2) 下游无隐藏校验：interface.py:235-245 send_action 仅发手部命令无任何检查；inner_loop.py:169-175 set_target 仅拷贝目标；_send_target(inner_loop.py:488-528) 只有 per-step delta clamp(503-506)+速度爬坡，无关节限位裁剪、无扭矩/温度门；_monitor(450-456) 明确"Never mutates commands or the error state"。(3) 扭矩/温度门确实缺失：脚本 L666 读取 get_dynamics() 但 _temps 直接丢弃、arm_tau 仅入录制；inner_loop.py:149-150 注释明确该回读设计用途是"recording + torque/temperature pre-send gates"，即架构预期外环做门控而此脚本未做；且固件碰撞灵敏度被刻意调到最低(xarm7.py:71-73 collision_sensitivity=1)，软件扭矩门(_ARM_TORQUE_LIMIT_NM, types.py:19)是遥操作期间设计上的主要接触力保护层。(4) 臂软限位裁剪缺失属实：IK 使用 planner 模型(URDF≈固件)限位而非 2.5° 内缩软限位(xarm7.py:32-48,95-97)，validate.py:81-83 与 xarm7.py:29-31 记载边界钉死命令会触发 reduced-mode 故障——后果是固件报错/中止采集(滋扰性停机)而非危险运动。(5) 对比准确：controller.py:346-352 每帧带 tau/temps 调 validate_action 且失败时 hold，之后才 set_target(363)/send_action(366)；CLAUDE.md 反模式清单明列"Ignoring validate_action() → always call before send_action()"。部分削弱点：标题"关节限位裁剪全部缺失"对手部言过其实——XHand 驱动 send_action 内部自带 _limit_joint_range(xhand.py:798-808)+E3 delta clip(520-530)，且本脚本手部命令为保持当前位置(L846)，故 validate_action 检查8缺失无实际后果；脚本也已有臂错误态轮询(L680,L655)和 IK 前工作空间夹紧(L802-804，实际比 validate_action 只夹 target_eef_pos 元数据更有效)。但扭矩门、温度门、臂软限位裁剪三项在真机数据采集入口(feat/collection-hardening 分支活跃修改文件)上确实缺失，违反项目显式安全约定，major 评级恰当；因固件限位/Mode6 轨迹规划/delta clamp/错误态轮询等纵深防御仍在、缺失项多转化为固件级故障而非失控运动，不足 critical。
</details>

### `examples/real/vr_teleop_arm_only_record_plus.py:874` — 录制中暂停/VR 过期/IK 失败时跳过 add_frame，缓冲区用过期数据回填空槽
- 来源: record_plus | 验证置信: high

控制循环在三个路径跳过 recorder.add_frame(): (a) L768-777: teleop_active=False 或 vr_stale → continue; (b) L832-834: IK 求解失败 → continue。录制期间这些空槽产生 gap，TimestampAlignedBuffer 在下一帧 add() 时回填所有跳过槽位（timestamp_buffer.py:64-67 forward-fill 逻辑）。后果：(1) 暂停段的数据(arm_qpos/qvel/tau/action)全是暂停前最后一帧的副本，而实际机械臂静止、速度近零，形成静/动错配；(2) 暂停 66 秒时网格填满触 max_frames 自动停录（L876-879），打断正常录制；(3) flag_held 在 schema 中始终为 False（L873:held=False），因为 held 帧从未录到——held 态的语义标记丢失。

<details><summary>验证记录</summary>

核心问题属实，三条跳过路径与后果均已亲自核实，但发现中有一处机制方向性错误需修正。

已核实的事实：
1) 跳过路径真实可达：vr_teleop_arm_only_record_plus.py L768-777（`if not teleop_active or vr_stale: ... continue`）、L832-834（IK 失败 continue），另有 L781-783（mapped is None）发现未列出。C 键（L574-593）只翻转 teleop_active、不动 recording_active，L577 明确打印"录制继续"——暂停+录制并存是宣传的功能；该脚本也没有生产控制器的 VR 累计超时自动停录（controller.py L277-293），vr_stale 可在录制中无限持续。
2) 回填机制属实：timestamp_buffer.py L55-67，下一次 add() 时 `n_repeats = global_idx - next_global_idx + 1`，所有跳过槽位写入同一个新样本（L172-177 `self._data_buffer[key][global_idxs] = value`；时间戳同样重复）。栅格锚定仅一次（episode_recorder.py L268-273 `_grid_anchored`），state.timestamp = time.perf_counter()（robot/interface.py:230），暂停期间栅格索引随墙钟推进，恢复时不重锚。
3) 数值复算正确：max_frames = round(60*16) = 960（脚本 L345），buffer 容量 = 960+100 = 1060 槽（episode_recorder.py L184），dt = 62.5ms → 栅格容量 66.25s。恢复时若总时长 > 66.25s，buffer.add L158-165 置 recording_stopped → add_frame L333-335 返回 False + max_frames_reached → 脚本 L875-879 自动停录保存（success=True）。60-66.25s 区间则先写入回填块再于下一帧触发 L254 检查停录。更普遍的是：任意时长的暂停都静默吞噬 60s 录制预算（如暂停 30s = 480 个重复槽位）。
4) flag_held 恒为 False 属实：L873 硬编码 `held=False`，且该行的 ik_ok 恒为 True（失败路径已在 L834 continue），三个 flag 在此脚本中全是常量；IK 实际失败的时刻在回填槽位中被标为 ik_ok=True。生产控制器对比之下会录制 held 帧（controller.py L389-401）。
5) 无下游兜底：脚本直接用 EpisodeRecorder（L343），不经过 RecordingSession/DataValidator；validator 的 no_duplicate_frames 检查（data_validator.py L192-217，要求 total_dup==0）本可捕获此类文件但在此路径未被调用。docs/16hz-arch-review-2026-07-17.md L27 佐证含暂停的录制文件在实践中确实存在。

需修正的不准确处（不影响成立）：发现称暂停段数据是"暂停前最后一帧的副本"，方向错了——timestamp_buffer.py 文档 L34-35 和代码明确是用"下一个到达样本"（恢复后第一帧）回填。由此"静/动错配"对臂状态的描述被夸大：暂停期臂物理保持位置，恢复首帧位姿≈保持位姿、qvel≈0，臂数据近似正确；真正被污染的是 vr_wrist/landmarks（操作员手已移动）、重复时间戳、相机流（_fill_to 用恢复后帧填满整个暂停跨度）和错误的 flag（ik_ok=True/held=False/camera_fresh 继承自恢复帧）。另外在 >66.25s 的极端情形，回填在写入前即中止（buffer.add L158-165 先返回），垃圾块不落盘，episode 只是在暂停起点截断并被停录。

严重性维持 major：这是数据采集加固分支上活跃使用的采集入口，暂停/VR 丢帧是常态操作；产出的 episode 以 success=True 保存，含成段重复的 obs/action/相机/VR 数据且 flag 全部失真，预算静默烧毁导致 episode 意外提前终止，且该路径无任何验证兜底——对以数据为产品的仓库属于静默数据质量缺陷。
</details>

### `examples/real/vr_teleop_shm.py:176` — Cartesian EMA 语义相反：真机 SHM 路径 1.0/1.0 直通（平滑交给 Mode 6 固件），仿真仍软件平滑 0.8/0.4
- 来源: x-realsim | 验证置信: high

真机生产入口显式设置 ema_alpha_pos=1.0/ema_alpha_rot=1.0 直通（依赖臂 Mode 6 固件在线规划做平滑）；仿真 VR 入口构造 TeleopPipeline(ema_alpha_pos=0.8, ema_alpha_rot=0.4) 仍在 IK 前做软件 EMA。仿真里观察到的目标滞后、抖动抑制、超调/响应特性均来自软件滤波，而真机路径把原始 mapper 输出直接送 IK——sim 里“平滑没问题/延迟可接受”的结论对真机不成立，反之真机的 IK 抖动问题在 sim 中会被 EMA 掩盖。属于一侧已改（直通化）另一侧未同步。

<details><summary>验证记录</summary>

核实了以下调用链：

**真机路径 (vr_teleop_shm.py:176-177)**：
`TeleopControllerConfig(ema_alpha_pos=1.0, ema_alpha_rot=1.0)` → `controller.py:153-154` 存储为 `self._ema_alpha_pos/rot` → `controller.py:190-192` 传入 `TeleopPipeline(..., ema_alpha_pos=self._ema_alpha_pos, ema_alpha_rot=self._ema_alpha_rot)` → `pipeline.py:52-53` 存储为 `self._ema_alpha_pos/rot` → `pipeline.py:134-138` 调用 `ema_smooth_pose(..., alpha_pos=1.0, alpha_rot=1.0)` → `signal_utils.py:95-102` 中 `pos = 1.0*target + 0.0*prev = target`，完全直通。

**仿真路径 (vr_teleop_sim.py:493-494)**：
`TeleopPipeline(arm_mapper, hand_retargeter, planner, ema_alpha_pos=0.8, ema_alpha_rot=0.4)` → `pipeline.py:52-53` → `pipeline.py:134-138` 调用 `ema_smooth_pose(..., alpha_pos=0.8, alpha_rot=0.4)` → 产生 `pos = 0.8*target + 0.2*prev` 和 `rv = 0.4*target_rv + 0.6*prev_rv`，是实际的 EMA 滤波，引入了滞后。

**关键差异**：
1. 两端都经过完全相同的 `TeleopPipeline.compute_action()` → `compute_arm_command()` → `ema_smooth_pose()` 函数调用链，仅 alpha 参数不同。
2. EMA 是 Cartesian-space 且位于 IK 之前 (pipeline.py:131-141，注释 "sole smoothing stage, before IK")，意味着仿真喂给 IK 的是滤波后的目标位姿，而真机喂的是原始 mapper 输出。
3. 真机依赖 Mode 6 固件轨迹规划做关节空间平滑 (controller.py:131, CLAUDE.md)，仿真无此等价物——`velocity_limited_step` (vr_teleop_sim.py:852) 是 bottleneck scaling，非 EMA，180°/s 上限很宽松，正常遥操作几乎不会触发。
4. 仿真 @50Hz (sim:105) vs 真机 @16Hz (real:68) 进一步放大了不一致性。

**后果成立**：仿真中观察到的目标滞后、抖动抑制、IK 行为均受 EMA 影响，不能代表真实硬件行为；真机的 IK 对原始 mapper 输出的响应在仿真中被 EMA 掩盖。
</details>

### `examples/sim/test_motion_planning_sim.py:1805` — test_motion_planning 配对断裂：sim main() 只跑 Pick-and-Place，文档声称的 IK/plan_path 测试全是死代码
- 来源: x-realsim | 验证置信: high

真机版按顺序跑 Test1 solve_ik、Test2 solve_teleop_ik、Test3 plan_path、Test4 硬件执行、Test5 IK 自碰撞；sim 版 main() 已被改成仅 Pick-and-Place episode 循环，ik_test()/plan_and_execute()/return_to_home_sim()/sweep_z_min() 均无调用点，argparse 只有 --headless/--seed/--episodes，而文件头 docstring(1-14 行) 仍声称做随机采样 plan_path + return_home + IK 独立测试。且 sim 的 TeleopProfile(check_self_collision=True) 用库默认跳变限 (90°)、pos 误差阈值，真机 create_planner 用 max_ik_jump_deg=(30,30,30,30,45,45,60)、max_pose_error_pos_m=0.01。后果：“先在 sim 跑通 motion planning 测试再上真机”的流程实际没有验证任何真机将要跑的测试项，IK 成功率/阈值也不可比。

<details><summary>验证记录</summary>

四项事实主张全部经代码核实成立：

1) **Docstring 过期**：sim 文件第 10-13 行声称测试流程为 plan_path 随机采样、return_home、IK 独立测试，但 main()（第 1805-1976 行）实际仅运行 Pick-and-Place episode 循环，argparse 仅接受 --headless/--seed/--episodes 三个参数。

2) **四个函数确为死代码**：`ik_test`(定义于:1117)、`plan_and_execute`(:967)、`return_to_home_sim`(:590)、`sweep_z_min`(:1186) grep 全仓库无任何外部调用点；`sweep_z_min` 内部调用 `_run_desk_test`(:1215) 但因自身无调用者同样死代码。

3) **TeleopProfile 配置确实分歧**：
   - Sim (:1840-1845)：`TeleopProfile(check_self_collision=True)`，其余字段用库默认值 → `max_ik_jump_deg=(90,)*7`, `max_pose_error_pos_m=0.008`（经确认 `dexmani_real/planning/types.py:211-214`）
   - Real (:180-184)：`TeleopProfile(max_ik_jump_deg=(30,30,30,30,45,45,60), max_pose_error_pos_m=0.01)`
   Sim 的跳变限是统一的 90°，真机对 J1-J4 仅 30°，J5-J6 为 45°，J7 为 60°，量级差异大。

4) **后果成立**：真机 main()（:903-968）运行 Test1 solve_ik、Test2 solve_teleop_ik、Test3 plan_path、Test4 硬件执行、Test5 自碰撞 IK 共五项测试；sim 仅跑 Pick-and-Place episode。sim 用 90° 跳变限跑出的 IK 成功率与真机 30° 限不可比，"先在 sim 验证再上真机"的信任链断裂。

严重程度维持 major：非纯文档问题，而是 sim 测试入口功能退化且配置基准与真机不一致，直接影响 sim-to-real 验证流程的有效性。
</details>

### `examples/sim/vr_teleop_sim.py:114` — 仿真 VR 工作空间边界与真机实际使用的 RobotInterfaceConfig 默认值不一致，注释声称一致
- 来源: x-realsim | 验证置信: high

sim 的 WORKSPACE_BOUNDS=[[0.27,0.70],[-0.40,0.40],[0.02,0.55]]，注释写“与 RobotInterfaceConfig 默认值保持一致”；但真机 vr_teleop_shm 未覆盖 workspace_bounds，实际生效默认为 [[0.28,0.72],[-0.45,0.45],[0.05,0.5]]。z_min 0.02 vs 0.05、z_max 0.55 vs 0.5、y ±0.40 vs ±0.45。后果：在 sim 里演示成功的近桌面动作（EEF z∈[0.02,0.05)，例如抓 3-4cm 高的物体）到真机会被 validate_action 的 workspace clamp 静默抬高，任务不可复现；反之 sim 顶部 5cm (0.5-0.55) 也是真机不可达区。

<details><summary>验证记录</summary>


已验证：发现属实，严重等级维持 major。

逐项核实：

1. Sim 侧数值与注释矛盾 — examples/sim/vr_teleop_sim.py:113-118 定义 WORKSPACE_BOUNDS = [[0.27, 0.70], [-0.40, 0.40], [0.02, 0.55]]，行 113 注释明确写"与 RobotInterfaceConfig 默认值保持一致"。但 dexmani_real/robot/types.py:133-139 中 RobotInterfaceConfig.workspace_bounds 的默认值为 [[0.28, 0.72], [-0.45, 0.45], [0.05, 0.5]]。差异：z_min 0.02 vs 0.05（3cm）、z_max 0.55 vs 0.5（5cm）、y +/-0.40 vs +/-0.45（每侧 5cm）、x_min 0.27 vs 0.28（1cm）、x_max 0.70 vs 0.72（2cm）。

2. 真机路径确认 — examples/real/vr_teleop_shm.py:129-139 构造 RobotInterfaceConfig(...) 时未传入 workspace_bounds 参数，使用 types.py 默认值。dexmani_real/robot/interface.py:56 用该默认值初始化 WorkspaceSafety，后续通过 interface.py:130-134 的 check_workspace()/clamp_workspace_pos() 暴露。dexmani_real/teleop/core/controller.py:434-435 将这些方法传入 pipeline.compute_action()。

3. Sim 路径确认 — sim 使用 SimRobotInterface（无 workspace 检查），唯一 workspace 强制点是通过 pipeline.compute_action(check_workspace=is_in_workspace, clamp_workspace_pos=clamp_to_workspace)（行 831-832），回调基于 sim 的 WORKSPACE_BOUNDS。

4. 额外夹持层 — 真机在 dexmani_real/robot/validate.py:79 还有第二层 workspace clamp：action.target_eef_pos[:] = robot.clamp_workspace_pos(action.target_eef_pos)，使用同样的真机默认 bounds。正常工作下是 no-op（pipeline 已夹持），不影响结论。

5. 后果验证成立 — sim 允许 z=0.02-0.05 的近桌面运动，真机在该范围会静默将 target 抬高到 z=0.05。反之 sim 允许 z_max=0.55，真机在 0.5 即被截断。两个 planner config 的 workspace_bounds 均为 None（禁用），无其他代码路径调和此差异。

6. 不存在使发现失效的前置条件 — 夹持机制相同（np.clip），无额外兜底处理。

</details>

### `examples/sim/vr_teleop_sim.py:205` — build_robot_action 丢弃了 target_eef_pos/rot6d 导致 /action_arm_ee 始终为 NaN
- 来源: vr_sim | 验证置信: high

build_robot_action（第 205-212 行）只构造了 arm_qpos_cmd 和 hand_qpos_cmd，丢弃了 pipeline.compute_action() 在 pipeline.py:84-88 行中填充的 target_eef_pos 和 target_eef_rot6d 字段。因此 action.target_eef_pos 始终为 None，/action_arm_ee 在 HDF5 中记录为 NaN（episode_recorder.py:302-303 行）。这导致仿真 episode 无法用于 EE 空间策略训练。真实 controller 直接传递 pipeline.compute_action() 返回的完整 action 对象，保留了末端执行器目标位姿。

<details><summary>验证记录</summary>

已验证完整数据流：pipeline.py:84-88 中 TeleopPipeline.compute_action() 构造的 RobotAction 包含 target_eef_pos 和 target_eef_rot6d（来自 compute_arm_command 返回的 target_pos/target_quat → rot6d 转换）。但 vr_teleop_sim.py:834-835 仅提取 arm_cmd/hand_cmd，随后 vr_teleop_sim.py:878-880 调用 build_robot_action()（定义于 205-212 行）构造新的 RobotAction 时不传 target_eef_pos/target_eef_rot6d。robot/types.py:112-113 确认这两个字段默认值为 None。collection_loop.py:104-105 直接将 action 透传给 EpisodeRecorder.add_frame()，无中间修复。episode_recorder.py:299-304 的 _make_action_ee() 对 None 值写入 np.full(n, np.nan)。真机路径 controller.py:312/393 在正常流程中保留完整 action 对象（仅在 validate_failed 时的 line 360 也丢弃 target_eef，但这是窄边界情况而非每帧必发生）。结论：sim episode HDF5 中 /action_arm_ee 恒为 NaN，发现属实。
</details>

### `examples/sim/vr_teleop_sim.py:297` — 相机深度计算完全错误：Position 图坐标系与世界坐标系混淆，导致所有仿真 episode 中 /depth 数据被结构性破坏
- 来源: vr_sim | 验证置信: high | 原评级: critical

在 capture_camera_frame 函数（第 297-312 行）中，SAPIEN 3.0.3 的 get_picture("Position") 返回的是相机坐标系（OpenGL 约定）下的坐标，而非世界坐标。对中心像素的实证测试显示值为 [0.008, -0.008, -0.9, 0.99]，其中 z=-0.9 是相机空间深度。但代码错误地将 pos[..., :3] 当作世界坐标（第 308 行 world_xyz），然后与世界空间中的相机位置做差（第 309 行）并沿相机世界 z 轴投影（第 311 行）。这产生无意义的混合坐标系运算结果。此外，SAPIEN 中相机前向轴为 +x（ROS 约定），而非 +z，所以 cam_rot_world[:, 2] 取的是上方向。每当启用相机时，每个仿真 episode 中录制的 /depth 数据都会被结构性破坏。

<details><summary>验证记录</summary>

我在仓库同一 conda 环境（real_robot，SAPIEN 3.0.3）下亲自复现并确认了该问题，未能反驳。

核实过程：
1. Shader 层面证据：SAPIEN 包内 vulkan_shader/trivial/gbuffer.vert:54 与 point/gbuffer.vert 均为 `outPosition = cameraBuffer.viewMatrix * (modelMatrix * vec4(position,1))`，即 "Position" 纹理输出的是 view/camera 空间坐标（OpenGL 约定，相机朝 -z），不是世界坐标。发现者对 API 语义的解读正确。
2. 实证复现：我在 real_robot 环境用相同配置（相机挂载实体在世界 (0,0,0.5)、identity 朝向即 SAPIEN ROS 约定朝 +x，盒子在 (1,0,0.5)）运行测试，中心像素为 [0.00768, -0.00768, -0.9, 0.9899]，z=-0.9 证实是 camera-frame；`-pos[...,2]` = 0.9 恰为真实深度。发现中引用的实证数值与我独立测得的完全一致。
3. 逐行复现 vr_teleop_sim.py:302-312 的运算：对该场景算出 depth=1.4（真实深度 0.9）——混合坐标系减法产生结构性错误值，且量级看似合理（在 near/far 范围内），属于难以肉眼察觉的静默数据损坏。
4. 附加确认：`cam.get_entity().get_pose()` 返回的是挂载实体（custom_eef_link）位姿而非相机全局位姿（相机另有 CAMERA_EEF_OFFSET_POS=[0.05,0,-0.03] + 30° 俯仰偏移，正确 API 是 cam.get_global_pose()/get_model_matrix()，我已验证两者不同）；SAPIEN 相机前向为 +x（我用 get_model_matrix() 验证：identity 挂载时 OpenGL -z 对应世界 +x），故 `cam_rot_world[:,2]` 确非前向轴。
5. 可达性与后果：vr_teleop_sim.py:518 `camera_enabled = not args.no_camera and not args.headless`，默认启用；每 tick 在 line 860 调 capture_camera_frame，line 881-887 传入 collection.record_frame，episode_recorder.py:467-491 将 camera_frame["depth"] 原样写入 /depth 数据集。默认配置下每个仿真 episode 的 /depth 均被结构性破坏，且不报错、episode 照常标记保存。

唯一的小瑕疵：发现将 `np.all(world_xyz==0)` 标为"wrong invalidity check"——实测背景像素在 camera-frame 下返回 [0,0,0,1]，该掩码碰巧仍然正确工作（有效命中点不可能恰在相机原点）。此点不影响核心结论。

严重性修正：critical → major。理由：该 bug 仅存在于 examples/sim/vr_teleop_sim.py 仿真示例路径；生产真机采集（examples/real/vr_teleop_shm.py + RealSense L515）的 /depth 完全不经过此代码，无安全影响，RGB 不受影响。属于仿真数据路径的静默数据损坏——若仿真 RGBD 数据用于训练则后果严重，但相对本仓库以真机数据采集为核心产出的定位，评 major 更恰当。修复方案（depth = -pos[..., 2]，或直接用 get_picture_names() 中已有的 "DepthLinear" 纹理）我已实证验证正确。
</details>

### `examples/sim/vr_teleop_sim.py:626` — stop_episode 使用 daemon 线程写入 HDF5——退出时被终止，导致 episode 文件截断/损坏
- 来源: vr_sim | 验证置信: high

collection.stop_episode()（第 626-629 行）立即返回，因为底层 recorder.stop_episode()（episode_recorder.py:615-622 行）生成一个 daemon 线程（t = threading.Thread(..., daemon=True)）来执行繁重的 _stop_episode_impl（HDF5 gzip 压缩 + camera 前向填充 + 元数据写入）。在 KeyboardInterrupt（第 643 行）或 finally 块（第 1015-1018 行）触发的退出序列中，collection.stop_episode() 生成 daemon 线程后立即返回，Python 解释器因没有非 daemon 线程而立即退出——daemon 线程在写入过程中被终止，留下 HDF5 文件元数据缺失、数据集部分冲且结构不完整（无 schema_version / success / num_frames 属性）。真实 controller 使用 RecordingSession（recording_session.py:81-84 行），通过 shutdown() + join() 确保写入线程在退出前完成。

<details><summary>验证记录</summary>

逐环节核实，发现属实。(1) episode_recorder.py:615-623 确认 stop_episode() 以 daemon=True 生成 "episode-stop" 线程后立即 return path；全部 HDF5 收尾工作在 _stop_episode_impl（625-717）中：cam writer join(5s)+drain、_flush_buffered()（尾部最多 _flush_interval=10s×control_hz≈500 帧 gzip 数据，episode_recorder.py:66）、camera tail-pad（663-680），而 schema_version/num_frames/success/duration/fps 等 meta attrs 在最末尾（686-697）才写入，之后 _file.close()（699-700）。(2) 唯一的 join 在 start_episode（134-136），仅在开启下一个 episode 时触发，退出路径无效；grep 确认 EpisodeRecorder 无 atexit/__del__/__exit__ 兜底。(3) vr_teleop_sim.py finally 块（1012-1030）：collection.stop_episode（→collection_loop.py:139→daemon spawn）→ tracker.disconnect（DummyTracker 为 pass，dummy_tracker.py:47-48）→ sim.disconnect → main 返回。其余 Python 线程全为 daemon：键盘监听（keyboard.py:85）、Quest 接收线程（vr_tracker.py:83 daemon=True）、cam writer（episode_recorder.py:389-390 daemon=True），因此解释器 finalize 不会被任何非 daemon 线程延迟，episode-stop 线程在下次获取 GIL 时被 CPython 直接终止（标准文档行为）。(4) 可达路径确认：录制中 Ctrl-C→finally:1017-1026；ESC 停录（592）后 Q 退出（603）；Q 自动保存（626）后 2 秒内双击 Q 退出（641-643）——后两条还会损坏"正常保存"的 episode。sim 开启 ee_camera 时（857-860 确实向 record_frame 传 camera_frame:885）daemon 线程需 join cam writer + 图像 forward-fill，耗时可达秒级，远超主线程剩余清理时间，中途被杀几乎必然；无相机时尾部 flush 约数十 ms，与 teardown 同量级，是真实竞态。(5) 后果核实：录制期间有周期 file.flush()（596），故文件多为"可读但不完整"而非彻底损坏——缺 schema_version/num_frames/success attrs、尾帧未刷、camera 未填齐，与发现描述一致；且 sidecar JSON 由主线程同步写出（collection_loop.py:144-145），产生 JSON 存在但 H5 元数据缺失的不一致产物，下游 visualize/export/DataValidator 会失败或拒收。一处不准确但不影响结论：发现称 RecordingSession 通过 shutdown()+join() 保证写完——实际 recording_session.py:103-120 的 _handle_stop 同样经 CollectionLoop.stop_episode 触发同一个 daemon 线程，shutdown()（81-84）只 join session 线程而非 episode-stop 线程，真实路径亦有暴露（controller._shutdown 814-821 走 discard 路径故影响小）；这使问题范围更大而非更小。严重性维持 major：这是数据采集完整性缺陷，且"异常退出尝试保存"的代码给出了虚假的安全承诺。
</details>

### `examples/sim/vr_teleop_sim.py:881` — record_frame 缺少 signals 参数导致 HDF5 中 /flag_ik_ok、/flag_retarget_ok、/flag_held 全部为 False
- 来源: vr_sim | 验证置信: high

collection.record_frame 在第 881-887 行被调用时未传入 signals 参数。EpisodeRecorder.add_frame()（episode_recorder.py:276 行）中 sig = signals or {}，因此 flag_ik_ok、flag_retarget_ok、flag_held 全部默认为 False（episode_recorder.py:320-322 行）。这意味着每个仿真 episode 中的每一帧都显示 IK 失败 + retarget 失败 + 非保持帧，与实际 IK/retarget 状态完全无关。下游基于这些 flag 筛选数据的策略会错误地将所有帧归类为失败。

<details><summary>验证记录</summary>

完整调用链验证确认：

1. sim 端缺失 signals: examples/sim/vr_teleop_sim.py:881-887 调用 collection.record_frame(state=..., action=..., vr_frame=..., camera_frame=..., T_base_eef=...) 时未传入 signals 参数。

2. CollectionLoop 透传: collection_loop.py:91-123 的 record_frame() 在第 111 行将 signals=signals（即 None，默认值）透传给 self.recorder.add_frame()。

3. EpisodeRecorder 默认 False: episode_recorder.py:276 行 sig = signals or {}，signals 为 None 时 sig 为空 dict {}。随后第 320-322 行：
   - flag_ik_ok: bool({}.get("ik_ok", False)) -> False
   - flag_retarget_ok: bool({}.get("retarget_ok", False)) -> False
   - flag_held: bool({}.get("held", False)) -> False

4. status 已计算但未传入: 第 825 行 action, status = pipeline.compute_action(...) 返回的 status 字典包含 ik_ok（第 840 行被读取）和 retarget_ok（第 845 行被读取），这些值仅用于内部失败计数统计（ik_fail_total、retarget_fail_total），从未传入 record_frame()。

5. 真机路径正确: controller.py:388-403 在调用 self._recorder_session.record(dict(...)) 时正确构造了 signals={"ik_ok": ..., "retarget_ok": ..., "held": ...}。

6. 实际后果: 每个仿真 episode 的每一帧 HDF5 中 /flag_ik_ok、/flag_retarget_ok、/flag_held 全部为 False，与实际 IK/retarget 成功状态完全无关。下游基于这些 flag 的数据筛选（如过滤 IK 失败帧）会将所有仿真数据错误归类为全部失败，产生系统性偏差。
</details>


## MINOR

### `examples/real/calibrate_camera.py:318` — 对 Rodrigues 旋转向量做逐元素中值在近 180° 位姿下会产生无效旋转，污染标定样本
- 来源: calib_cam | 验证置信: medium | 原评级: major

detect_aruco_stable 对 5 帧的 rvec 做 np.median(axis=0) 逐分量中值。本场景 marker 贴在末端、正对三脚架相机，camera→marker 旋转角普遍接近 π；cv2.Rodrigues 的规范表示 θ∈[0,π] 在 θ≈π 处不连续，帧间噪声会使 rvec 在 ≈+πu 与 ≈−πu 之间翻转符号；且 SOLVEPNP_IPPE_SQUARE 存在众所周知的平面翻转歧义（文件自己在 113-115 行注释也承认'平面翻转歧义'），逐帧可能在两个歧义解之间跳变。混合了符号翻转/歧义分支的 rvec 做逐分量中值（最少只需 2 帧检出，此时中值退化为算术平均）得到的向量不对应任何真实旋转——'稳定化'反而伪造出坏的 marker 旋转，注入手眼求解输入。后果被一致性门槛部分兜住（该样本残差变大、被拒或需 X 删除），但会无谓浪费采样并可能影响 5 算法比选。tvec 中值没问题，只有旋转部分错。正确做法：按四元数符号对齐后平均，或选与 tvec 中值最近的单帧检测。

### `examples/real/calibrate_camera.py:774` — 采样时不校验机械臂静止：EE 位姿在 ~170ms 的 5 帧 ArUco 采集之后才读取，运动中按 SPACE 会注入时间错位误差
- 来源: calib_cam | 验证置信: high

SPACE 处理顺序是先 detect_aruco_stable（5 帧 @30fps 阻塞 ≈170ms，取中值）再 _get_ee_pose 读一次末端位姿。没有任何机制阻止按住移动键的同时按 SPACE（事件与移动键在同一循环迭代处理），Mode 6 固件在松键后也仍会向最后目标收敛一小段时间。臂在采集窗口内移动时，marker 位姿（5 帧中值）与 EE 位姿（末尾单次读取）描述的不是同一时刻，按 0.25m/s 的最大键控速度可注入数十 mm 级样本误差，且中值还会把运动模糊进 marker 位姿。ArmInnerLoop 已提供现成的静止判据（get_dynamics 的 qvel、tracking_error 属性，inner_loop.py:181-206），但未被使用。误差最终会被一致性门槛暴露，但表现为'莫名残差大'而非明确拒绝原因。

### `examples/real/calibrate_camera.py:822` — ENTER 标定路径无异常保护：退化数据可致未捕获异常，整场会话崩溃并丢失全部已采样本
- 来源: calib_cam | 验证置信: high | 原评级: major

ENTER 分支（816-875 行）直接调用 calibrate_and_select / save_cameras_json，外层 try 只捕获 KeyboardInterrupt（996 行）。可触发的未捕获异常：(1) 全部 5 种算法抛 cv2.error 时 calibrate_and_select 主动 raise RuntimeError('所有手眼算法均失败')（510 行）；(2) calibrate_and_select 内部 try 只 except cv2.error（502 行），compute_marker_consistency 里 scipy R.from_matrix/rots.mean() 对退化算法（如 ANDREFF 在低旋转多样性下返回非正交/非有限 R）抛的 ValueError 会穿透；(3) save_cameras_json 中 json.load 读到损坏的旧 cameras.json（541 行）。任一异常都会终止主循环 → finally 清理 → 10-20 组手工采集样本全部丢失且无法重试。与项目 fail-safe 约定（错误→warning+fallback）相悖；SPACE 分支已按此约定加了 try/except 并注明'保住会话与已采集样本'（768-790 行），ENTER 分支却没有。硬件侧无危险（finally 会 stop 内环 + disconnect）。

### `examples/real/calibrate_camera.py:1001` — 初始化段不在 try 保护内且 finally 清理链未逐项隔离：相机异常会跳过 arm_inner.stop()/robot.disconnect()
- 来源: calib_cam | 验证置信: high

两个对称缺口：(1) ArmInnerLoop.start()（649 行）到 try（745 行）之间的 RealSense 启动/30 帧预热（667-693 行，`pipeline.wait_for_frames()` 按项目 L515 停流记录是现实故障点）若抛异常，会绕过 finally——内环线程仍在运行、robot 未 disconnect，仅靠 daemon 线程随进程退出被杀。(2) finally 块（998-1014 行）内 `pipeline.stop()`（1001 行）在 arm_inner.stop()/robot.disconnect()（1011-1014 行）之前且未包 try：相机中途拔线/停流时 pipeline.stop() 抛 RuntimeError 会跳过后续机械臂清理与 tty 恢复。Mode 6 固件断连后保持位置，无直接硬件危险，但违背项目'ExitStack for cleanup'约定，且 tty 不恢复会污染终端。建议每步清理独立 try 或改用 contextlib.ExitStack。

### `examples/real/calibrate_l515_depth.py:214` — 定时等待循环存在负数 sleep 竞态：ValueError 崩溃并（经由发现 1）丢弃全部数据
- 来源: calib_l515 | 验证置信: medium

213-214 行 `while time.monotonic() < target: time.sleep(min(1.0, target - time.monotonic()))`——while 条件判断与 sleep 实参求值之间若被调度抢占跨过 target，`target - time.monotonic()` 变为负数，time.sleep(负数) 抛 ValueError: sleep length must be non-negative（已在本机 Python 实测确认）。25 分钟无人值守会话中每个 interval 都要过这个窗口，概率虽低但一旦命中即整场崩溃且 JSON 未写。改为 `time.sleep(min(1.0, max(0.0, target - time.monotonic())))` 即可。

### `examples/real/calibrate_l515_depth.py:346` — 标定在 1024x768 深度流上拟合 sigma_poly，生产 SHM 路径在 640x480 上消费——分辨率域不一致
- 来源: calib_l515 | 验证置信: high

脚本 346 行 depth_resolution=(1024, 768)，注释（51-52 行）称 mirror 的是 examples/real/test_pointcloud_process.py（诊断脚本，确为 XGA）。但 sigma_poly 的真正生产消费方是 CameraProcess：CameraProcessConfig 默认 depth_width/height=640/480（并注明是为省 8ms validity-gate 开销而特意从 XGA 改的），validity gate（含用 sigma_poly 建 LUT 的 edge 阈值）跑在 640x480 原始深度上。L515 的 VGA 深度由固件从原生 XGA 缩放而来，每像素时域噪声特性与 XGA 不同，故 XGA 上拟合的 sigma_z(z) 用于 VGA 阈值存在系统偏差。实际影响被 t_min=10mm floor 部分兜底（生产注释自述 z≲1.0m 内 floor 主导），但 1.0-1.5m 段 sigma 项起作用，且脚本 docstring 声称的"mirror production"对深度分辨率并不成立。建议标定分辨率与 CameraProcess 生产值对齐，或两档各测。

### `examples/real/keyboard_teleop_real.py:441` — C24 (Speed Exceeds Limit) 被误判为硬故障触发急停
- 来源: kbd_real | 验证置信: high | 原评级: major

line 427-443 将所有非 22 错误码统一执行急停。但 C24 (Speed Exceeds Limit) 在 ArmInnerLoop._RECOVERABLE_ERRORS (inner_loop.py:107) 中明确分类为可恢复错误：内环遇 C24 时仅保持位置并清除目标，不设 error_state=True。然而外层键盘脚本在 line 427 通过 robot.arm.is_error() 检测到 error_code=24 后，因 arm_code != 22 走 line 441 _emergency_stop()，错误地急停而非恢复。

### `examples/real/keyboard_teleop_real.py:594` — 未调用 validate_action() 安全门，跳过 joint-limit 裁剪、torque/temperature 门控
- 来源: kbd_real | 验证置信: high | 原评级: major

脚本 line 594 arm_inner.set_target(arm_cmd) 和 line 600 robot.send_action(action) 均未先调 validate_action()。CLAUDE.md 要求 send_action 前必调此门，TeleopController (controller.py:346) 生产路径始终调用。缺失的 joint-limit 软限位裁剪 (validate.py:84-86) 是 ArmInnerLoop 不具备的防御：delta clamp (max_joint_delta) 不裁剪绝对位置超限。虽 IK 通常产出合法解，但防御纵深被跳过。

### `examples/real/replay_traj.py:717` — 只有臂命令有 NaN 防护，手命令 NaN 会穿透库内 np.clip 直达硬件并永久污染 E3 delta clip
- 来源: replay_traj | 验证置信: high

循环对 arm_cmd 做了 `np.all(np.isfinite(...))` 兜底（NaN→当前 qpos，符合 fail-safe 约定），但 hand_cmd 无任何检查。HDF5 中若某帧 action_hand_joint 含 NaN：XHand.send_action 的 _limit_joint_range 用 np.clip，NaN 经 clip 仍是 NaN；E3 delta clip 中 `delta = NaN - last` 也是 NaN；write_command_positions 以 float(NaN) 逐关节写入 SDK 命令，直接把 NaN 发给手硬件（行为未定义），且成功发送后 last_qpos_cmd 被置为 NaN，此后所有帧的 delta clip 全部失效（NaN 传染）。应与臂命令对称：hand_cmd 非有限时跳过该帧手命令。

### `examples/real/replay_traj.py:818` — _emergency_stop 在急停后立即 clear_error()，几毫秒内解除急停闩锁并重新使能电机
- 来源: replay_traj | 验证置信: high | 原评级: major

ESC/错误路径调用 `self.robot.emergency_stop()`（XArm7.stop → set_state(4) 并置 error_state=True，即急停闩锁）后紧接 `self.robot.arm.clear_error()`——其实现为 clean_error + clean_warn + motion_enable(True) + set_state(0)，等于立刻撤销急停：臂回到就绪态、真实硬件故障码也被抹掉。生产控制器 _escalate_to_emergency 只做 set_target(None)+stop()+emergency_stop()，绝不随后清错。虽然此刻内环已停、无新命令，物理上臂保持不动，但操作员按 ESC 期望的是'臂被锁死直至人工干预'，而当前实现让臂处于 motion-enabled 就绪态；且后续 H 回家路径重新 connect() 时本来就会自行 clean_error，这里的 clear_error 纯属多余且有害。

### `examples/real/replay_traj.py:1013` — 收尾 H 回家后重启的新 ArmInnerLoop 永不 stop：泄漏线程与 XArmAPI 连接，并把刚归位的臂重新切回 Mode 6 空转
- 来源: replay_traj | 验证置信: high

main 的 finally 在 replayer.shutdown()（内环已停、robot 已断开）之后，H 键路径经 _do_return_home 回家并 `new_inner = ArmInnerLoop(cfg=_INNER_CFG); new_inner.start()`（line 555-557）——新内环建立自己的 XArmAPI 连接、把臂从 reset() 留下的 Mode 0 重新切到 Mode 6，并在 target 超时后以 50Hz 持续 _hold_position。随后 main 只做 `r.disconnect()`（line 1015），新内环从未被 stop()/join：按 Q 退出时它作为 daemon 线程被解释器硬杀，_run 的 finally（arm.disconnect）不执行，连接与 Mode 6 状态被遗弃。该重启模式抄自 teleop 入口（回家后继续遥操作才需要），在'回家即退出'的回放脚本里纯属多余——_do_return_home 不应重启内环，或退出前应 stop 新内环。

### `examples/real/replay_traj.py:1013` — post-loop 归位后重启的 ArmInnerLoop 无人停止，进程退出时 daemon 线程被强杀
- 来源: x-safety | 验证置信: high

finally 的 post-loop 交互中按 H 会经 _do_return_home 创建并 start 一个新 ArmInnerLoop（new_inner，持有独立 XArmAPI 连接、50Hz 伺服），随后 `r.disconnect()` 只断开临时 RobotInterface；用户按 Q break 后 main() 直接结束——new_inner 从未被 stop()，作为 daemon 线程随进程退出被强杀，inner_loop.py:441-445 finally 里的 arm.disconnect() 不保证执行，臂被留在 mode 6 伺服态、连接未优雅关闭。replayer.shutdown()（994）在此之前已执行，管不到这个新实例。

### `examples/real/test_motion_planning_real.py:293` — Test 2 随机游走漏传 rng：rot_max_deg=RANDOM_ROT_DEG 是死参数，姿态从未随机化；且 waypoint 预验证姿态与实际执行姿态不一致
- 来源: mp_real | 验证置信: high

test_solve_teleop_ik 行 293 调 build_target_pose(pos, home_eef.q, rot_max_deg=RANDOM_ROT_DEG) 未传第三个位置参 rng，函数在行 83-84 `if rng is None: return Pose(p=pos, q=quat)` 提前返回 home 姿态，显式传入的 rot_max_deg 完全无效——与 Test 1（行 236 传 rng）和 Test 4（行 521 传 rng）不一致，说明大概率是漏传而非有意；Test 2 的成功率/max_dq 统计因此只覆盖固定姿态，比表面弱。同主题：validate_waypoint_ik（行 448-449）用 home 姿态 Pose(p=pos, q=home_eef.q) 预验证可达性，而 run_waypoint_test 实际执行的目标带 ±30° 随机旋转（行 521），预验证覆盖不了真实执行目标，旋转后 plan 失败只能在运行期暴露（行 534-536 已处理，不危险但预筛选失去意义）。

### `examples/real/test_motion_planning_real.py:733` — safe_return_home 的规划结果纯属摆设：seg_check 分支是死代码，且越危险越直接盲走 reset
- 来源: mp_real | 验证置信: high

safe_return_home 里 plan_path 成功后再调 planner.check_path_collisions(result.qpos_path) 判 path_self_collision——但 plan_path 成功本身已经在 _check_self_collision（planner.py:514-530）里对同一条最终路径用同一默认步长 0.02 rad 跑过 check_path_collisions，成功即意味着 path_self_collision=False，因此行 733-743 的"segment collision → 直接 reset"分支不可达（死代码）。更根本的问题：规划出的无碰撞路径从不被执行，实际动作始终是 arm.reset(home_qpos)——Mode 0 固件关节空间直线插值（xarm7.py:283-292），与规划路径无关；当 plan_path 失败（找不到无碰路径，本应是危险信号）时（行 745-746）只打印 WARNING 后照样直接 reset。叠加发现 1 中 desk/env 层缺失，从低位/别扭构型归位时关节直线扫掠不受任何碰撞约束。对照生产归位（interface.py _execute_waypoints + _check_joint_path_safe，含 desk 检查后按检查过的路径逐点执行）。测试脚本有人监督下风险有限，但"safe_"前缀名不副实。

### `examples/real/test_motion_planning_real.py:972` — 异常/Ctrl-C 清理只有 disconnect：无 stop()、无归位尝试，阻塞 reset 期间中断后固件仍继续运动
- 来源: mp_real | 验证置信: high

main 的 finally（行 972-973）仅调 arm.disconnect()（xarm7.py:135-138，只关 SDK 连接）。执行期任意异常或 KeyboardInterrupt：(1) 在 execute_path_on_arm 中间中断——机械臂停在任意 waypoint，无提示、无归位（Mode 1 最后目标≈当前位置，本身会停住，后果有限）；(2) 在 arm.reset() 阻塞等待（set_servo_angle wait=True，xarm7.py:283-292，20°/s）期间 Ctrl-C——运动指令已下发固件，Python 退出/断连后固件继续把该次运动跑完，脚本对此既不 stop()（set_state(4)，xarm7.py:185-190，存在但从未在清理路径调用）也不告知。reset 目标是 home、幅度受限，故为 minor；但按仓库"急停路径缺口"标准，finally 里至少应在异常分支尝试 arm.stop() 或提示当前位姿。

### `examples/real/test_pointcloud_process.py:39` — 原始深度分辨率与生产不一致：诊断脚本 XGA (1024x768)，生产 CameraProcess 为 640x480；stream 测试 docstring 还错误声称 XGA
- 来源: pointcloud | 验证置信: high

本脚本以 DEPTH_RESOLUTION=(1024,768) 采集原始深度，而生产 CameraProcessConfig 默认 depth_width=640/depth_height=480（camera_process.py 注释：与 color 同分辨率省 8ms gate 开销）。validity gate（confidence/IR/edge）在对齐前的原始深度域运行，饱和 dilate 3px、edge 3x3 邻域等像素域参数在两种像素间距下作用范围不同，XGA 上的验证结论（95% valid 等）外推到生产 VGA 未经确认。另外 test_pointcloud_stream.py:4 docstring 声称 'CameraProcess child (XGA depth + ...)'，但它不覆盖默认值，实际按 640x480 运行——描述已过时。

### `examples/real/test_pointcloud_process.py:361` — Z 下界与生产不一致：测试 z_lo=0.0 保留桌面，生产 workspace z_min=0.005 去除桌面，下游各阶段统计/计时在不同点分布上得出
- 来源: pointcloud | 验证置信: high

测试最终 Z 裁剪为 [0.0, 0.8]（行 361，前置 guard 裁剪 z∈[-0.2,0.8] 行 305），桌面平面保留进入体素降采样、离群滤除和 2048 点 FPS；生产 workspace=(0.0,-0.6,0.005,0.8,0.6,0.8)，注释明确 'z_min = 0.005 removes the desk surface itself'。后果：测试中 2048 点预算大部分落在桌面上，FPS 计时（'~9 ms @ 5.3k pts'）和滤波前后点数都不代表生产（纯物体点云）的工况；且行 47-51 '贴桌 2x2cm 物体 16/16 保留' 的验证依赖物体与桌面簇合并（'merges into that structure's cluster'），生产既无桌面也无 DBSCAN，该验证链完全失效。

### `examples/real/test_pointcloud_process.py:380` — 离群点滤除算法与生产路径不一致：测试用 DBSCAN 簇大小过滤，生产用半径离群点移除，测试验证结论不能迁移到生产
- 来源: pointcloud | 验证置信: high | 原评级: major

测试脚本用 cluster_dbscan(eps=0.01, min_points=1) + 丢弃 <30 点的连通簇（CLUSTER_MIN_SIZE=30，行 52-53），而生产 PointCloudProcessor 用 remove_radius_outlier(nb_points=3, radius=0.01)。两者保留/丢弃行为差异巨大：测试注释（行 47-51）声称验证过 '300 specks, 10-pt string, 8-pt clump all removed'，但在生产的半径滤波下，8 点密集团簇和 10 点串中每个点在 1cm 半径内都有 ≥3 个邻居，会全部存活进入录制的 /pointcloud 训练数据；反之孤立的 <30 点真实小物体在测试里被删、生产里保留。更严重的是 pointcloud_processor.py 模块 docstring（行 3-6）声称该流水线 'validated in examples/real/test_pointcloud_process.py ... -> radius outlier removal -> fixed-size sample'——本脚本从未执行过 radius outlier removal，生产滤波阶段实际处于未验证状态。

### `examples/real/test_pointcloud_stream.py:66` — 轮询循环不检查 camera.crashed，子进程崩溃后测试盲跑满 --duration 才报 FAIL 且不给原因
- 来源: pointcloud | 验证置信: high

CameraProcess 专门提供 crashed 属性（子进程 connect 失败或采集循环崩溃时置位，且 parent 侧会在 is_alive()==False 时置位），但测试主循环只调用 poll_latest_frame()。若子进程启动即失败（如 RealSense connect 失败），poll 永远返回 None，测试空转整个 duration（默认 60s）后仅打印 'FAIL: no frames received.'；若中途崩溃，poll 持续返回最后一帧（frame_number 不变），逐秒统计静默停止，直到超时才在汇总里体现为低帧率。循环内加一条 `if camera.crashed: break` 即可秒级失败并指明原因。

### `examples/real/test_pointcloud_stream.py:80` — --vis 路径 add_geometry 条件挂在首帧而非首个有效点云，首帧无效则整程黑窗
- 来源: pointcloud | 验证置信: high

行 80 用 `len(frame_times) == 1`（第一个新帧）决定 add_geometry vs update_geometry，但外层条件（行 75）要求 pointcloud_valid 为真才进入渲染块。若第一个新帧的点云无效（生产者在首个有效云出现前发全零 pc_num_points=0，见 camera_process.py:306-308 'before the first valid cloud, send zeros'；或首帧恰逢空云），则首帧跳过渲染块、frame_times 已计入 1；之后有效帧到来时 len(frame_times)>1，只调用 update_geometry——对从未 add 过的几何体，open3d 的 update_geometry 静默返回 False，整个运行窗口保持空白。应以'是否已 add' 的独立标志判断。

### `examples/real/test_quest_hand_teleop.py:153` — 手部 delta clip 与控制频率与生产入口不一致，测试结论对 16Hz 生产不成立
- 来源: quest_hand | 验证置信: high | 原评级: major

本测试 CONTROL_HZ=50（第 71 行）且 XHandConfig 未指定 max_delta_rad（第 153-156 行），落到库默认 0.3 rad/step（xhand.py:207），@50Hz 等效速度上限 ≈15 rad/s（859°/s）。生产入口 vr_teleop_shm.py 已迁移到 CTRL_HZ=16 并显式派生 XHandConfig(max_delta_rad=deg2rad(90)/16≈0.098 rad/step)，速度语义 90°/s——两者差 9.5 倍。此外循环频率本身 50 vs 16 使每帧地标位移大 3 倍、NLP warm-start 步长与 delta-clip 触发行为完全不同。用此脚本调参/验证重定向（额外重点关注项）得到的平滑性、响应性、clip 触发结论无法迁移到生产。注：LPFilter alpha 恰好 τ-一致（测试 YAML 默认 0.6@50Hz ↔ 生产 alpha_from_tau 换算 0.943@16Hz，同 τ≈21.8ms，hand_retarget.py:383 + signal_utils.py alpha_from_tau），仅 delta clip 与频率是实质分歧。文件头虽标 DEPRECATED，但 CLAUDE.md 仍将其列为'Standalone hand-retargeting test'入口。

### `examples/real/test_quest_hand_teleop.py:223` — VR 接收线程在首帧到达前死亡时无法检测，主循环无声空转
- 来源: quest_hand | 验证置信: high

线程死亡退出检查（L233-235 `if not status["running"] and not status["started"]: break`）被嵌套在 `if had_first_frame and (...)` 门内（L223）。默认 tcp_server 模式下 connect() 只验证线程存活即返回 True（vr_tracker.py:108-111），若接收线程随后在收到首帧前因 ConnectionError/RuntimeError 崩溃（_receive_loop 的 finally 置 running=False、started=False，vr_tracker.py:246-250），主循环将以 50Hz 永久空转：get_latest() 恒为 None、had_first_frame 恒为 False、无任何告警输出，只能靠用户手动按 q 退出。线程死亡检查应移出 had_first_frame 门。

### `examples/real/test_quest_hand_teleop.py:341` — finally 中 _save_recording 先于归位/断连执行，保存异常会跳过全部硬件清理
- 来源: quest_hand | 验证置信: high

清理顺序为：restore stdin(L338) → _save_recording(L341) → 手归位(L343-355) → xhand.disconnect(L356) → tracker.disconnect(L357)。_save_recording 内 np.savez_compressed 可因磁盘满/目录不可写抛 OSError；另外 L98 只检查 timestamps 非空，若 KeyboardInterrupt 恰落在首帧 L268（timestamps.append）之后、L269 之前，则 landmarks_raw 为空列表，L110 `np.stack([], axis=0)` 抛 ValueError('need at least one array to stack')。任一异常从 finally 传播后，手停在最后遥操作姿态不归位、EtherCAT 设备句柄不关闭、VR 接收线程不 join。保存应包 try/except 或移到硬件清理之后。

### `examples/real/test_quest_hand_teleop.py:344` — 真机手测试入口归位注释两处失实（3.6°/step、2.4s@50Hz），且手限速仍是旧 859°/s 语义
- 来源: x-16hz | 验证置信: high

归位循环注释称 "delta limit caps it to ~3.6°/step"，但库默认 XHandConfig.max_delta_rad=0.3 rad=17.2°/step，实际归位单步可达 17.2°（等效 ~515°/s 收拢，非注释暗示的 ~108°/s 缓收）；注释又称 "up to ~2.4s at 50Hz"，但循环 sleep 的是 xhand.config.dt=1/30s → 实为 30Hz、上限 4s。另外主循环真 50Hz（CONTROL_HZ=50 + RateLimiter），XHandRetargeter 默认 α=0.6@50Hz 自洽，但 clip 用库默认 0.3 rad/step ≙ 859°/s，未采用迁移定案的 90°/s 手限速语义——同一实体手在本入口与生产入口（vr_teleop_shm.py:70,133）等效限速差 9.5 倍。测试脚本行为与迁移前一致故定 minor，但注释会让操作者低估归位速度。

### `examples/real/test_quest_hand_teleop.py:347` — 归位循环速度上限 ≈515°/s（远超生产 90°/s 语义），注释的 3.6°/step 与 50Hz/2.4s 全部与实际参数脱节
- 来源: quest_hand | 验证置信: high

L345 注释称'delta limit caps it to ~3.6°/step'，该数字来自旧的 180°/s@50Hz 推导（deg2rad(180)/50≈0.063 rad）；当前库默认 max_delta_rad=0.3 rad≈17.2°/step（xhand.py:207），大 4.8 倍。L353 用 `time.sleep(xhand.config.dt)` 步进，而 config.dt=1/30（xhand.py:82），故 120 步≈4s 而非注释的'~2.4s at 50Hz'（L347）。实际归位关节速度上限 0.3 rad/33ms≈9 rad/s≈515°/s——从抓握姿态一次甩回 home，是生产 90°/s 手部限速语义的 5.7 倍。注释基于错误参数论证了循环设计的安全性，实际防护远弱于声称值；16Hz 迁移后的速度语义（vr_teleop_shm.py L70-73 明确记录了 0.3 rad 旧默认的换算关系）未同步到此清理路径。

### `examples/real/test_realsense.py:60` — workspace 裁剪在相机光学坐标系中执行，但边界值按世界系语义书写，与生产语义不一致
- 来源: test_rs | 验证置信: high

脚本调用 rgbd_to_pointcloud 时未传 T_out_camera（行 309-312、539-541），库内流程是 depth_to_xyz → transform_points(None 直通) → crop_points，因此 workspace 裁剪发生在相机光学系（x 右、y 下、z 前）。而 DEFAULT_PCD_WORKSPACE=(-0.3,-1.0,-0.5, 2.0,1.0,1.5) 的非对称 x∈[-0.3,2.0] 是典型的机器人基座/世界系边界写法：在相机系中它会裁掉光轴左侧 0.3m 以外的所有点，z∈[-0.5,1.5] 则与 max_depth=1.5 重复。生产路径（pointcloud_processor.py:71-117）是先乘 T_world_camera 再在世界系裁剪（workspace=(0.0,-0.6,0.005,0.8,0.6,0.8)）。后果：用 'w' 键在本测试里调出的 workspace 观感/点数结论无法迁移到生产配置，且裁剪方向性容易误导标定。

### `examples/real/test_realsense.py:339` — 性能汇总的 total_ms/fps 不含 imshow、open3d 渲染与 waitKey 开销，avg fps 会被高估
- 来源: test_rs | 验证置信: high

total_ms 在行 339 计算（loop_start 到此为止），但 cv2.imshow（行 371）、pcd_viewer.update（行 374，含 Vector3dVector 重建 + poll_events/update_renderer，sampling=none 时可达数十万点）、cv2.waitKey（行 381）都在计时点之后。当渲染耗时占主导时，pipeline 内部队列会缓冲帧使下一次 read() 立即返回（read_ms 变小），于是 avg_total 偏小、avg_fps=1000/avg_total 高于真实循环帧率。步骤 4 的"性能汇总"（行 469-476）据此打印的 avg fps / avg frame total 失真。测量应把 total_ms 的计时点移到 waitKey 之后。

### `examples/real/test_realsense.py:577` — 测试未启用生产标定的 L515 深度有效性门控，valid_ratio/点云数据与录制路径不可比
- 来源: test_rs | 验证置信: high

RealSenseConfig（行 182-187 与 577-582）未传 depth_validity，落在库默认 None（realsense.py:105），即不开 confidence/IR/edge 门控。而生产 SHM/录制路径（camera_process.py:255-261）传入 L515_CALIBRATED_DEPTH_VALIDITY（realsense.py:73-85，2026-07-15 标定的 sigma_poly 边缘门控，注释称其为 SHM/录制路径的 single source of truth）。后果：本脚本作为相机验收测试，其 HUD 与汇总打印的 avg valid depth、点云点数、pcd 耗时都是未门控深度流的数字，与实际录进 HDF5 的门控后深度不一致（门控会额外置零低置信/饱和/边缘像素），用它做验收会高估生产有效深度比例。

### `examples/real/vr_teleop_arm_only.py:323` — 50Hz 遗留采集入口与 16Hz 生产入口写同一数据目录，形成混速率 episode 池
- 来源: x-16hz | 验证置信: high | 原评级: major

vr_teleop_arm_only.py 以 50Hz 实际循环录制（RateLimiter(1/0.02)=50Hz），EpisodeRecorder 未传 control_hz → 库默认 50，写入 data_dir="episodes"——与 16Hz 生产入口 vr_teleop_shm.py（data_dir="episodes", control_hz=16）同一目录；episodes_arm/ 同理：vr_teleop_arm_only_record.py:330（50Hz）与 vr_teleop_arm_only_record_plus.py:344-346（16Hz）共写；vr_teleop_sim.py:111 DEFAULT_DATA_DIR="./episodes" 是第三个 50Hz 写入者。叠加 export_hdf5_to_zarr 已确认的零速率元数据+静默拼接缺陷（arch-review #3），任一遗留入口跑过一次后批量导出即得 50/16Hz 无标记混拼的训练集。判定：16Hz 侧（shm/record_plus）是生产标准；50Hz 侧应改目录、迁移或标注弃用。

### `examples/real/vr_teleop_arm_only.py:419` — ESC 急停后立即 clear_error()，motion_enable(True)+set_state(0) 解除急停锁存
- 来源: arm_only | 验证置信: high | 原评级: major

_emergency_stop() 在 robot.emergency_stop() 之后紧接着调用 robot.arm.clear_error()。XArm7.stop() 的语义是 set_state(4) 并锁存 error_state=True（急停锁存）；而 XArm7.clear_error() 无条件执行 clean_error() + clean_warn() + motion_enable(True) + set_state(0)，即急停刚生效就把臂拉回 READY 并重新使能电机，急停锁存被当场抵消。虽然此后 running=False 不再发指令，但『急停后臂处于使能待命态』削弱了 E-stop 的安全语义（如需为退出后的 H 归位清错，应在归位前清，而不是在急停瞬间清）。

### `examples/real/vr_teleop_arm_only.py:419` — ESC 急停后立即自动 clear_error——急停不闩锁，运动使能被马上恢复
- 来源: x-safety | 验证置信: high

_emergency_stop() 在 robot.emergency_stop()（set_state(4) 停止）之后无条件立刻 robot.arm.clear_error()，而 XArm7.clear_error()（xarm7.py:170-173）会 motion_enable(True)+set_state(0) 重新使能运动并清掉错误闩锁。对照 TeleopController 的设计（controller.py:592-600 急停只停；480-487 仅当操作员按 H 才 clear_error 再归位），示例把『解除急停』从操作员显式动作变成自动行为，ESC 语义被弱化，也抹掉了诊断现场。同 pattern：vr_teleop_arm_only_record.py:434-435、vr_teleop_arm_only_record_plus.py:450-451、replay_traj.py:817-818。（急停后 running=False 循环退出，无后续命令下发，故为设计弱化而非失控。）

### `examples/real/vr_teleop_arm_only.py:491` — B/C-恢复用 read_latest() 建立 wrist→EEF 映射基准时不检查帧新鲜度，过期基准导致恢复瞬间目标跳变
- 来源: arm_only | 验证置信: high

B 处理（L491-495）与 C 恢复（L478-486）只判 frame is None 就用该帧 wrist 做 arm_mapper.reset() 基准。SHM 环形缓冲 read_latest 返回『最后写入过的帧』，VR 断流时可能是几分钟前的旧帧。此后主循环因 vr_stale（L587，0.5s 阈值）保持不动，但一旦新鲜帧到达，delta = 新 wrist − 过期基准可能很大；且本脚本 ArmWristMapper 未配置 eef_delta_bounds（L316-320），clip_delta_pos 为 no-op，位置增量只受 workspace clamp 封顶（旋转有 1.0 rad/帧限幅），EEF 目标会一步跳到工作区边界方向。建立基准时应复用 L587 的 local_recv_ns 新鲜度判定。

### `examples/real/vr_teleop_arm_only.py:498` — start_episode() 无参数调用，task_label/operator/record_config 全部缺失
- 来源: x-recording | 验证置信: high

recorder.start_episode() 无任何参数，/meta 中 task_label=''、operator=''、无 record_config 属性。使该条目在混合数据集中难以标识且无法复现控制参数。不影响功能，但 metadata 不完整。

### `examples/real/vr_teleop_arm_only.py:540` — wait_ready 超时宣称『降级为直接读取』但主循环未实现：每拍仍消费内环可能的全零 qpos
- 来源: arm_only | 验证置信: medium

启动 wait_ready(30s) 超时只把『初始』state 改为直接 SDK 读取（L288-297），主循环每拍仍固定用 arm_inner.get_state() 的 qpos（L540→L551）。ArmInnerLoop._run 在初次 get_joint_states 失败时把 current_qpos 置 zeros(7) 且照常 set ready（error_state=False），此时零向量是有限值，能通过 L579 的 isfinite 守卫，state 变成 FK(zeros) 的错误 EEF；hold 分支还会把 prev_qpos_cmd 刷成 zeros（L646），若操作者此窗口按 B，映射基准与 IK 种子全错，内环恢复后会驱动真臂朝错误目标运动（受 0.3rad/步 clamp 与固件 90°/s 限速，但仍是真实错误运动）。启动/归位路径有 `np.all(arm_qpos == 0)` 守卫（L294、L451），每拍路径应加同样守卫或真正落实降级读取。

### `examples/real/vr_teleop_arm_only.py:563` — 示例直接调用 robot.arm.* 甚至 robot.arm.arm.error_code，绕过 RobotInterface 门面
- 来源: x-safety | 验证置信: high

错误检查/恢复逻辑直接操作 XArm7 驱动层：`robot.arm.is_error()`、`robot.arm.arm.error_code`（穿透两层直达裸 SDK 对象）、`robot.arm.clear_error()`（268/419/569 行）。CLAUDE.md 规定硬件访问 ONLY via RobotInterface；实际行为差异：robot.arm.clear_error() 只清臂不清手，而 RobotInterface.clear_error()（interface.py:139-150）同时清 arm+hand。同 pattern 遍布 vr_teleop_arm_only_record.py:635-647、record_plus.py:680-693、keyboard_teleop_real.py:427-433、replay_traj.py:744-749。C22/C24 专项恢复在库层无对应门面方法，属实用性妥协，但至少 error_code 读取应有封装。

### `examples/real/vr_teleop_arm_only.py:753` — 退出路径缺口：finally 内无限阻塞等按键、stop_episode 后台 daemon 线程无人 join、二次 Ctrl-C 跳过 disconnect/SHM unlink
- 来源: arm_only | 验证置信: high

finally 块先进入 `while True: kb.poll(timeout=0.1)` 无超时出口——pynput 监听失效（无 X/权限）时进程永久挂死在退出段；此时或用户在该提示处按 Ctrl-C，KeyboardInterrupt 会从 finally 内部抛出，跳过后面的 robot.disconnect() 和 vr_receiver.stop()（后者是唯一执行 shm.close()+shm.unlink() 的地方 → /dev/shm 段泄漏，影响下次启动）。另外 stop_episode() 把 HDF5 flush/close 放到 daemon 线程（episode-stop）而脚本从不 join，若清理路径走得快，进程退出会硬杀该线程，meta（num_frames/success/schema_version）可能未写、文件未 close。

### `examples/real/vr_teleop_arm_only.py:753` — finally 块内交互式 while True 等按键——Ctrl-C/异常路径无法让程序安全终止
- 来源: x-safety | 验证置信: high | 原评级: major

四个入口的 finally 块在清理硬件之前先进入无限循环等待 H/Q 按键（vr_teleop_arm_only.py:752-759、vr_teleop_arm_only_record.py:854-861、vr_teleop_arm_only_record_plus.py:898-905、keyboard_teleop_real.py:627-634）。后果：(1) 操作员第一次 Ctrl-C（KeyboardInterrupt）不会退出，而是落入这个隐藏的交互提示，此时 arm_inner 内环线程仍存活、臂保持伺服使能；(2) 第二次 Ctrl-C 在 finally 内抛出，直接跳过其后的 kb.stop()/arm_inner.stop()/robot.disconnect()/vr_receiver.stop()/camera.stop()——内环是 daemon 线程被强杀，其 _run 的 finally arm.disconnect()（inner_loop.py:443）不保证执行，臂被留在 mode 6 伺服使能态且所有清理被绕过；(3) 主循环内未捕获异常（try 只包了 get_state 段）同样先阻塞在此提示，traceback 要等交互结束后才打印。

### `examples/real/vr_teleop_arm_only_record.py:148` — 两个真机入口决策环仍 50Hz + 每 tick solve_teleop_ik，保留了迁移文档已实证的超预算事故链
- 来源: x-16hz | 验证置信: high | 原评级: major

vr_teleop_arm_only_record.py 与 vr_teleop_arm_only.py 的主循环实际节拍是 50Hz（CTRL_DT=0.02 + RateLimiter），每 tick 在真机上调用 solve_teleop_ik 并录数据。16hz-rationale 文档实证：IK 基线 27ms = 50Hz 预算 20ms 的 135%，慢性超期曾导致相机写入错位累积 20s（ep_211557）并参与 C24 急停链——这正是迁移到 16Hz 的核心动因。迁移只覆盖 shm 与 record_plus（"16 仅入口点单点定义"），这两个同硬件同用途的采集入口既未迁移也无 deprecated 标注，今天运行仍复现慢性超期工况并产出 50Hz 数据（与发现 1 叠加）。

### `examples/real/vr_teleop_arm_only_record_plus.py:607` — record_plus 硬编码 camera_name="camera_0"，与子进程按 serial 独立解析产生不一致风险
- 来源: x-triplet | 验证置信: high

B 通过 _resolve_camera_name() 从子进程共享的 camera_serial 反向查 cameras.json 得到正确条目名。C 直接硬编码 "camera_0"。子进程 (camera_process.py L439-441) 独立按 serial 解析 extrinsics 烘焙点云世界坐标。若实际连接的相机 serial 与 cameras.json 中 "camera_0" 不匹配，/meta 写入的是 camera_0 的 extrinsics，而点云数据用的是正确相机的 extrinsics——产生元数据不一致。此外 EpisodeRecorder._write_meta_attrs (L215) 校验 expected_serial 的逻辑因 _pending_meta 从未填入 camera_serial 而实质失效，责任完全落在调用方传入正确 camera_name。B 的 serial 解析方式是正确的。在单相机 rig 且始终 camera_0 对应该 serial 的场景下 C 无实际影响，故降为 minor。

### `examples/real/vr_teleop_arm_only_record_plus.py:873` — flag_held 在录制中始终为 False——held 态帧被跳过而非标记
- 来源: record_plus | 验证置信: high

add_frame 的信号 dict（L873）hardcode held=False。但 held 态（暂停/VR 过期/IK 失败）发生时，对应帧根本不调用 add_frame（L768-777/L832-834 的 continue 跳过录制）。因此 flag_held 在 HDF5 中始终为 False，而 schema（CLAUDE.md:352 行）将其记录为 bool 标志。对比 TeleopPipeline 会在 VR 过期或 IK 失败时设 held=True 并录制该帧（从而保留 held 语义）。下游训练代码若依赖 flag_held 过滤无效帧，将获得全部 "非 held" 数据。

### `examples/real/vr_teleop_shm.py:149` — arm 连接失败早退路径不调用 robot.disconnect()，已连上的 XHand 句柄泄漏
- 来源: vr_teleop_shm | 验证置信: high

robot.connect() 独立尝试 arm 与 hand (interface.py:103-115)：arm 失败时 hand 可能已成功连接（EtherCAT/RS485 句柄已打开）。行 147-150 的失败分支只做 vr_receiver.stop() + return，不调用 robot.disconnect()，与紧随其后的 preflight 失败分支（行 154-158，先 robot.disconnect() 再 vr_receiver.stop()）不一致。后果是 XHand SDK 连接未释放直至进程退出，下次冷启动可能需要重试（XHandConfig.open_serial_retries 存在正是因为 RS485 冷启动敏感）。

### `examples/real/vr_teleop_shm.py:168` — 60s max_frames 自动停止发生在写线程内，controller.recording 不回写：REC 假显示，超过 60s 的数据静默丢失且 S 键拿不到路径
- 来源: vr_teleop_shm | 验证置信: high

入口设 max_frames=960（60s@16Hz）。达到上限时 CollectionLoop.record_frame 在 RecordingSession 写线程内自动 stop_episode(success=True, reason='max_frames') (collection_loop.py:114-116)，但 TeleopController.recording 标志无任何回写通道，控制器每 tick 继续入队帧（record_frame 因 not is_recording 直接丢弃），状态行持续显示 REC；操作者之后按 S 时 _recorder_session.stop 里 stop_episode 返回 None（已非 recording），日志只有 'Episode stopped' 无保存路径。演示超过 60s 的部分被静默截断且无明确提示。60s 上限本身是入口注释声明的刻意选择（旧值 3000@50Hz 同为 60s，非迁移回归），问题在于截断后的静默与状态失同步。

### `examples/real/vr_teleop_shm.py:202` — main() 缺 try/finally：等待循环 Ctrl-C 或 run() 逃逸异常跳过全部清理，SHM 段泄漏且下次运行 VR 门被陈旧帧假通过
- 来源: vr_teleop_shm | 验证置信: high | 原评级: major

第 9 步清理 (vr_receiver.stop() + robot.disconnect(), 行 227-229) 不在任何 try/finally 内。两条可达路径会跳过它：(1) 在最长 120s 的 VR 等待循环中按 Ctrl-C——KeyboardInterrupt 在 time.sleep(0.5) (行 215) 抛出，行 221 的 finally 只执行 kb.stop()，随后异常直接冲出 main()；(2) controller.run() 只捕获 KeyboardInterrupt/(RuntimeError, ConnectionError, ValueError) (controller.py:241-243)，_tick 内的 AttributeError/TypeError/OSError 等在 finally _shutdown() 后继续向上抛。后果已逐环验证：vr_receiver.stop() 是唯一调用 shm.unlink() 的地方 (vr_receiver_process.py:130-131)，跳过后 /dev/shm 的 dexmani_vr_frames 段残留；下次运行 VRReceiverProcess.__init__ 以 create=True 建 SHM 时命中 FileExistsError 会静默 attach 旧段 (frame_manager.py:63-72)，ring buffer 的 _write_seq 持久在 SHM 中非零，read_latest() 返回上一会话的最后一帧 (ring_buffer.py:154-171)，于是行 211-214 的等待门在 Quest 未连接时立即打印"收到首帧…就绪"假通过。同时 robot 未断开、ArmInnerLoop 守护线程被解释器硬杀（见下一条）。

### `examples/real/vr_teleop_shm.py:206` — 等待循环三个早退路径不停止已启动的 ArmInnerLoop；ESC"急停"在此阶段也不调用 emergency_stop
- 来源: vr_teleop_shm | 验证置信: high

TeleopController 在行 188 构造时 (dry_run=False) 即启动 ArmInnerLoop 守护线程 (controller.py:126-134)，该线程持有独立 XArmAPI 连接并执行 motion_enable(True) + set_mode(6) + set_collision_sensitivity(1) (inner_loop.py:259-263)。等待循环的 Q/ESC 分支 (行 206-210) 与 120s 超时分支 (行 216-220) 只做 vr_receiver.stop() + robot.disconnect() + return，从不调用 controller 的关停或 _arm_inner.stop()：守护线程在解释器退出时被硬杀，_run 的 finally 中 arm.disconnect() (inner_loop.py:441-445) 不会执行，SDK 连接被粗暴掐断，臂留在 mode 6 + 使能 + 低碰撞灵敏度状态。ESC 语义标为"急停"，但此路径既不 stop 内环也不调用 robot.emergency_stop()。由于此阶段无运动目标（内环仅 hold）且 Mode 6 固件断连保持位置，硬件危险有限，故降为 minor，但属急停/退出路径缺口。

### `examples/real/vr_teleop_shm.py:212` — VR 就绪门只判 frame is not None，不查帧龄也不查 vr_receiver.crashed
- 来源: vr_teleop_shm | 验证置信: high

行 211-214 收到任意非 None 帧即宣布就绪，不检查帧龄——配合上条泄漏的陈旧 SHM 段（或同机其他曾写过该段的进程），会用上一会话的旧帧假通过启动门；vr_receiver.read_latest_with_age() (vr_receiver_process.py:193-195) 现成可用却未使用。另外循环从不检查 vr_receiver.crashed：子进程若因 hand_tracking_sdk ImportError 立即崩溃 (vr_receiver_process.py:379-381 置 _crashed)，主进程仍会空转满 120s 才退出，而非快速失败并提示原因。进入遥操作后 controller 端的 _frame_age（基于 local_recv_ns 的 monotonic 差）能兜住陈旧帧，故危害限于启动阶段误导操作者。

### `examples/sim/keyboard_teleop_sim.py:85` — 键盘对工作空间 z 上限漂移 (真机 0.5 vs 仿真 0.55) 且边界策略不同（硬 clamp vs 方向感知软墙）
- 来源: x-realsim | 验证置信: high

真机 WORKSPACE_BOUNDS z=[0.05,0.5]，仿真 z=[0.05,0.55]：仿真顶部 5cm 在真机被 clamp 不可达。另外真机对累计 target 直接 np.clip 到边界，仿真是方向感知软墙（向外拒绝、向内允许并可解除告警）。顶部区域可达性与贴边操作手感在两侧不一致，边界附近行为验证不可迁移。

### `examples/sim/keyboard_teleop_sim.py:320` — 归位自碰撞检查作用在错误的路径段：检查的是规划器已验证过的段，新追加的 last-waypoint→home_qpos 段反而未检查
- 来源: kbd_sim | 验证置信: high

行 317-322 先对 plan_path 已带 check_self_collision=True 验证过的 dense 路径做自碰撞检查，通过后才追加 home_qpos 并重新插值——真正新增、未经任何验证的段（规划终点构型→精确 home_qpos，冗余关节可能有差异）恰恰不做检查就执行。同库参考实现 vr_teleop_sim.py:330ff 的 execute_return_home 逻辑相反且正确：先 vstack home 再插值，对含追加段的 dense_joint 整体检查后才采用。另外若 dense 路径检出自碰撞，本文件仅跳过追加 home，带碰撞的路径仍照常执行。仿真中 SAPIEN 模型本身 disable_self_collision=True（xarm7_xhand.py:21），物理上不会卡死，故降为 minor。

### `examples/sim/keyboard_teleop_sim.py:441` — max_consecutive_errors=10 定义后从未使用——连续 IK 失败永远不会触发停机，疑似安全上限被遗漏
- 来源: kbd_sim | 验证置信: high

行 441 定义 `max_consecutive_errors = 10`，全文件无任何引用；ik_fail_consecutive 只用于节流打印（行 594）和状态行显示（行 526-527），IK 连续失败任意多次循环也只是每次把目标吸回实际位姿并保持（行 596-599）。保持本身符合 fail-safe 约定，但这个死变量表明原意是 '连续 N 次 IK 失败后中止'，该安全上限实际未实现。仿真中后果仅为可无限卡在 held 状态。

### `examples/sim/keyboard_teleop_sim.py:463` — ESC 分支打印 'emergency_stop' 但从未调用 sim.emergency_stop()，与文件头声明的 'ESC 紧急停止 + 退出' 不符
- 来源: kbd_sim | 验证置信: high

行 463-466 仅 print 后 break，SimRobotInterface.emergency_stop()（sim_adapter.py:208-209，软停：以当前位置为目标保持）存在但未被调用；ESC 与 Q 的实际行为完全相同。仿真中进程随即退出、后果为零，但该文件明确自称 keyboard_teleop_real.py 的设计参考（行 30-31），'急停路径打印却不停' 的模式与 CLAUDE.md 安全架构第 7 条（emergency_stop → arm.stop + hand.stop）相悖，照搬到真机会成为急停缺口。

### `examples/sim/keyboard_teleop_sim.py:475` — R 键归位忽略 do_return_home() 失败返回值，规划失败后仍把命令快照到 home，导致下一 tick 无规划的大幅关节跳变
- 来源: kbd_sim | 验证置信: high | 原评级: major

do_return_home() 在 plan_path 失败时打印 PLAN FAILED 并 return False（机械臂原地不动，行 311-314），但调用处（行 475-479）丢弃返回值，无条件执行 target_pos/target_quat/prev_arm_cmd = home 快照。松开 R 后进入 idle 分支（行 601-605）arm_cmd = prev_arm_cmd = ARM_HOME_QPOS，行 610 直接 apply_action(home)——SAPIEN PD 会以一整个构型差的阶跃目标猛拉机械臂，绕过全部路径规划/碰撞检查/2° 插值。idle 分支不经过 _joint_safety_clamp，且关节跟踪保护阈值 5.0 rad（行 106）与 360°/s 兜底都拦不住这种跳变；EEF 发散保护又因 2s 采样延迟（见另一发现）最多 2 秒后才反应。仿真中后果是失控扫掠（可能穿过 table actor）；该文件自称是 keyboard_teleop_real.py 的设计参考，此模式若被照搬到真机属危险模式。修复：if not do_return_home(...): 保持原 target/prev_arm_cmd 不变。

### `examples/sim/keyboard_teleop_sim.py:628` — EEF 跟踪发散安全检查在 50Hz tick 上消费 0.5Hz 刷新的过期误差，'连续 5 次' 去抖语义完全失效
- 来源: kbd_sim | 验证置信: high | 原评级: major

eef_pos_err/eef_rot_err 只在 `now - last_status_time > 2.0` 的周期状态块内重算（行 503、514-517），但行 628-640 的临界检查每 tick（50Hz）执行并递增 consecutive_eef_divergence，MAX_EEF_DIVERGENCE_CONSEC=5（行 103）在 0.1s 内即被打满。双向后果：(a) 单次瞬时超限采样即使已恢复也必然在 5 tick 后停机（stale 值 2 秒内不变，'恢复' 永远观察不到）；(b) 两次采样之间发生的真实发散最长 2 秒才被检测，与注释宣称的 'EEF 跟踪误差监控（安全兜底）' 不符。修复：把 compute_pose_error 移到每 tick（成本可忽略，pose_utils.py:104-108 仅 dot+arccos），或把计数器改为按采样次数计。

### `examples/sim/keyboard_teleop_sim.py:628` — 仿真键盘版 EEF 跟踪安全检查使用最长 2 秒前的陈旧误差值
- 来源: x-realsim | 验证置信: high

eef_pos_err/eef_rot_err 只在每 2 秒一次的状态打印块里更新 (513-517 行)，但 critical 安全判定 (>8cm/30°) 每 tick 都在 628-640 行执行，MAX_EEF_DIVERGENCE_CONSEC=5 tick 仅 0.1s。后果：真实 divergence 发生后最长 2s 才被采样到；而状态打印瞬间采到的一次瞬态大误差会在接下来 5 个 tick 内直接停机（单次采样被重复计数 5 次）。安全监控的时序语义是坏的——这是 sim 独有逻辑的 bug，真机版无此机制。

### `examples/sim/test_motion_planning_sim.py:10` — 模块 docstring/常量描述的'workspace 随机采样 + return_home + IK 测试'已不存在：约 1100 行死代码，main() 只跑 Pick-and-Place
- 来源: mp_sim | 验证置信: high

文件头 docstring（1-14 行）描述随机采样 plan_path 测试、return_home、solve_ik 成功率三个流程，93-95 行还保留 '--comprehensive 已废弃保留 flag 向后兼容' 注释；但 main()（1805-1811 行）的 argparse 只有 --headless/--seed/--episodes，description 是 'Pick-and-Place 抓取放置仿真测试'。build_target_pose、IKStats、smooth_drive_to_target、animated_reset_to_home(带 planner 分支)、return_to_home_sim 及其全部 helper、hand_randomize、plan_safe_descent、plan_and_execute、ik_test/_run_ik_loop/print_ik_stats、sample_z_biased、place_marker、sweep_z_min/_run_desk_test/_spawn_table_objects、check_fingertips、append_joint_goal、STRATIFIED_REGIONS/Z_*/NUM_SAMPLES 等常量全部无 main() 可达调用；第 44 行导入的 execute_dense_path 也只被死代码使用。后果：读者/CI 会以为桌面安全回归（分层低 Z 采样、IK 往返验证）仍在运行，实际早已不测。另 874-882 行残留一段 'But wait: ... Let me re-check' 的推理草稿注释。按 CLAUDE.md 准则提出但不代删。

### `examples/sim/test_motion_planning_sim.py:1043` — plan_and_execute 的 safe-descent 分支中 tips_before 在运动执行之后才采集，tip_drift 检查恒为 0、形同虚设
- 来源: mp_sim | 验证置信: high

正常分支在执行前采 tips_before（1036 行）、执行后采 tips_after（1065 行），用 EEF 局部指尖偏移变化检验手在臂运动中未相对 EEF 漂移。但 used_safe_descent 分支里 plan_safe_descent（1014 行）已经把路径执行完，1041-1043 行才第一次采 tips_before，与 1065 行 tips_after 之间只隔一次纯 FCL 查询（check_path_desk_safety_sim 不推进物理），两个快照几乎相同，tip_drift_mm≈0 永不触发 2mm 阈值——该分支的漂移验证失效。此函数当前无 main() 调用（死代码），故 minor。

### `examples/sim/test_motion_planning_sim.py:1329` — sweep_z_min/_run_desk_test 已损坏：_env_cm 未初始化必触发 AssertionError，且被检查的 CollisionModel 不含桌面障碍、不随扫描 margin 变化
- 来源: mp_sim | 验证置信: high

_run_desk_test 自建带 margin override 的 planner（1251-1279 行），但从不设置模块全局 _env_cm（只有 main() 在 1853-1854 行设置，而 main() 的 argparse 没有 sweep 入口）。首个成功执行的样本走到 1329 行 check_hand_desk_clearance_sim → _get_cm() → 137-139 行 assert 直接崩溃。即便修好 assert，还有两层问题：(1) _run_desk_test 从不对其 planner.collision_model 调 add_table，而 check_segment_env_collision_free/check_env_collision 在无障碍物时直接返回安全（collision_model.py:447-448, 471-472），1272 行注释宣称的 'enable FK desk safety' 只有 FingertipDeskSafety 半边生效；(2) 扫描的 hand_safe_margin 只影响 plan_path 内的 FK 阈值，统计口径里的 'desk_collisions'（FCL 检查）与 margin 完全无关，'网格搜索最优 margin' 的结论不成立。此路径当前 CLI 不可达（死代码），故降为 minor。

### `examples/sim/test_motion_planning_sim.py:1795` — episode 成功判据与阶段统计不一致：return 规划失败→整集判 FAILED 且无失败原因；return 执行中止→仍判 SUCCESS；中止的阶段计入完成率
- 来源: mp_sim | 验证置信: high

success 判据是 `len(result.phases_completed) >= 5`（4 个抓放阶段 + return）。三处不一致：(a) return 阶段在检查 completed 之前就 append（1785 行在 1786 行 `if not completed` 之前），所以 return 因碰撞 HOLD 超限 ABORT 的 episode 仍被判 SUCCESS；(b) 若 plan_path(safe_return_pose) 返回 success=False（未抛异常），既不 append 也不做 animated_reset，episode 落到 1795 行被判 FAILED，但 failure_reason 为空串——打印为 '❌ FAILED: '，汇总里归为 unknown，尽管抓放全部成功；(c) 主阶段循环同样先 append 再查 completed（1705 行在 1707 行之前），因此 1946-1950 行的'阶段完成率'表把执行到一半被 ABORT 的阶段也计为完成。该脚本的输出就是这些统计，误分类直接扭曲测试结论。

### `examples/sim/test_motion_planning_sim.py:1859` — add_table 注册桌面障碍时丢弃了基座系 Y 偏移(−0.20m)与 −30° 偏航，工作区远角存在未建模的真实桌面
- 来源: mp_sim | 验证置信: high

仿真机器人根位姿默认绕 Z 旋转 30°（xarm7_xhand.py:22 `root_pose=sapien.Pose(p=[0,0,0], q=euler.euler2quat(0,0,np.pi/6))`，sim_adapter.py:75 构造时未覆盖）。main() 把桌面世界坐标 (0.4,0,0) 变换到基座系得到 table_in_urdf.p≈(0.346,−0.200,0)，但 cm.add_table 只接收 table_height 和 x_center——CollisionModel.add_table 签名没有 y 参数，内部硬编码 position=(x_center, 0.0, h−half_z) 且姿态为单位旋转（collision_model.py:637-655）。于是 FCL 桌面盒在基座系中比真实桌面偏 +0.2m（Y）且未旋转 30°。当前 TABLE_HALF=(0.5,1.0) 远大于采样区，Z 顶面高度也恰好不受 Z 轴旋转影响，所以立方块采样区 (x≤0.70,y≤0.30) 内仍被覆盖；但在规划器允许的 workspace 角落（world (0.75,0.5) → base_x≈0.90 > 建模盒 x_max≈0.846）真实桌面无 FCL 覆盖，RRT 中间路点若在该 5cm 条带内下探到 z<0 会被 env 检测漏判为安全。正确做法是直接用 cm.add_box_obstacle 传入 table_in_urdf 的完整位姿（rotation=Rz(−30°), y=−0.2）。一旦 TABLE_HALF、采样范围或根偏航改变，覆盖会静默失效。

### `examples/sim/vr_teleop_sim.py:105` — 16Hz 迁移未同步到仿真 VR 入口：sim 仍 50Hz，且注释错误声称与真机控制器一致
- 来源: x-realsim | 验证置信: high | 原评级: major

真机生产入口已迁移到 16Hz 决策/录制 (schema v7)，并据此派生了 recorder 网格 (control_hz=16, max_frames=960≈60s, min_frames=16, skip_initial_frames=4)、nullspace 步长 (1°×50/16) 和滤波参数；仿真 VR 入口仍是 CTRL_HZ=50.0，注释还写着“与 TeleopControllerConfig.target_hz 一致”（该默认 50 已不是生产值）。sim 的 EpisodeRecorder(data_dir=...) 全用库默认：50Hz 时间网格、max_frames=10000(≈200s)、min_frames=50。后果：在 sim 里验证的每 tick 步长、IK 求解节奏、episode 时长上限、录制 fps/meta(control_hz=50) 全部与真机 16Hz 数据不可比，混入数据集会得到两种时间网格。

### `examples/sim/vr_teleop_sim.py:110` — VR 帧过期阈值注释与真机实际值相反：注释称真机 0.1s，真机实际 0.5s，仿真反而更严格
- 来源: x-realsim | 验证置信: high

sim 的 VR_FRAME_MAX_AGE_S=0.2 注释写“仿真容忍度高于真机（0.1s）”；真机 TeleopController 的单帧过期阈值 _VR_STALE_THRESHOLD_S=0.5s（另有 3s 累计断连超时）。实际关系与注释相反：sim(0.2s) 比真机(0.5s) 严格 2.5 倍。后果：在 sim 里评估 VR 丢帧/网络抖动的鲁棒性（多久触发软减速/hold）得到的结论对真机不成立，且注释会引导开发者按 0.1s 的错误认知调参。

### `examples/sim/vr_teleop_sim.py:299` — cam.get_entity().get_pose() 返回挂载实体位姿——不包含相机本地位姿偏移——导致相机位姿与世界↔相机变换错误
- 来源: vr_sim | 验证置信: high

在 capture_camera_frame 第 299 行，cam_entity = cam.get_entity() 返回挂载实体（custom_eef_link），而非相机本身。cam_entity.get_pose()（第 302 行）返回的是末端执行器连杆的世界位姿，不包含 setup_ee_camera 中应用的本地位姿偏移量（CAMERA_EEF_OFFSET_POS=[0.05, 0.0, -0.03]，CAMERA_EEF_OFFSET_QUAT_WXYZ 约为 30 度俯仰）。这对第 1 个发现中的错误深度计算属于次要因素，但如果相机位姿被日用于世界↔相机坐标变换，则是一个独立的 bug。SAPIEN 的 add_mounted_camera 在 mount entity 上挂载一个 RenderCameraComponent，get_entity() 返回 mount，而非相机。

### `examples/sim/vr_teleop_sim.py:315` — camera_frame 缺少 "frame_number" 导致 /flag_camera_fresh 永久为 False
- 来源: vr_sim | 验证置信: high | 原评级: major

capture_camera_frame（第 315-318 行）返回一个包含 "rgb"、"depth"、"timestamp" 键的字典，但没有 "frame_number" 键。EpisodeRecorder.add_frame()（episode_recorder.py:284 行）中 camera_frame.get("frame_number") 返回 None，因此 fresh_token 保持为 None，第 291-297 行的新鲜度检查被跳过，/flag_camera_fresh 永久为 False，即使每 tick 都捕获了新的相机帧。这无法区分相机真实冻结与正常帧——下游质量过滤会错误标记所有相机帧。

### `examples/sim/vr_teleop_sim.py:793` — VR 帧过期时跳过 record_frame 导致 TimestampAlignedBuffer 前向填充大量重复数据
- 来源: vr_sim | 验证置信: high | 原评级: major

仿真脚本在 VR 帧过期（stale）期间完全跳过 record_frame 调用（第 793-816 行仅推进物理步进，不记录任何帧）。当 VR 恢复正常后，下一帧的时间戳（sim_state["timestamp"] 来自 time.time()）已经跳过了失联时长。TimestampAlignedBuffer.add() 在 episode_recorder.py:331 行将时间戳差量解释为需要向前填充的网格槽位（timestamp_buffer.py:63-67 行），从而用恢复帧的数据复制填充所有缺失槽位。例如，一次 5 秒的 VR 掉线会产生约 250 帧完全相同的重复数据（50Hz 下），且 flag_ik_ok/flag_retarget_ok 均被标记为 False。真实 controller 在 VR 掉线期间会继续录制带 held=True 的帧，因此没有这个间隙。

### `examples/sim/vr_teleop_sim.py:821` — stale_frame_count 语义混淆——只计数连续掉帧，不累计——vr_drop 显示不反映真实总掉帧率
- 来源: vr_sim | 验证置信: high

stale_frame_count 在每次 VR 帧过期时递增（第 795 行），但在有效帧到达时重置为 0（第 821 行）。在 episode 结束时，vr_drop = stale_frame_count / max(episode_tick_count, 1)（第 615 行）计算的是最近一次连续掉帧占比，而非累计掉帧占比。例如，VR 掉线 10 帧后恢复，再掉线 50 帧后恢复，最后再掉线 10 帧——episode 结束时的 stale_frame_count 为 10，而非 70。虽然 classification（第 621-625 行）使用 ik_rate 而非 vr_drop，但状态打印中的 vr_drop 显示具有误导性。

### `examples/sim/vr_teleop_sim.py:852` — 限幅拓扑分叉：仿真对臂加 180°/s 软件限速、手无 per-send 限幅；真机手有 0.098rad/send 硬限幅、臂无软件限速
- 来源: x-realsim | 验证置信: high | 原评级: major

sim 在 IK 结果上应用 velocity_limited_step(180°/s bottleneck scaling) 且录制的是限幅后的 arm_cmd；真机臂路径无任何软件限速（Mode 6 直发，仅 planning/ik.py 的 90° IK 异常跳变门→held）。手方向相反：真机 XHandConfig(max_delta_rad=deg2rad(90)/16≈0.098) 在驱动层逐次硬限幅，sim 的 hand_cmd 从 retargeter 直接 apply_action，完全无限幅。后果：(1) sim 中被等比缩放的快速臂动作，在真机上要么直发要么整帧 held，失败模式不同；(2) sim 中手部瞬时大角度跳变可正常执行，真机被限到 90°/s，动作滞后/变形，录到的 action 与执行分离——sim 验证的快速抓取节奏对真机不成立。


## INFO

### `examples/real/calibrate_camera.py:688` — 使用 RealSense 畸变系数时未检查 intr.model，若设备上报 inverse_brown_conrady 会引入系统性位姿偏差
- 来源: calib_cam | 验证置信: low

`DISTORTION = np.array(intr.coeffs)` 直接把 RealSense 内参的畸变系数喂给 cv2.solvePnP，而 OpenCV 期望的是正向 Brown-Conrady (k1,k2,p1,p2,k3)。RealSense 彩色流的畸变模型因机型而异（如 D4xx 彩色流常上报 inverse_brown_conrady），若模型为逆向且系数非零，按正向使用会让所有 marker 位姿带同向系统偏差，最终整体偏移标定出的外参而一致性指标看不出来（各样本被一致地偏）。建议加一行断言：intr.model 属于 {none, brown_conrady}（或系数全零），否则拒绝/警告。当前接的 L515 若上报 brown_conrady 则无害，属防御性检查缺失。

### `examples/real/calibrate_camera.py:731` — 架构偏离（标定脚本有意例外）：绕过 RobotInterface 直调 XArmAPI.get_position，且发送 IK 目标前不调 validate_action
- 来源: calib_cam | 验证置信: low

(1) _get_ee_pose 调 `robot.arm.arm.get_position(is_radian=True)`，越过 RobotInterface 甚至越过 XArm7.get_position() 包装（xarm7.py:306-312 返回同样的 6 维 mm/rad 向量，本可复用）。理由在文件内有明确文档（727-730 行：手眼需要基座系 TCP 位姿、is_radian 与 degrees=False 的耦合警告），单位处理正确（[:3]/1000 mm→m，RPY 已是 rad，与 SDK xarm7.py:309 的 is_radian=True 一致），坐标系链路正确（基座系解 T_base_camera 后经 base_pose_world 转 world，且 base_pose_world 值与生产入口 vr_teleop_shm.py:112-114 一致）。(2) 994 行 arm_inner.set_target(ik_result.qpos) 前未调 validate_action（力矩/温度门被跳过），但这与现有键盘示例模式一致——keyboard_teleop_real.py:594 同样直发，validate_action 目前仅在 teleop/core/controller.py:346 接线；内环自身有 0.3rad 步进钳位、错误锁存与固件限速兜底。两项均属标定脚本的有意例外，降级为 info。

### `examples/real/calibrate_l515_depth.py:62` — 生产 gate 阈值内联复制而非 import L515_CALIBRATED_DEPTH_VALIDITY，存在漂移风险（当前已逐字段核对一致）
- 来源: calib_l515 | 验证置信: low

PROD_GATE（62 行）与 EDGE_CFG（64 行）手工内联生产值，realsense.py 的 L515_CALIBRATED_DEPTH_VALIDITY 注释自称 "Single source of truth ...（the diagnostic script inlines the same values）"。当前一致性已核对：confidence_min=2<<4=32 恰等于驱动运行时对 confidence_min=2 的 <<4 预移位副本；ir_min=2、ir_saturation=250、saturation_dilate_px=3、sigma_poly=(-0.00094,0.00293)、n_sigma=5.0、t_min=0.010、t_max=None、dilate_px=0 全部逐字段相等。但未来生产值更新而脚本未同步时，Phase 2 的 gate 统计与 "current" 对比列会静默失真。可改为 `replace(L515_CALIBRATED_DEPTH_VALIDITY, confidence_min=...<<4)` 派生。

### `examples/real/calibrate_l515_depth.py:194` — sigma_poly 拟合数学、单位与保存-消费链核对通过（任务指定重点，无缺陷）
- 来源: calib_l515 | 验证置信: low

逐项核对结论：(1) 单位——z/s 均为 raw*depth_scale（米），拟合与 c0/c1 全程米制；JSON 的 bins 表另存 sigma_mm=sm*1e3 且键名带 _mm，无混用。(2) polyfit 解包——deg=1 返回 [最高次, 常数]，`c1, c0 = np.polyfit(...)` 顺序正确；w=sqrt(counts) 乘残差、平方后即按样本数加权，正确。(3) 系数排列——打印/消费均为低次在前 (c0, c1)，与 DepthEdgeConfig 文档 "sigma_poly[0] + sigma_poly[1]*z ...（meters, low order first）"一致；build_edge_threshold_lut 的 Horner 用 reversed(sigma_poly) 从高次起算，求值正确。(4) 消费端一致性——realsense.py L515_CALIBRATED_DEPTH_VALIDITY 的 (-0.00094, 0.00293)（米）与其注释 "sigma_z = -0.94 + 2.93*z mm" 换算吻合，与本脚本 EDGE_CFG 及 2026-07-15 标定记录一致。(5) 方差数值——E[x²]-E[x]² 在 float64/raw≤6000/120 帧下取消误差 ~1e-9 raw²，远小于典型方差，安全。

### `examples/real/calibrate_l515_depth.py:283` — 绕过 RealSense.read() 直接访问 pipeline/first_depth_sensor —— 标定脚本的有意例外，逻辑自洽
- 来源: calib_l515 | 验证置信: low

脚本直接调 camera.pipeline.wait_for_frames()（98-102/114 行）与 profile.get_device().first_depth_sensor().set_option/get_option（283-305 行），绕过驱动的 read() 封装。核对确认这是有意且必要的：read() 会执行 _apply_depth_validity 就地置零无效像素（realsense.py:608 + 546-582 的 in-place np.multiply），而标定需要未经 gate 的原始深度并在脚本内用生产阈值重算统计；docstring（18-23 行）与 GATE_NOOP 注释（55-58 行）明确说明该设计，GATE_NOOP 的 0/None 阈值同时保证 create_rs_config 仍启用 confidence+IR 流（realsense.py:486-493，0 is not None 判定成立）。CLAUDE.md 的"仅经 RobotInterface"规则只约束 XArm7/XHand，不涉及相机。无问题，按任务要求记为 info 备案。

### `examples/real/calibrate_l515_depth.py:365` — 漂移阶段重连后继续使用 main 里缓存的旧 depth_scale 与 edge LUT
- 来源: calib_l515 | 验证置信: low

depth_scale 与 lut 在 main（364-365 行）连接后取一次，此后所有 capture_burst/拟合沿用；run_drift_phase 内可能发生 disconnect+connect 重连（224-228 行），重连后驱动会重读 depth_scale 并重建自身 LUT，但脚本的旧副本不更新。L515 的 depth_units 是设备常量（0.00025，属 JSON-only 参数、保持硬件默认，重连不变），因此当前实际无害——但若换到 depth_units 可配置的设备或未来固件行为变化，raw→米换算与 ROI 阈值会静默错位。

### `examples/real/keyboard_teleop_real.py:421` — get_state 异常分支连续两个 break，第二个为不可达死代码
- 来源: x-safety | 验证置信: low

连续错误超限触发急停后的 `break` 之后紧跟又一个 `break`，永不可达——疑似编辑残留。无运行时影响，但出现在急停分支这种关键路径上，说明该段代码未被审视过，建议顺手清理。

### `examples/real/keyboard_teleop_real.py:421` — 真机键盘版 get_state 异常分支存在重复 break（死代码）
- 来源: x-realsim | 验证置信: low

except 分支中连续两个 break，第二个永不可达。无功能影响（第一个 break 已正确退出主循环并在此前完成急停），但属于真机侧独有逻辑中的明显笔误，提示该分支可能改动时未清理。

### `examples/real/keyboard_teleop_real.py:593` — 注释错误：声称 250Hz 位置伺服，实际为 50Hz
- 来源: kbd_real | 验证置信: low

line 593 注释写 'Arm: via inner loop (250Hz position servo)'，但 ArmInnerLoop 配置为 loop_period=0.02 即 50Hz (inner_loop.py:73)，内部 RateManager 以 50Hz 运行。250Hz 是旧 velocity 模式的参数，当前 Mode 6 已是 50Hz。

### `examples/real/replay_traj.py:62` — 16Hz 迁移后的 50Hz 陈旧痕迹：模块 docstring、死常量 CTRL_DT、--speed help 文案未同步
- 来源: replay_traj | 验证置信: low

本次 diff 已正确让 replay_hz 跟随 episode 的 control_hz（16Hz episode 按 16Hz 回放），但三处 50Hz 痕迹未同步：(1) 模块 docstring 'Reads an HDF5 episode (schema v5, 50Hz aligned grid)'（line 4）——当前 schema 是 v7/16Hz；(2) `CTRL_DT = 0.02  # 50Hz`（line 62）在全文件无任何使用（回放节拍由 RateManager(replay_hz) 决定），是死常量；(3) --speed 的 help 'Replay speed factor (1.0=50Hz real-time...)'（line 870）在 replay_hz=fps*speed 语义下已不准确，1.0 现在表示按录制网格实时。均无功能影响，但会误导操作者对回放频率的预期。

### `examples/real/replay_traj.py:62` — CTRL_DT=0.02 "# 50Hz" 是死常量——replay 实际节拍已由 meta control_hz 驱动
- 来源: x-16hz | 验证置信: low

CTRL_DT 在全文件（1026 行）仅此一处出现，无任何使用点；迁移后重放速率正确地从 /meta control_hz 取（16Hz episode 按 16Hz 重放，带 1-100Hz 合理性钳制回退 50）。残留的 "50Hz" 注释会误导维护者以为 replay 固定 50Hz。附带外观项：:786/:833 每 50 帧打印进度，16Hz 重放下变为 ~3.1s 一条而非 1s（无功能影响）。

### `examples/real/replay_traj.py:135` — 旧 schema（无 control_hz）fps 回退使用实现帧率而非标称帧率
- 来源: x-recording | 验证置信: low

load_trajectory 优先 control_hz (v7)，回退到 fps。但 v7 的 fps 在 stop_episode 时被覆盖为 frame_count/duration (episode_recorder.py:693)，暂停过的 episode 因 wall-clock 持续导致 fps 稀释。line 138 钳位 1-100 无法捕捉稀释但 plausible 的值（如 30Hz），replay 将以此错误速率回放。对 v7 无影响（control_hz 存在），仅影响 v6 及以前文件但 fps 被 stop 阶段覆写的情况。

### `examples/real/test_motion_planning_real.py:666` — 休眠代码：simulate_path_in_sapien 把任意 contact 计为碰撞失败，启用后大概率全路径误报
- 来源: mp_real | 验证置信: low

SIMULATE_PATHS 默认 False，此路径休眠；一旦置 True：行 665-668 `contacts = scene.get_contacts(); if len(contacts) > 0 → errors` 把 PhysX 报告的任意接触对当作碰撞。PhysX 在间距小于 contact_offset（constructor.py:38 set_shape_config(contact_offset=...) 全局设置）时即生成接触对，未必有穿透；手指链节在全零手姿下彼此邻近，且 set_qpos 后仅 balance_passive_force 一次 + 3 步物理积分（行 657-662），关节无位置驱动、重力下会微沉——每个 waypoint 都可能报出 "N contacts detected"，使该验证在启用时不可用（只会把 ok 置 False 打 SIM WARNING，不阻断执行，故无害）。若要启用，应过滤 separation<0 的穿透接触或换用运动学摆位 + FCL 查询。

### `examples/real/test_motion_planning_real.py:865` — 有意例外：绕过 RobotInterface 直连 XArm7；另 set_hand_qpos 从未调用，自碰撞按张开手（全零）近似且每次 plan_path 刷 warning
- 来源: mp_real | 验证置信: low

本脚本直接 `arm = XArm7(arm_config)` 并自行 send_action/reset，绕过 RobotInterface.validate_action——对纯臂运动规划测试属有意例外（XArm7 本就是 RobotInterface 内部使用的阻塞移动封装，且不涉及手指令），按审查规则降级为 info。附带影响：planner 以 hand_dof=True 构造（planner.py:107 默认），但全程未调 set_hand_qpos，CollisionModel 用全零（张开手）姿态展开 19-DOF 自碰撞检查——若真机 XHand 断电停在非张开姿态，自碰撞检查与实际几何不符可能漏检；且 plan_path 每次调用都会触发 "_hand_qpos was never set" logger.warning（planner.py:188-193），Test 3 的 num_samples 次 + Test 4 每个 waypoint + 两次归位都会刷屏。测试开始前把真机手摆到张开位、或显式调一次 planner.set_hand_qpos(np.zeros(12)) 消除歧义。

### `examples/real/test_pointcloud_process.py:137` — DepthValidityConfig 逐字段内联复制生产常量 L515_CALIBRATED_DEPTH_VALIDITY，再标定需改两处，存在漂移风险
- 来源: pointcloud | 验证置信: low

测试在行 137-154 手写 DepthValidityConfig(confidence_min=2, ir_min=2, ir_saturation=250, saturation_dilate_px=3, edge=DepthEdgeConfig(sigma_poly=(-0.00094,0.00293), n_sigma=5.0, t_min=0.010, t_max=None, dilate_px=0))——本次已逐字段核对与 realsense.py 的 L515_CALIBRATED_DEPTH_VALIDITY 完全一致，当前无错。但该常量自述为 'Single source of truth for the SHM / recording path (the diagnostic script inlines the same values)'：下次 sigma_poly 重标定（memory 记录冷机漂移复测待做）时必须同步改两处，否则诊断脚本验证的 gate 与生产录制的 gate 静默分叉。建议直接 import 常量。

### `examples/real/test_pointcloud_process.py:406` — FPS 后端与生产默认不一致：测试用 pytorch3d+CUDA，生产默认 o3d CPU（有意差异，但计时数字不可外推）
- 来源: pointcloud | 验证置信: low

测试固定走 pytorch3d.sample_farthest_points（CUDA 可用时上 GPU，行 399-408），而生产 PointCloudProcessorConfig 默认 fps_backend='o3d'（open3d CPU farthest_point_down_sample，~6.6ms，且 CameraProcess 使用默认配置），pytorch3d 仅为可选后端且注明 fork 子进程内有 CUDA 前置条件。属诊断脚本的有意选择（生产配置注释也记录了两条路径），但测试打印的 FPS 耗时（~9ms GPU）不代表生产路径实际耗时，报告数字对生产外推时需注意。

### `examples/real/test_quest_hand_teleop.py:279` — 绕过 RobotInterface 直接驱动 XHand、无 validate_action——文件头已声明 DEPRECATED 的有意独立测试例外
- 来源: quest_hand | 验证置信: low

全文直接实例化 XHand 并调用 send_action/get_state/reset_connection，未经 RobotInterface 与 validate_action() 前置门（无扭矩/温度门，仅依赖 XHand 内建关节限位裁剪与 E3 delta clip）。文件头 L1-5 明确声明 DEPRECATED 并指向 vr_teleop_sim.py 与 TeleopPipeline.compute_hand_command() 替代，属独立手部硬件调试脚本的有意例外，按审查约定降级为 info。附带说明：L310 直接改写驱动内部状态 `xhand.last_qpos_cmd = qpos_now.copy()` 以避免 E3 delta clip 卷绕，语义正确但属越过驱动封装的调试写法；L45 导入的 JOINT_NAMES 全文未使用。

### `examples/real/test_realsense.py:5` — 用法 docstring 写 `conda activate real`，与项目约定的 conda 环境名 `real_robot` 不符
- 来源: test_rs | 验证置信: low

CLAUDE.md 约定表明确写明 conda env 为 `real_robot`。docstring 第 5 行的 `conda activate real` 会把新用户引到错误/不存在的环境（缺 pyrealsense2/open3d/pytorch3d 依赖）。纯文档错误。

### `examples/real/test_realsense.py:157` — 重试提示声称递增退避（1s/2s/3s）但实际固定 sleep 1s；main 中最后一次失败仍打印"重试"
- 来源: test_rs | 验证置信: low

list_available_cameras 打印 `{1.0*(attempt+1):.0f}s 后重试` 但紧接 `time.sleep(1.0)`——第二次失败时提示 "2s 后重试" 实际只等 1s。main（行 596-601）在 attempt=2（最后一次）失败后仍打印"连接失败，3s 后重试..."并 sleep 1s，然后直接退出，既无重试也与提示秒数不符。仅日志误导，无功能影响。

### `examples/real/test_realsense.py:456` — Ctrl-C/未捕获异常中断主循环时跳过 pcd_viewer.close() 与 destroyAllWindows()（相机句柄本身已由 finally 覆盖）
- 来源: test_rs | 验证置信: low

行 456-457 的 `pcd_viewer.close(); cv2.destroyAllWindows()` 只在循环正常 break 后执行，不在 try/finally 内；循环中仅捕获 read 的 RuntimeError（行 296）和点云的 ValueError（行 314），KeyboardInterrupt 或其他异常会跳过窗口清理。main 的 finally（行 611-613）保证了 camera.disconnect()（关键硬件句柄无泄漏），残留的 open3d/cv2 窗口随进程退出销毁，实际无害——测试脚本可接受，故仅记 info。

### `examples/real/test_realsense.py:537` — 变体对比中首个 fps 变体的耗时包含 pytorch3d 一次性 import，污染对比数据
- 来源: test_rs | 验证置信: low

run_pcd_variants 用 perf_counter 逐变体计时（行 537-543），但库的 sample_points 在 sampling='fps' 时才在函数内 `import pytorch3d.ops`（pointcloud_utils.py:478），首次导入可能耗时数百毫秒到数秒，全部计入第一个 fps 变体（"voxel 5mm + fps 1024"）的打印耗时，使该行数字与后续 fps 变体不可比。同理，实时循环中首次按 's' 切到 fps 也会出现一帧卡顿。若在计时前先做一次 warm-up fps 调用即可消除。

### `examples/real/vr_teleop_arm_only.py:141` — 该入口整体保留 50Hz 旧参数（自洽），16Hz 迁移落在 record_plus 姊妹脚本；robot.arm.* 直访属刻意例外
- 来源: arm_only | 验证置信: low

CTRL_DT=0.02（50Hz）、EMA 0.6/0.3、EpisodeRecorder 未传 control_hz（默认 50）——三者互相一致，不构成 16Hz 迁移遗留 bug，但与生产 16Hz 路径（record_plus: CTRL_HZ=16 + τ-invariant EMA 换算 + control_hz=CTRL_HZ）并存，建议确认是否有意保留为 legacy 入口并在 docstring 标注，避免误用旧脚本采集出 50Hz 数据混入 16Hz 数据集。另：L268/419 robot.arm.clear_error()、L562-564 robot.arm.is_error()/robot.arm.arm.error_code/last_sdk_error_code 绕过 RobotInterface 包装直访臂驱动——臂-only 脚本刻意避开 RobotInterface.clear_error()（会连带清 hand），按例外降级为 info。

### `examples/real/vr_teleop_arm_only.py:686` — TrajectoryLogger 无界内存增长，且实际行为与类 docstring 相悖
- 来源: arm_only | 验证置信: low

traj_logger.append 在 50Hz 遥操作全程逐帧追加（无上限、无分段落盘），长会话内存持续增长（每条 ~1KB，量级 ~180MB/小时），与 recorder 的 max_frames=4500 上限不对称。另外类 docstring 声称 'Frame data is recorded regardless of teleop/recording state'，实际 append 位于 teleop_active 且 vr 不 stale 且 map 成功之后（L639/648、L652-654 的 continue 都在其前），非遥操作时段完全不记录——文档承诺的『全程可分析』不成立。debug 工具无碍安全，报 info。

### `examples/real/vr_teleop_arm_only_record_plus.py:413` — 长时间阻塞操作后未调用 limiter.reset()，触发虚假超预算日志
- 来源: record_plus | 验证置信: low

RateManager（rate_manager.py:86-93）提供 reset() 方法专用于"长时间阻塞操作后重置期限以消除虚假超预算警告"。但代码中的长时间阻塞操作——do_return_home（~1-5s，L540）、退出时的 30 秒保存/丢弃提示（L506-529）、post-loop 的 H/Q busy-wait（L898-905）——均未调用 limiter.reset()。重新进入主循环后，limiter.wait() 看到旧的 deadline 已经过去，执行 overdue 分支（rate_manager.py:69-84）并发送一条警告日志。这不会导致漂移或 burst——deadline 会 re-anchor 到 now——但会产生虚假的 "Control loop over budget" 警告。

### `examples/real/vr_teleop_shm.py:91` — hand_side="right" 与 VRReceiverConfig 注释矛盾：HeadFrame 可能被 SDK 过滤，head 位姿恒为单位值
- 来源: vr_teleop_shm | 验证置信: low

VRReceiverConfig 默认 hand_side='both' 且注释明确 '"both" needed for HeadFrame (heading calibration)' (vr_receiver_process.py:51)；入口传 'right'。接收循环本身已无条件跳过 LEFT 手帧 (vr_receiver_process.py:320-322)，所以 'right' 对手部数据无增益，唯一效果是 HandFilter('right') 可能让 SDK 不再下发 HeadFrame，使打包进每帧的 head_pos/head_quat_wxyz 恒为初始单位四元数 (vr_receiver_process.py:272-275)。当前入口从不调用 ArmWristMapper.set_heading（controller._reset_mapper 只传 wrist/eef，controller.py:701-706），head 位姿未被消费，故现在无功能影响；但一旦按 arm_mapper.py:103-155 的设计接入 heading 校准，会拿到单位值静默失效。

### `examples/real/vr_teleop_shm.py:227` — controller.run() 后的清理不在 try/finally 中，非预期异常会跳过 vr_receiver.stop()/robot.disconnect()
- 来源: x-safety | 验证置信: low

主生产入口整体合规（validate_action/急停/VR 断连超时均由 TeleopController 承担，controller.py:346/476/283 已核实），且 run() 内部 finally _shutdown() 保证内环停止、臂安全。剩余小缺口：run() 只捕获 RuntimeError/ConnectionError/ValueError（controller.py:243），其他异常在 _shutdown 后继续向上传播，225 行之后的 vr_receiver.stop()（228）与 robot.disconnect()（229）被跳过——VRReceiverProcess 为 daemon 进程会随主进程消亡，仅遗留连接未优雅关闭，无运动风险。

### `examples/sim/keyboard_teleop_sim.py:398` — --headless 模式下 SAPIEN 场景无桌面/地面 actor，但规划器仍注册桌面障碍——规划与物理不一致（保守方向，无害）
- 来源: kbd_sim | 验证置信: low

行 396-404 的注释称桌面障碍 '匹配 SAPIEN 场景中 constructor.py 的 table actor'，但 SimRobotInterface.connect() 只在非 headless 时调用 add_base_components()（sim_adapter.py:73-74），headless 下场景既无桌面也无地面碰撞体。结果：headless 时规划器/teleop IK 仍会因桌面障碍在 z≈0 附近 hold（TeleopProfile.check_env_collision 默认 True），而物理上机械臂本可穿过该区域。方向保守（多拦不少拦）且 WORKSPACE z_min=0.05 高于桌面，实际影响仅为 headless 行为与注释所述场景不完全对应。add_table 参数本身核对无误：table_height=0.0, half_z=0.04 → 盒中心 z=-0.04、顶面 z=0，与非 headless 时 SAPIEN 桌面（中心 [0.4,0,-0.5]、half_z=0.5、顶面 z=0）一致。

### `examples/sim/keyboard_teleop_sim.py:610` — 主循环直接调用 sim.robot.apply_action + sim._step_physics（私有方法），绕过 SimRobotInterface.send_action()——属仿真示例的既定例外模式
- 来源: kbd_sim | 验证置信: low

行 609-611（以及行 391 的 sim._step_physics(n=10)）绕过 send_action() 的关节限位裁剪与 last_qpos_cmd 记账，直接驱动底层模型并调私有 _step_physics。降级为 info 的理由：库自身的 sim_helpers.execute_dense_path/settle_at_target（sim_helpers.py:42-44、76-78）就是同一模式；apply_action 内部已做 qlimits 裁剪（xarm7_xhand.py:252-255）故安全等价；SimRobotInterface 本无 validate_action 可跳过，且其 docstring 明示 '非 RobotInterface 直接替代品'。CLAUDE.md 的 '硬件只经 RobotInterface' 约束针对真机，此处不构成违规，但意味着 step_count/last_qpos_cmd 等接口侧状态不更新。

### `examples/sim/test_motion_planning_sim.py:164` — CollisionModel 包装函数保留旧 FK-Z 签名但返回占位值，日志与 docstring 与实现不符
- 来源: mp_sim | 验证置信: low

check_path_desk_safety/check_hand_desk_clearance 为兼容旧调用者保留 (safe, min_z, idx/name) 签名，但 FCL 化后 min_z 恒为 0.0 或 inf、name 恒为 'fcl'/'ok'。调用处仍按旧语义打印：643 行 '{start_name} z={start_z:.3f}m < desk+margin' 和 678-679 行 'fingertip_z_min={min_z:.3f}m' 在违规时永远打印 'fcl z=0.000m'，诊断信息失真。check_hand_desk_clearance_sim 的 docstring（167-172 行）称 'planner cm is 7-DOF, hand fixed at home'——实际 main() 的 cm 是 19-DOF 且 _to_full_qpos 会用 set_hand_qpos 存入的当前抓取手型自动扩展（collision_model.py:307-330），并非固定 home 手型；check_path_desk_safety_sim（179-188 行）自称 'using current sim hand' 却与 check_path_desk_safety 实现完全相同，未从 sim 取任何手型。这些函数的调用者目前均为死代码，无运行时危害。

### `examples/sim/test_motion_planning_sim.py:1557` — 全文件绕过 SimRobotInterface.send_action 直接驱动 sim.robot 并调用私有 _step_physics（仿真测试脚本的有意例外）
- 来源: mp_sim | 验证置信: low

执行路径统一使用 `sim.robot.balance_passive_force(); sim.robot.apply_action(...); sim._step_physics(n=...)`，跳过 SimRobotInterface.send_action（后者做 qlimits 裁剪与计步）。这属于 CLAUDE.md '硬件只经 RobotInterface' 的仿真侧例外：库自身的 sim_helpers.execute_dense_path/settle_at_target 采用完全相同的模式（sim_helpers.py:42-44, 76-78），且 XArm7XHand.apply_action 内部自带 clip_action 限幅（xarm7_xhand.py:252-253），因此无正确性风险；validate_action 也不适用于 SimRobotInterface（其接口签名与 RobotInterface 不同，sim_adapter.py:50-52 明示非替身）。仅 `sim._step_physics` 与 `cm._collision_model.ngeoms`（1864 行）的私有成员访问值得留意。降级为 info。

### `examples/sim/test_motion_planning_sim.py:1827` — --headless 时 SAPIEN 场景没有物理桌面：headless 与 GUI 运行的物理结果不可比
- 来源: mp_sim | 验证置信: low

SimRobotInterface.connect() 仅在 `not headless` 时调用 add_base_components（sim_adapter.py:72-74），即物理桌面 actor 和地面只在带 GUI 时存在。--headless 运行中唯一的'桌面'是 CollisionModel 里的 FCL 盒（只用于检测，不产生接触力），check_env="warn" 的阶段路径若下探穿过桌面平面，headless 下顺利执行，GUI 下手指会与 kinematic 桌面产生真实接触、PD 收敛误差和 HOLD/settle 统计随之不同。用同一 seed 比较两种模式的 holds/成功率会得出不一致结论。这是库的行为（组件按'可视化'归类），本文件原样继承，值得在脚本或库层面注明或统一。

### `examples/sim/vr_teleop_sim.py:23` — 文档字符串引用了不存在的 CLI 参数，并提及已废弃的键盘实现技术栈
- 来源: vr_sim | 验证置信: low

文档字符串（第 23-25 行、第 50-55 行）记录了不存在的命令行参数：--pre-record-duration、--success-dir、--failure-dir 在 argparse（第 369-399 行）中从未定义。实际代码仅使用单一的 --data-dir，并依赖 CollectionConfig.save_sidecar_json 输出 sidecar JSON。此外，文档字符串（第 51 行）声称键盘处理使用“cbreak 键盘：termios + select”，但 KeyboardHandler（keyboard.py）使用基于 pynput 的全局按键捕获。

### `examples/sim/vr_teleop_sim.py:105` — sim 入口有意保留 50Hz（循环实频已确认），但已不再镜像生产 16Hz 动力学与平滑配置
- 来源: x-16hz | 验证置信: low

vr_teleop_sim.py 与 keyboard_teleop_sim.py:72 的循环实际以 RateLimiter(50) 运行，全部参数在 50Hz 下自洽（EMA 0.8/0.4、retargeter 默认 α=0.6、recorder 默认 control_hz=50 标注正确），属有意保留而非遗漏。但生产已迁 16Hz + EMA 1.0/1.0 直通：sim 的停走临界 v*=jT²/32 与真机差约 10 倍、TeleopPipeline 平滑配置相反，在 sim 中验证的遥操手感/IK 行为结论不可直接迁移到 16Hz 生产；且 sim 录制默认目录 ./episodes（50Hz 文件）与生产 16Hz 目录同池（见发现 1）。建议 sim 至少在文档/打印中标注与生产的频率差异，或提供 CTRL_HZ=16 档位做部署一致性验证。

### `examples/sim/vr_teleop_sim.py:950` — 未使用的变量 ep_elapsed——死代码
- 来源: vr_sim | 验证置信: low

在第 950 行计算了 ep_elapsed 变量（ep_elapsed = now - tick_start + episode_tick_count * CTRL_DT），但在状态打印代码块（第 946-990 行）中从未使用。日志行实际使用的是 rec_frames 和 ep_dur（第 952 行）。这是死代码——无害，但可能引起混淆。


## 被反驳的发现（对抗验证判定不成立，供参考）

- `examples/real/vr_teleop_arm_only_record.py:330` EpisodeRecorder 默认 control_hz=50.0 未显式从 CTRL_DT 推导 — CTRL_DT 改动时记录网格与环路频率失配 — 发现的代码事实全部核实无误：examples/real/vr_teleop_arm_only_record.py:148 为 CTRL_DT=0.02（50Hz），行 330 的 EpisodeRecorder(data_dir="episodes_arm", max_frames=3000) 未传 control_hz，行 392 RateLimiter(1.0/CTRL_DT)=50Hz；de…
- `examples/real/vr_teleop_arm_only_record.py:635` robot.arm.is_error() 仅检查臂错误 — 手错误静默通过预发安全门 — 发现的机械事实全部属实（635 行确为 robot.arm.is_error()；interface.py:136-137 组合检查；interface.py:194-202 及 xhand.py:489-498 手读取失败返回 NaN；762-763 行跳过 set_hand_qpos），但其"缺陷"定性不成立，理由如下：(1) 该脚本是显式支持"手不可用降级运行"的 arm-only 入口（v…
- `examples/real/vr_teleop_arm_only_record.py:275` robot.arm.clear_error() 绕过 RobotInterface.clear_error() — 残留手错误无法清除 — 发现对 XHand.clear_error() 的语义和后果均属误读，我逐层核实后结论如下：

1. 手侧 clear_error() 根本不触碰硬件。xhand.py:398-402 的 XHand.clear_error() 只重置进程内 Python 标志（error_state/last_error_code/last_error_message），不向手发送任何清错命令。与之对比，XAr…
- `examples/real/vr_teleop_arm_only_record.py:570` recording_active 无条件设为 True — 未检查 start_episode() 返回值 — 发现的表面事实正确（examples/real/vr_teleop_arm_only_record.py:563-570 确实未捕获 start_episode() 返回值就无条件设 recording_active = True），但其声称的失效机制经核实不成立。

关键误读：发现称"若 start_episode() 因前一个异步 stop_episode 的 join 超时 (episode…
- `examples/real/vr_teleop_arm_only_record_plus.py:451` _emergency_stop() 调用 robot.arm.clear_error() 立即重新使能伺服，抵消急停效果 — 代码事实链属实但所述后果不成立，核心论断基于两处误读。(1) "臂伺服立即重新上电"是误读：XArm7.stop() (dexmani_real/robot/xarm7/xarm7.py:185-190) 仅 set_state(4)+error_state=True，从不调用 motion_enable(False)——ESC 路径下伺服全程通电、主动力矩保持，state 4 与 state 0…
- `examples/real/vr_teleop_arm_only_record_plus.py:681` 绕过 RobotInterface 直接访问 XArm7 driver 内部属性 — 发现引用的代码位置属实（vr_teleop_arm_only_record_plus.py L680-693 确实下钻到 driver 和 raw SDK），但判定不成立，理由有三。(1) 发现者误读了封装的 API 语义：RobotInterface.is_error() (interface.py:136-137) 会 OR 上 hand.is_error()，而 XHand.is_error…
- `examples/real/vr_teleop_arm_only_record_plus.py:208` do_return_home 启动新 ArmInnerLoop 前未确认旧线程已 join — 发现的表层事实属实但前提和后果链均不成立。

1. 属实部分：ArmInnerLoop.stop()（dexmani_real/robot/inner_loop.py:218-227）确实 join(timeout=3.0) 后只打 warning、无返回值；do_return_home（examples/real/vr_teleop_arm_only_record_plus.py:199-210…
- `examples/real/keyboard_teleop_real.py:598` Hand 未连接时每帧发送 NaN 命令到 XHand 硬件 — 发现的前半段属实，但核心后果（NaN 写入/发送到硬件）在其所述场景下不会发生，被 xhand.py:516-517 的入口守卫拦截。

已核实的部分：hand 连接失败时 state.hand_qpos 确实是 NaN(12)——但路径与发现所述不同：XHand.get_state() 未连接时不抛异常，read_raw_state (xhand.py:682-684) 因 connected_…
- `examples/real/keyboard_teleop_real.py:502` wall_warned 节流跨轴共享，允许连续触发边界警告 — 核实位置：examples/real/keyboard_teleop_real.py:317-318（wall_warned/last_wall_time 初始化，grep 确认全文件仅 317/504/507 三处出现、无重置点）、:499（np.clip 边界钳制，真正的安全逻辑）、:500-508（警告条件 `if not wall_warned[axis] or now - last_wa…
- `examples/real/test_motion_planning_real.py:408` 递增运动验证 Step 2 对全部 7 关节同时下发单次 +2° Mode-1 伺服阶跃，超出全库 ≤1° 插值规范 — 发现的事实部分（行408一次性下发全部7关节+2°经Mode 1）正确，但所述后果不成立，基于以下三点代码证据：

1. **Mode 1 并非瞬时跳变。** `send_action` → `set_servo_angle_j`（xarm7.py:264）走Mode 1伺服模式。xArm SDK中`set_servo_angle_j`接受可选speed/acc参数，未传入时使用固件默认伺服速度（…
- `examples/real/test_quest_hand_teleop.py:183` 初始 last_qpos 无 isfinite 检查，特定失败序列下 NaN 位置命令直发手部固件 — 发现声称的代码路径不可达：初始 last_qpos 为 NaN 后，retarget 返回 None 导致 NaN fallback 进而 send_action 发送 NaN 到固件。

经过完整追踪，retarget 链中的每一步都不会返回 None：

1. test_quest_hand_teleop.py:250 调用 retargeter.retarget(mano_landmarks…
- `examples/real/vr_teleop_arm_only_record_plus.py:181` 两个 16Hz 生产入口的 Cartesian EMA 策略互斥：record_plus τ 换算平滑 vs shm 直通 — 该发现基于对两个结构不同的入口点（entry point）的混淆，认定它们应使用相同的 Cartesian EMA 参数。逐一驳回：

**1. 架构不同，非同类入口**
- `vr_teleop_shm.py:174-192` 经过 `TeleopController` → `TeleopPipeline`（`controller.py:190-192`），其 `ema_alpha_pos/em…
- `examples/real/test_quest_hand_teleop.py:157` 已标 DEPRECATED 的脚本直接实例化 XHand 驱动手部运动，无 ESC 急停 — Finding correctly observes architectural deviation（test_quest_hand_teleop.py:157 直接 XHand(hand_config)，line 279 send_action() 驱动硬件，line 86-90 仅 'q' 键无 ESC），但不构成实际缺陷：

1. **安全闸门在 send_action() 内部而非 Rob…
- `examples/real/vr_teleop_arm_only_record_plus.py:425` record_plus 的 _stop_recording 缺少键盘缓冲区清空，耗时 HDF5 保存后积压按键可能误触发 H/Q — 发现的结构性事实属实（vr_teleop_arm_only_record.py:416-417 有 kb.poll(timeout=0.0) 清空，vr_teleop_arm_only_record_plus.py:425-436 的 _stop_recording 没有），但其核心机制——"stop_episode 需要等待异步写入线程完成（最多 10s），此期间按键积压后误触发"——是对 AP…
- `examples/real/vr_teleop_arm_only.py:323` A 的 data_dir="episodes" 与 B/C 的 data_dir="episodes_arm" 不一致 — 引用本身属实：examples/real/vr_teleop_arm_only.py:323 为 EpisodeRecorder(data_dir="episodes", max_frames=4500)，vr_teleop_arm_only_record.py:330 与 vr_teleop_arm_only_record_plus.py:343-344 为 data_dir="episodes…
- `examples/real/vr_teleop_arm_only.py:323` A 的 max_frames=4500 (90s@50Hz) 与 B/C 的 60s 等效值不一致 — 发现引用的三处代码属实且换算正确（我逐一核实）：vr_teleop_arm_only.py:323 为 max_frames=4500，且该文件 L141 CTRL_DT=0.02（50Hz），4500/50=90s；vr_teleop_arm_only_record.py:330 为 max_frames=3000（50Hz→60s）；vr_teleop_arm_only_record_plus…
- `examples/real/keyboard_teleop_real.py:514` 键盘遥操作对：真机用 Cartesian EMA+松键 snap，仿真用 EEF 速度硬限幅+松键继续收敛，运动语义不可迁移 — 
**核心反驳：finding 声称的“空闲时继续向 prev_arm_cmd 收敛（松键后仍会走完最后目标）”经代码追溯不成立。**

关键证据链：

1. **sim 中 target_pos 每 tick 被覆盖为速度限制后的值**（sim:576-577）：
   ```python
   target_pos = ik_target_pos
   target_quat = ik_tar…
- `examples/real/vr_teleop_arm_only_record.py:729` add_frame 不传 signals/camera_frame，flag_ik_ok/flag_camera_fresh 始终为 False — 发现描述的是过时代码状态，与当前仓库不符。核实过程：(1) 当前工作树 examples/real/vr_teleop_arm_only_record.py 第 729 行是 "ema_prev_pos = ema_prev_quat = None"（笛卡尔 EMA 重置），根本不是 add_frame 调用；全文件唯一的 add_frame 在第 826-827 行："sig = {\"ik_o…
- `examples/real/replay_traj.py:764` replay 发送动作前不调用 validate_action，违反项目安全约定 — 事实层面属实但后果不成立。我确认 replay_traj.py:764 set_target 与 :778 send_action 之间确无 validate_action 调用，但逐层核实后，validate.py:21-94 的 8 项检查在 replay 路径上均有等效覆盖，所述"硬件过载"后果不会发生：(1) 手部限位——validate_action 第 8 步的 np.clip(han…