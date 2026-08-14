# Phase B — XHand EtherCAT Lifecycle A/B Record

> 真人执行，Claude 不碰硬件。一次只改一个变量，用 git branch/commit 做 A/B。
> 基线 = `main`（两阶段 discovery + INIT-watchdog disconnect）。
> 实验 = `b1-single-controller`（B1），选定后再开 B2。

## 0. 环境记录（每次实验填一次，§8.1）

| 字段 | 值 |
|---|---|
| Git commit（baseline） | `main` @ `62d803d` |
| Git commit（B1） | `main` @ `789418e`（cherry-pick 自 `b70eeca`） |
| Git commit（B2） | `b2-close-only` @ `2b5a44b`（含 `32c0823` close-only） |
| XHand sdk_version | `1.4.6` |
| XHand serial_number | `012R320220251128022` |
| hand type | `R` |
| Python version | `3.10.20`（conda env `real_robot`） |
| controller package / native lib 位置 | `xhand_controller`（import 名 `xhc`） |
| EtherCAT interface / device_name | `eno1`（discovery 模式，`device_name=None`） |
| process start method | harness 单进程；driver 在循环内 create/destroy（非 mp spawn） |
| OS / kernel | `Linux workstation 6.17.0-40-generic`（Ubuntu 24.04） |

## 0.1 运行方式（自动化 harness，真人执行）

`checks/hardware/xhand_reconnect_soak.py` 自动跑 connect→fresh-read→disconnect 循环（无 motion），
写 JSONL 结果并自动统计 sdo / retry / latency。**每个分支各跑一轮，用不同 `--out`**：

```bash
git checkout main
conda run -n real_robot python checks/hardware/xhand_reconnect_soak.py --cycles 100 --out main_soak.jsonl

git checkout b1-single-controller
conda run -n real_robot python checks/hardware/xhand_reconnect_soak.py --cycles 100 --out b1_soak.jsonl
```

- 输出逐行进度 + 末尾 `SOAK SUMMARY`（connect success/fail、error code 分布、`write sdo failed` 计数、open retries、latency avg/max、identity）。
- 单次 connect 失败会记录并继续（当作 next-session reconnect 观察自愈）；连续 `--max-consecutive-failures`（默认 3）次失败判定 wedged 停止并打印 power-cycle 指令，用 `--start-cycle <n>` 续跑。
- 每个 run 的 vendor log 落在 `./xhand_soak_logs/run_<ts>_<pid>/`，与另一分支隔离，`write sdo failed` 按 run 统计不会跨分支串味。
- 把两份 `SOAK SUMMARY` 关键数字抄进 §1 表，再做 §2 判定。

## 1. B1 正常 reconnect soak（§8.2）

目标：≥100 次起步，500 次更有判别力。每循环：
`create worker/driver → connect → 一次 fresh read →（可选只读健康检查）→ disconnect → destroy → 短间隔 → reconnect`

> ⚠️ 没有明确运动授权时**不要发 finger motion command**。

| 指标 | baseline（`main`） | B1（`b1-single-controller`） |
|---|---|---|
| 总循环数 | 100 | 100 |
| connect success | 100 | 100 |
| connect fail | 0 | 0 |
| `open_ethercat` error code（分布） | 0 次非零（1 次 transient "No device found" 后重试成功，见下） | 0 次 |
| `write sdo failed` 出现次数 / 字符串 | 0 | 0 |
| 需要 power-cycle 次数 | 0 | 0 |
| close/reconnect hang 次数 | 0 | 0 |
| 下次 session 无法 reconnect 次数 | 0 | 0 |
| open retries（`succeeded on attempt 2+`） | **1（cycle 77）** | **0** |
| connect 平均/最大 latency | avg=1387.4ms / **max=15326.3ms**（cycle 77 重试） | avg=1243.3ms / max=1330.7ms |
| disconnect 平均/最大 latency | avg=2018.3ms / max=2029.0ms | avg=2019.0ms / max=2044.5ms |

关键 vendor stdout/stderr（复现时粘贴）—— baseline cycle 77 的 transient stale-slave 重试：

```text
[22:43:52] [WARNING] [dexmani_real.robot.xhand] XHand open attempt 1/2 vendor output:
ec_init on eno1 succeeded.
[22:43:52] [WARNING] [dexmani_real.robot.xhand] XHand connect attempt 1/2 failed: No device found — waiting 3.0s for potential stale-slave recovery before retry...
[22:43:55] [WARNING] [dexmani_real.robot.xhand] XHand connect succeeded on attempt 2/2 (retries indicate SDO/communication glitch) — adding 1.0s post-recovery stabilisation delay.
```

**附加发现（teardown segfault，非 per-cycle 指标）**：
baseline 这轮在 100 个 cycle 全部完成、`SOAK SUMMARY` 打印 `stop reason: completed` **之后**，
进程退出时 native `EcatUpdateThread` 在解释器卸载 `.so` 时 use-after-free → SIGSEGV（exit 139）。
vendor log 尾部是一条线程/cycle 的 `EcatUpdateThread: Operation not permitted`（RT 调度失败，driver 已忽略）。
B1 复跑 `--cycles 2` **确认 `EXIT_CODE=0` 干净退出**（无 segfault），且 B1 的 `EcatUpdateThread` 消息
在 run 启动即打印（每条/线程），而 baseline 是 teardown 后洪流——两者 native 线程回收时序不同。
故 teardown segfault 是 baseline 侧现象，B1 未复现（确认两次：先 100-cycle 无崩溃痕迹，后 2-cycle exit 0）。

## 2. B1 判定（§8.3，go / no-go）

仅当**全部**满足才保留 single-controller：

- [x] 正常循环无 reproducible `write sdo failed` 回归 —— 两分支均 0
- [x] failure rate 不高于 baseline —— 均 0/100；B1 反而少 1 次 transient "No device found"（0 vs 1）
- [x] 不增加 power-cycle requirement —— 均 0
- [x] 不出现 close/reconnect hang —— 均 0
- [x] repeated run 后资源无明显泄漏 —— B1 干净退出（exit 0，确认两次）；baseline teardown segfault（exit 139，native EcatUpdateThread 未被 join）。B1 不劣于 baseline，反而少了 baseline 的 teardown 崩溃
- [x] 失败时 diagnostics 仍足以定位 —— 两分支 vendor log 完整（含 open attempt N/M、stale-slave 恢复、succeeded on attempt）

**只有**出现可重复现象 `single-controller → SDO fail；isolated discovery → 同 setup 成功`，
才有证据保留 isolated discovery workaround —— 此时才加**极窄** compatibility switch，不提前加。
→ 本次未出现该现象。

结论：**go**（推荐保留 B1）。理由：100 次循环 reconnect 可靠性两分支均 100/100 无回归；
B1 的 single-controller 路径在 100 次中 0 次 transient "No device found" 重试，baseline 出现 1 次（cycle 77，
连带 3s stale-slave 恢复 + 1s 稳定化 = 15.3s connect 尾部）。方向性支持 B1，但 n=1 判别力弱，
建议后续按 §8.2 扩到 500 次再确认。teardown segfault 是 SDK 层 close_device 未 join native 线程的独立问题
（发生在完成后、不影响 per-cycle 指标），单独跟踪，不作为 B1 否决项。

## 3. B2（仅 B1 选定后，§8.4）

close-only disconnect 验收（在正常 reconnect soak 之外）：

- [x] normal exit —— exit 0（无 baseline teardown segfault）
- [ ] SIGTERM / orderly process termination —— 未测（harness 只跑 normal 循环退出）
- [ ] Python exception 触发 finally cleanup —— 未测
- [ ] worker crash 后由 supervisor 回收 —— 未测
- [ ] SIGKILL（**只记录恢复行为**，不要求 Python cleanup 执行）—— 未测

判定重点：close-only 后下一次 reconnect 是否稳定、是否仍出现 stale OP / SDO failure、
是否仍需 2–3s watchdog wait。

结果：100/100 connect、100/100 read，但 **16 次 `write sdo failed`（cycle 3 → 15，cycle 4 → 1）** +
1 次 open retry（cycle 4，14.25s）；cycle 3 identity 读回 serial/hand_type `unavailable`（cycle 5 起自愈）。
→ **仍出现 stale OP / SDO failure**，§8.4 判定重点不满足。

结论：**no-go**（拒绝 close-only）。理由：B2 是三轮 soak 中唯一出现 `write sdo failed` 的 run；
移除 INIT+watchdog 后下一次 reconnect 仍出现 stale-OP/SDO 抖动（transient 自愈，非硬 wedge）。
两阶段 disconnect（`_request_slave_init` + 2s watchdog wait）保留为 load-bearing。
归因 caveat（n=1）：16 次 sdo 集中在 cold-start 区（cycle 3-4）且自愈，非 16 个独立观测；baseline
自身也有 1 次同类事件（cycle 77）；B2 一次移除两件事（INIT 请求 + 2s sleep）未单独隔离。disconnect
仅 teardown、不在控制回路，2s wait 无 servo/estop 代价（安全审查确认 4 处 disconnect 调用点均在 command loop 之外）。
