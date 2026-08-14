# Phase B — XHand EtherCAT Lifecycle A/B Record

> 真人执行，Claude 不碰硬件。一次只改一个变量，用 git branch/commit 做 A/B。
> 基线 = `main`（两阶段 discovery + INIT-watchdog disconnect）。
> 实验 = `b1-single-controller`（B1），选定后再开 B2。

## 0. 环境记录（每次实验填一次，§8.1）

| 字段 | 值 |
|---|---|
| Git commit | `git rev-parse --short HEAD` |
| XHand sdk_version | |
| XHand serial_number | |
| hand type | |
| Python version | `python --version` |
| controller package / native lib 位置 | |
| EtherCAT interface / device_name | |
| process start method | spawn / fork |
| OS / kernel | `uname -a` |

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

| 指标 | 计数 |
|---|---|
| 总循环数 | |
| connect success | |
| connect fail | |
| `open_ethercat` error code（分布） | |
| `write sdo failed` 出现次数 / 字符串 | |
| 需要 power-cycle 次数 | |
| close/reconnect hang 次数 | |
| 下次 session 无法 reconnect 次数 | |
| connect 平均/最大 latency | |
| disconnect 平均/最大 latency | |

关键 vendor stdout/stderr（复现时粘贴）：
```text
```

## 2. B1 判定（§8.3，go / no-go）

仅当**全部**满足才保留 single-controller：

- [ ] 正常循环无 reproducible `write sdo failed` 回归
- [ ] failure rate 不高于 baseline（main 分支同条件各跑一轮对比）
- [ ] 不增加 power-cycle requirement
- [ ] 不出现 close/reconnect hang
- [ ] repeated run 后资源无明显泄漏（`ps`/fd/内存观察）
- [ ] 失败时 diagnostics 仍足以定位

**只有**出现可重复现象 `single-controller → SDO fail；isolated discovery → 同 setup 成功`，
才有证据保留 isolated discovery workaround —— 此时才加**极窄** compatibility switch，不提前加。

结论：`go` / `no-go`，理由：____________

## 3. B2（仅 B1 选定后，§8.4）

close-only disconnect 验收（在正常 reconnect soak 之外）：

- [ ] normal exit
- [ ] SIGTERM / orderly process termination
- [ ] Python exception 触发 finally cleanup
- [ ] worker crash 后由 supervisor 回收
- [ ] SIGKILL（**只记录恢复行为**，不要求 Python cleanup 执行）

判定重点：close-only 后下一次 reconnect 是否稳定、是否仍出现 stale OP / SDO failure、
是否仍需 2–3s watchdog wait。

结论：____________
