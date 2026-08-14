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
