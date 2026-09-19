对当前工作树按 `.claude/tasks/research-runtime-refactor/TASKBOOK.md` 做独立、只读的交付审查。这是已实施代码的审查，不是重新设计更复杂的系统。先读 AGENTS.md、CLAUDE.md、TASKBOOK 和本地 `$(git rev-parse --absolute-git-dir)/research-runtime-refactor/STATUS.md`，再核对 Git log、IMPLEMENTATION_BASE→HEAD diff、staged/unstaged diff、测试证据和实际源码。STATUS 只是索引，不是通过证明。

不修改文件、不提交、不 push、不安装依赖、不连接硬件、不执行 examples 或未检查的测试。测试结果只引用实际日志和明确运行范围；缺少证据记验证缺口，不把它自动当作代码 bug，也不把 skip 当通过。

按文件/函数追踪，重点找以下问题：
1. FULL 是否仍会隐含丢弃/替换 candidate、推进 index/reference/clock，或忙轮询；Teleop pause 是否会发送旧人类意图；所有 publication caller 是否适配。
2. generation reset、旧 SDK 晚返、absent actuator、慢 hand、CRC、中间 setpoint 与 exact endpoint ACK 是否正确；是否把 FIFO/SDK 接受冒充物理同步/收敛。
3. 同一 soft delta 是否仍重复 hard reject；IK miss 是否仍 terminalize；replan 是否误清已提交前驱 reference；真实硬界限是否被削弱。
4. age/skew/grid-lag/post-inference/pointcloud/camera 或 loop heartbeat 是否仍间接 veto 合法慢推理；因果、模态身份、旧 tactile 与真正 source stall 是否保留。
5. Recorder 错误是否经 START/poll/service/supervisor/计数/cleanup 间接停止 RUNNING；partial 是否被删除或伪装 complete；旧 B 是否覆盖 S/Q。
6. 原 raw 是否保持内容与可读性；legacy 支持是否有真实依据；--processed replay 是否仍能找到并正确读取 raw 命令；是否误用 processed float32 动作。
7. EX01–EX14 的 CLI/help/config/session/outcome/reader 是否闭环；无 trace sync raw 是否被误报损坏；是否存在危险的 --help/示例测试。
8. 真实拒绝/丢弃是否一行可见，同一等待是否刷屏；四项统计是否有正确来源/分母；新抽象是否是 v2 已取消的框架。
9. README/docstrings/comments/tests 是否仍描述旧行为；commits 是否夹带数据/秘密/用户改动；完整 diff 与验证范围是否一致。

输出按风险排序的具体 findings：文件:行号、可复现条件/数据流、实际后果、最小修复方向和针对性验证。没有证据不要制造 finding，不因“还能更工程化”要求增加 validator/timeout/registry。另列缺验证、明确条件暂缓、无需修改的合理差异；用 V01–V22/EX01–EX14 标记遗漏。最后给出核心离线交付是否满足任务书及仍未覆盖的硬件/真实数据范围。不要声称无 findings 就等于真机安全已验证。
