请在当前 dexmani_real 工作树执行 `.claude/tasks/research-runtime-refactor/TASKBOOK.md`，完成允许范围内的代码修改、相关文档/注释更新、离线验证和分阶段本地 Git 提交，不要只返回计划或伪代码。

先读取根 AGENTS.md、CLAUDE.md、受影响目录的局部指令，再完整读取 TASKBOOK。任务书已吸收 Handoff 与 v2 的最终修订，不依赖聊天历史；不要恢复旧版的 FULL 即终止、ACK/STOP registry、心跳专用线程、预 ACK 录制缓存或完整 source archive。

开始前核对仓库/branch/HEAD、worktree/index/untracked 和现有实现。若有用户未提交修改、当前在 main、分支/目标不明确，先按 T0 处理，禁止自行 stash/reset/clean/覆盖。若已有本任务 commits/STATUS，先复核再续做，不能重复重构。冻结 IMPLEMENTATION_BASE，使用 `$(git rev-parse --absolute-git-dir)/research-runtime-refactor/STATUS.md` 保存本地进度与证据，不提交运行日志或真实数据。

用不超过 12 条列出实际执行顺序和已发现的具体偏差，然后立即开始最小可验证阶段；不需要再次请求我批准已经明确定义的无硬件代码修改。每阶段完成 producer→representation→consumer→side effect 闭环，同时改相关配置、测试、docstrings、README/help。一个主实现者负责共享核心文件；可用只读审查，但不要多个 agent 并发改 executor/lifecycle/transport。

执行时必须满足：
- 原 raw/processed/Zarr/checkpoint 和用户修改不覆盖；现有 reader/schema 含义不变。先只读盘点已知数据根，禁止扫描整个 home。未知 legacy 只阻塞真实转换/兼容子项，不臆造字段。
- FULL 保留同一 candidate/action ID，回主循环等待；不 pop、不更新 reference/clock、不换 chunk。真正 sequence 丢失才报 transport failure；STOP/Teleop pause 能撤销旧 pending。
- 单 owner projection，普通 IK/workspace miss 显式 drop 未发布后缀后 replan；保留已提交前缀。Policy 删除 generic timing gate 及上游间接 gate，保留因果/真实读取/旧 tactile/硬边界。I/O liveness 保留，不用 loop heartbeat 否决慢 predict。
- Recorder 使用非 RUNNING 的现有 bounded START；运行中 evidence 错误不结束控制。Trial 与 saved count 独立；camera_stall 保存前缀，安全关闭的 partial 不销毁。
- 每次真实 drop/reject 一行可见；同一 WAIT/retry 不刷屏、不仅 DEBUG。14 个 examples 逐个核对，不能只改底层签名。
- 不接设备、不 home/servo/teleop/replay/rollout、不写标定、不批跑 examples/--help。先检查 imports/测试路径，再运行 offline checks。不要把 execute=False、通过 CLI 权限提示或这个任务当成真机授权。

按 T0–T7 推进。缺真实 raw、hand 容差、模型 artifact、依赖或硬件授权时，给出精确 BLOCKED/DEFERRED_CONDITIONAL 和所需输入，继续不受阻的安全工作；不要猜安全参数，也不要宣称全项完成。不擅自升级系统依赖/SDK或改相邻 dexmani_policy。可选 decision sidecar 不阻塞核心交付。

每个行为闭环验证后审查 focused diff 与 staged diff，仅显式 stage 本任务文件并创建本地 commit；源码/注释/帮助和必要测试一起提交。不要 git add .、amend、--no-verify、push、merge 或提交真实数据/秘密。没有配置 Git 身份或 hooks 失败时如实报告，不自动绕过。

最终按 TASKBOOK 的 V01–V22、EX01–EX14 做反向审查，重点检查背压 reference、旧 generation 晚返、间接 timing veto、recording→supervisor 终止链、raw reader/replay 兼容以及注释和参数漂移。修复有依据的问题，再验证和提交。测试失败、skip、未执行要区分；用 tee 时保留退出码；最终检查 IMPLEMENTATION_BASE→HEAD 的完整 diff 和工作树，不能只检查一个干净的 git diff。

交付：分阶段完成矩阵、删除/保留机制、14 个 examples 结论、实际验证命令和结果、数据不变/兼容结论、本地 commit SHA/标题、最终 Git 状态、条件阻塞与未验证范围。中断前先更新 STATUS 的已做/未做/下一步；恢复时以 Git 和源码事实为准。现在开始执行。
