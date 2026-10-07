# 离线复现入口

本仓负责真实机器人工作流和 Raw → canonical 导出；[dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy) 负责 Dataset、训练与推理。两仓采用源码 checkout + editable 安装。保留本仓 `assets/`，不把 wheel 安装视为受支持的资产分发方式。

Raw、Canonical 和预测 sidecar 的不可覆盖发布使用 Linux/libc `renameat2(RENAME_NOREPLACE)`，将目标占用检查与重命名合为一次原子操作。环境或文件系统不支持时发布报错并保留 staging，不退回可能覆盖已有产物的普通 rename。

2026-10-07 本地核实环境为 Linux x86_64、Python 3.10.20，NumPy 1.26.4、SciPy 1.15.3、PyAV 17.1.0、h5py 3.16.0、Zarr 2.18.3、OpenCV 4.9.0.80、Pinocchio（`pin`）2.7.0、NLopt 2.7.1、mplib 0.2.1。开发检查使用 pytest 9.1.1、Ruff 0.16.8；Policy CPU 定向测试使用 PyTorch 2.4.1+cu124。这是一次核实的环境记录，不是跨版本兼容承诺，也不要求升级已有实验机。

基础和 dev 依赖见 `pyproject.toml`；桌面键盘另选 `interactive`。完整几何导出需要 Pinocchio，选定 retargeting 需要 NLopt，碰撞/规划需要 mplib；厂商 SDK 依实验机实际安装，不加入通用 pip 依赖。纯 `--help` 和声明配置无需这些后端。缺 PyAV 无法创建/解码 Raw 视频；缺 Pinocchio 无法完成全 13 字段导出；缺 Policy 不影响下面三个 Real 数据入口，但不能验收 Policy Dataset/推理桥接。

驱动使用当前 SDK 的明确字段：xArm 要求 `connected`、`mode`、`axis`、`error_code`；XHand 关节板状态读取厂商原名 `jonitboard_err`，内部统一为 `jointboard_err`。缺少必需字段不会补成已连接、正确模式或无错误；辅助电流缺测仍保留 NaN。

VR 离线回归使用本机 `hand_tracking_sdk 1.1.0` 的公开 transport、parser 和 assembler 接口，以 fake transport 替换网络读写；未安装 SDK 时该组测试跳过。worker 自行检查每次读取超时后的退出请求，因为 SDK 的高层迭代器在内部持续重试超时。源端时间戳和序号越出 uint64 范围时拒绝该帧，不截断或回绕。默认 pytest 仅发现本仓 `tests/`，不扫描实验归档的源码快照。

在 Real 根目录、已配置的 Python 环境中运行：

```bash
python -m pip install -e ".[dev]"
# 需要 Policy 桥接时，按相邻仓 README 配置其环境并 editable 安装。
```

小样本全为 synthetic，不含真实采集、真实策略预测或物理成功证据。它使用真实 writer、EpisodeReader、公共点云/FK 处理和原行号取窗。所有目标、时间和 dispatch 都是明确标注的测试值，触觉故意缺测为 NaN。

```bash
sample_dir=$(mktemp -d /tmp/dexmani_synthetic.XXXXXX)
python examples/create_synthetic_episode.py "$sample_dir/raw"
python examples/visualize_episode.py "$sample_dir/raw/episode_synthetic" --info
cat > "$sample_dir/processing.yaml" <<'YAML'
environment:
  table:
    plane_path: null
    plane_abcd: [0.0, 0.0, 1.0, -0.022]
YAML
python examples/export_policy_zarr.py "$sample_dir/raw" \
  --output "$sample_dir/canonical.zarr" --config "$sample_dir/processing.yaml"
python examples/read_policy_windows.py "$sample_dir/canonical.zarr" --horizon 4
```

预期产物为 8 行、名义 10 Hz 的 Raw（`data.h5`、`rgb.mp4`），以及保留全部 13 字段、`row_info`、episode 边界和 `export_report.json` 的新 Zarr。默认点云各处理阶段保持启用；这里只为合成场景显式指定内联桌面。窗口示例报告 5 个 RGB 窗口和 0 个触觉窗口，不把 NaN 填成有效数据。重复使用已存在输出路径会拒绝覆盖。`--info` 和取窗不打开 GUI。

```bash
python -m compileall -q dexmani_real examples
ruff check dexmani_real examples tests
ruff format --check dexmani_real examples tests
python -m pytest -q tests
git diff --check
```

完整本地测试集包含原生几何和相邻 Policy 消费检查，需上述完整环境；故障调度测试使用 fake robot/model/clock，不接设备。单独的数据贯通检查在 `tests/test_synthetic_workflow.py`。它们不验证真实权重、GPU 时限或硬件行为。

真实实验沿 [README](../README.md) 的标定 → 示教 → 检查/导出 → Policy 训练 → [部署](policy_execution.md) 工作流进行。配置从 `experiment.example.yaml` 复制到自己的实验路径，明确选择当前标定、Raw、新缓存和输出目录；Policy 使用对应配置、checkpoint 及其保存的 normalizer/点云配方。本页不提供已发布论文、真实数据或模型下载，相关地址、许可证、第三方资产再分发权利和引用元数据需作者确认。未经确认不上传资产，不从 synthetic 样本报告实验成绩。

真机入口即使 `execute=False` 仍可连接设备；上面的离线命令不能替代现场的触觉无接触确认、停止测试、标定确认或保守运动验证。
