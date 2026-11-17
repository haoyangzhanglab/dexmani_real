# CPU 离线环境与回归

验证平台为 Linux x86_64、Python 3.10。安装时需要网络；以下运行过程不连接机器人、相机、VR，也不加载在线 HOI 权重。

## 最小安装与运行

在仓库根目录运行，资产随 checkout 提供：

```bash
python3.10 -m venv /tmp/dexmani-offline
source /tmp/dexmani-offline/bin/activate
python -m pip install pip==25.3
python -m pip install -r requirements-offline.txt
python -m pip check

python scripts/preview_hand_retargeting.py tests/fixtures/hand_retargeting --no-view
python scripts/export_zarr.py tests/fixtures/hand_retargeting \
  --output /tmp/hand-example.zarr \
  --fields joint_state action eef_pose fingertip_xyz hand_pointcloud
```

输出路径必须尚不存在。样本预览应报告 397 帧、0 次拒绝；导出应报告 1 个来源、1 个连续段、397 帧。`--no-view` 不需要 Rerun，纯预览和导出不需要 Policy。Python 3.10 自带的旧 pip 可能误判 CPU wheel 的元数据名称，因此明确使用上面验证过的 pip 版本。

| 功能 | 实际依赖 |
| --- | --- |
| Raw、关节/图像/点云导出 | 项目基础依赖：NumPy、SciPy、PyYAML、h5py、Zarr 2、PyAV、OpenCV、fpsample、tqdm |
| 手部求解 | `nlopt==2.7.1`；Pinocchio 的 PyPI 包名是 **`pin`**，此路径固定 `pin==2.7.0`，导入名为 `pinocchio`；它的原生库由 cmeel wheels 提供 |
| EEF、指尖和手部点云导出 | Pinocchio、仓库 URDF；手部点云使用已提交的 `hand_1024.npz`，无需生成或下载模板 |
| 可视化、接触标注回归 | Rerun、PyArrow、Open3D；Open3D 也供配套 Policy 的 CPU 接触标签测试使用 |
| 控制规划回归 | Ruckig；测试不连接设备 |
| Policy 跨仓回归 | 固定 `dexmani_policy@b1ddc36c932ba5a295a5e439a7241bf10415666b`，CPU Torch 2.4.1、Torchvision 0.19.1 及其声明依赖，详见 `requirements-cpu.txt` |
| 真机／在线感知 | xArm/XHand SDK、RealSense、VR、交互组件；GPU HOI 还需要对应 CUDA 组件和模型资产。这些不属于最小离线路径 |

`requirements-offline.txt` 固定已验证的数值组合和 OpenCV 4.9，避免 NumPy 1.x 环境尝试安装要求 NumPy 2.x 的 OpenCV。完整 Policy 与 viewer 依赖留在回归安装中，未加入基础安装依赖。

## CPU 回归

在同一干净环境继续安装：

```bash
python -m pip install --build-constraint requirements-build.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt
python -m pip check
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m pytest -p no:cacheprovider -q -ra \
  tests/test_hand_retargeting.py tests/test_hand_retargeting_preview.py \
  tests/test_hand_cloud.py tests/test_export.py tests/test_raw_metadata.py \
  tests/test_teleop_session.py tests/test_keyboard_teleop.py tests/test_home.py \
  tests/test_policy_deployment.py tests/test_policy_session.py \
  tests/test_rollout_hoi_recovery.py tests/test_experiment_pipeline.py
```

`requirements-cpu.txt` 明确使用 `+cpu` wheels，不安装 CUDA Torch。配套 Policy 直接安装固定 Git 提交，不读取旁边可能正在修改的工作区。跨仓流水线在临时目录生成小型 Raw、Zarr、训练/恢复 checkpoint，再执行真实 CPU 推理，无需下载预训练权重。

Ruckig 0.17.3 的源码包使用旧的 `cmake.targets` 构建配置，构建约束将 scikit-build-core 限制在 0.10 之前；约束只作用于隔离构建，不改变运行时依赖。

2026-11-17 在不继承系统包的新 venv 实测：最小安装的预览、五字段导出通过；继续安装 CPU 回归依赖后，上述测试 **559 passed**，`pip check` 无冲突。已有研究环境的完整测试为 **882 passed / 4 skipped**，四项跳过均为 CUDA 专项。本轮未连接真机，也未运行 GPU 实验。

[CPU workflow](../.github/workflows/cpu.yml) 用一个 job 先检查最小安装与 CLI，再安装回归依赖并执行上面的测试。PR、main 更新和手动触发均会运行。性能基准由研究者在同机运行；共享 CI 机器不作耗时阈值断言。

## 观测测量复现

先用既有跨仓测试生成小型 checkpoint；`--basetemp` 是本次测试独占的临时目录，再次执行 pytest 会重建该目录：

```bash
python -m pytest -p no:cacheprovider -q tests/test_experiment_pipeline.py --basetemp=/tmp/dexmani-obs-bench
python - <<'PY'
from pathlib import Path
import time
import numpy as np
import torch
import zarr
from dexmani_policy.deployment import load_policy
from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.observation import ObservationBuilder

torch.set_num_threads(1)
root = next(Path('/tmp/dexmani-obs-bench').glob('test_current*/run')).parent
data = zarr.open_group(str(root / 'training.zarr'), mode='r')['data']
build_times = []
def build(rows):
    start = time.perf_counter()
    result = builder.build(rows)
    build_times.append((time.perf_counter() - start) * 1000)
    return result
worker = PolicyWorker(lambda: load_policy(root / 'run', device='cpu'), build_observation=build)
def receive():
    # 等待真实 Future，避免忙轮询与工作线程竞争 GIL，干扰小数组计时。
    worker.future.result()
    return worker.poll().result()
try:
    worker.load()
    info = receive()
    builder = ObservationBuilder(info, RealConfig())
    rows = tuple(dict(arm_qpos=data['joint_state'][i, :7], hand_qpos=data['joint_state'][i, 7:],
                      object_pointcloud=data['object_pointcloud'][i]) for i in range(info.n_obs_steps))
    for name, window in [('first', (rows[0],) * len(rows)), ('continuous', rows),
                         ('resumed', (rows[-1],) * len(rows))]:
        worker.warmup(window)
        receive()
        build_times.clear()
        elapsed = []
        for i in range(40):
            start = time.perf_counter()
            worker.predict(window, reset_episode=i == 0)
            receive()
            elapsed.append((time.perf_counter() - start) * 1000)
        print(name, 'build P50/P95 ms', np.percentile(build_times, [50, 95]),
              'worker P50/P95 ms', np.percentile(elapsed, [50, 95]))
finally:
    worker.close().result()
PY
```

这段测量将启动、连续和恢复时的窗口形状固定下来，几何仍在真实工作线程中执行；完整暂停/恢复的来源与调度行为由回归测试验证。Worker 往返包括任务提交及回收开销。要测实际实验，用相应 checkpoint、观测字段和来源快照替换这个小型合成训练样本。

## 收尾边界

`EpisodeRecorder.finish(save=..., reason=..., wait=False)` 登记首次结束意图并轮询一次；返回 False 表示线程仍在写入，True 表示收尾已结束，成败从 `result()` 读取。省略 `wait` 时同步等待。重复调用不改变首次保存/丢弃意图，也不延长等待预算；完成不会再返回发布路径，路径由结果统一提供。

Recorder 拥有 writer、队列、staging 与发布结果。Rollout 的 `snapshot_result()` 在 Recorder 活动时刷新快照，释放后保留最后结果。PolicyWorker 拥有在途预测及模型关闭 Future，Session 只在收尾完成后释放当前段。Teleop 保留自己的声音提示、丢弃和重新启动流程。

正常段结束先停止设备，再完成录制、回收本段预测并保存结果；退出/急停沿用先释放关键设备、再完成录制和模型关闭的顺序。writer 或模型仍未退出时保留句柄和 pending 结果。I/O 错误不会发布文件；首个错误与后续关闭错误都保留日志，程序错误或中断仍按既有优先级上报。
