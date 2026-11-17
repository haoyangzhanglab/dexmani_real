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

输出路径必须尚不存在。样本预览应报告 397 帧、0 次拒绝；导出应报告 1 个来源、1 个连续段、397 帧。`--no-view` 不需要 Rerun，纯预览和导出不需要 Policy。上述 pip 版本支持下方 CPU 回归所需的构建约束。

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
  tests/test_rollout_hoi_recovery.py tests/test_experiment_pipeline.py \
  tests/test_profile.py tests/test_pointcloud.py tests/test_benchmark_episode_visualizer.py
```

`requirements-cpu.txt` 明确使用 `+cpu` wheels，不安装 CUDA Torch。配套 Policy 直接安装固定 Git 提交，不读取旁边可能正在修改的工作区。跨仓流水线在临时目录生成小型 Raw、Zarr、训练/恢复 checkpoint，再执行真实 CPU 推理，无需下载预训练权重。

Ruckig 0.17.3 的源码包使用旧的 `cmake.targets` 构建配置，构建约束将 scikit-build-core 限制在 0.10 之前；约束只作用于隔离构建，不改变运行时依赖。

性能画像测试使用 Fake 设备和小型临时 Episode，验证采样不改变控制与录制结果；benchmark 将真实 Rerun SDK 输出到空设备，不启动 Viewer。合成数据只用于功能回归，不作为真实实验性能结论。

[CPU workflow](../.github/workflows/cpu.yml) 在 PR、main 更新和手动触发时运行：先检查最小安装与 CLI，再安装回归依赖并执行上面的测试。它覆盖本仓库的 CPU 离线流程，不包含真机、GPU 或 `third_party/` 的独立环境测试。
