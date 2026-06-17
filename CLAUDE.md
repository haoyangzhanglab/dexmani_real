# dexmani_real

> **Python 环境**：`source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real`（`/home/zhy/anaconda3/envs/real/bin/python`）

## 项目简介

dexmani_real 是一个灵巧操作机器人遥操作与数据采集系统。

## 目录结构

| 目录 | 说明 |
|------|------|
| `dexmani_real/` | 主 Python 包 |
| `dexmani_real/robot/` | 硬件驱动（xarm7/xhand 子包） |
| `dexmani_real/simulation/` | 物理仿真（SAPIEN） |
| `dexmani_real/teleop/` | VR 遥操作（core/vr/control 子包） |
| `dexmani_real/planning/` | 运动规划与 IK |
| `dexmani_real/recording/` | 数据录制（HDF5） |
| `dexmani_real/sensor/` | 传感器驱动（RealSense） |
| `dexmani_real/config/` | 全局配置（CameraCalib, PipelineConfig） |
| `dexmani_real/utils/` | 工具函数 |
| `assets/` | 3D 模型、URDF 等静态资源 |
| `scripts/real/` | 真机测试脚本 |
| `scripts/sim/` | 仿真测试脚本 |
| `configs/` | 运行时配置文件（cameras.json 等） |
| `docs/` | 文档 |
