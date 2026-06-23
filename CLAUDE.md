# dexmani_real

> **Python 环境**：`source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real`（`/home/zhy/anaconda3/envs/real/bin/python`）

## 项目简介

dexmani_real 是一个灵巧操作机器人遥操作与数据采集系统。

## 开发进度

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 文档修正（6 项） | ✅ |
| Phase 1 | P0 安全修复（4 项） | ✅ |
| Phase 2 | P1 运动质量（6 项） | ✅ |
| Phase 3 | P2 架构增强（4+4 项） | ✅ |
| Phase 4 | P3 工程优化（3 项） | ✅ |
| **Phase 5** | **集成收尾 & 生产就绪（6 项）** | ✅ **2026-06-23** |
| Phase 6 | 高级特性（可选） | 📋 待排期 |

### Phase 5 实施内容（2026-06-23）
- 5.1: CollectionLoop ↔ TeleopController 集成缝合
- 5.2: 多相机集成到控制回路（MultiCameraManager + HDF5 per-camera paths）
- 5.3a: auto_stop_on_quality_drop（连续低质量帧自动停止）
- 5.3b: Episode sidecar annotation JSON（stop_episode 时写入 metadata）
- 5.4: 仿真端到端验证（待手动运行）
- 5.5: 真机测试脚本完善（scripts/real/ 受权限保护，跳过）
- 5.6: 文档同步更新

## 目录结构

| 目录 | 说明 |
|------|------|
| `dexmani_real/` | 主 Python 包 |
| `dexmani_real/robot/` | 硬件驱动（xarm7/xhand 子包） |
| `dexmani_real/simulation/` | 物理仿真（SAPIEN） |
| `dexmani_real/teleop/` | VR 遥操作（core/vr/control 子包） |
| `dexmani_real/planning/` | 运动规划与 IK |
| `dexmani_real/recording/` | 数据录制（HDF5）+ CollectionLoop/Config/Annotator |
| `dexmani_real/sensor/` | 传感器驱动（RealSense + MultiCameraManager） |
| `dexmani_real/config/` | 全局配置（CameraCalib, PipelineConfig） |
| `dexmani_real/utils/` | 工具函数 |
| `assets/` | 3D 模型、URDF 等静态资源 |
| `scripts/real/` | 真机测试脚本 |
| `scripts/sim/` | 仿真测试脚本 |
| `configs/` | 运行时配置文件（cameras.json 等） |
| `docs/` | 文档 |
