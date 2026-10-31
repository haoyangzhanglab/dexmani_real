# cordvip_geometry

用于个人研究的离线物体资产准备与 Raw 几何处理。Grounding DINO 提供首帧文本候选，人工确认后由 SAM2.1 连续传播；可选 FoundationPose++ 使用 2D bbox guidance 细化位姿，不含 Kalman。Hunyuan3D-2.1 单图生成的 mesh 需要显式校准米制尺度。

常用入口是 [reconstruct_object.py](../../scripts/cordvip/reconstruct_object.py) 和 [track_object.py](../../scripts/cordvip/track_object.py)。先看[短示例](../../scripts/cordvip/README.md)；安装、数据与输出细节见[使用文档](docs/usage.md)。

```bash
conda activate real_robot
# 新工作站先准备基础环境；安装脚本不下载模型、不连接设备。
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download --models grounding sam
python -m cordvip_geometry check
```

`real_robot` 是示例环境名。SAM 可通过 `--sam-python` 使用独立解释器，RPC 隔离保留。默认型号 small；官方型号由 `--sam-variant` 同时选定配置与本地 checkpoint，自训模型显式提供完整配对。显式 `check` 只核对解释器、源码和权重路径，可加 `--json`；不证明 GPU 推理、扩展 ABI 或长序列内存可用。

主流程为 Raw 首帧确认 → 单个 RawReader 顺序处理每行 → 保存每行结果 → 独立查看。全部对象不带 asset 时只做 mask 与实测点，不要求 FoundationPose。带 asset 时，深度几何支持参与 pose-valid 与 `POSE_LOST`，并非纯诊断；observed/model 点及法向仍分别保存。手点来自同行实测关节和数据对应的相机标定，同行关联不证明物理同步。

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 --scene scene.json \
  --with-hand --headless --output outputs/scene_001
python -m cordvip_geometry view outputs/scene_001
```

追踪默认处理全部行，只有 `--max-frames` 显式限制前缀。`frames/` 必存，`scene.json` 保存一次对象/role/asset/selection，`run.json`、`models.json`、`result.json` 分别记录配置、已就绪模型来源与最终结果。逐帧事实只写入 NPZ，包括失败原因、门控前候选位姿和已计算的支持掩码、深度残差；不再生成 `records.jsonl` 或默认报告统计。输出使用新目录，Raw 始终只读。首帧选择仍需与源数据及相机几何精确对应。

建模的分割和 prepare 直接调用普通函数；只有 Hunyuan 生成使用一次子进程，退出释放 GPU。生成只保存未定尺度 mesh；最终米制资产按原 seed/点数采样一次。结果查看器只按行前进及切换 observed/model，不另建实时服务。

本工具不提供实时相机追踪、latest/心跳发布或外部 Frame 宿主，也未接入主包控制、Raw writer、Canonical 导出或策略部署。上游源码与模型继续遵循各自的许可证及模型条款，详见[安装与权重](docs/usage.md#安装与权重)。
