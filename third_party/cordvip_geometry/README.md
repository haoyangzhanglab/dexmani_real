# cordvip_geometry

DexMani Real 共用的物体资产与 RGB-D 几何核心，提供 CLI 和 Python API。支持图像分割、网格重建、实测尺度校准、位姿跟踪及物体／手部点云分析。

有米制 Mesh 的对象使用 FoundationPose 估计位姿；无资产对象输出分割与实测点云。两类对象可在同一场景中处理。

## 安装与命令

在仓库根目录、已有兼容 Torch/CUDA 的 `real_robot` 环境中执行：

```bash
conda activate real_robot
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download
# 单独准备 FoundationPose 权重后检查本地路径
python -m cordvip_geometry check
```

`download` 默认下载 Grounding DINO、SAM2.1 Small 和 Hunyuan3D 权重至 `third_party/weights/`，可用 `--models` 按需选择；FoundationPose 权重需单独放入 `third_party/FoundationPose/weights/`。推理只加载本地模型，相关命令可用 `--models-dir` 指定权重目录，GPU 通过 `CUDA_VISIBLE_DEVICES` 选择。

`check` 核对本地源码、解释器、权重和扩展文件路径；实际 CUDA 与模型推理需另行验证。

## 使用入口

| 入口 | 用途 |
| --- | --- |
| `python -m cordvip_geometry object` | 分割、候选选择、网格生成和米制资产准备。 |
| `python -m cordvip_geometry track` | 按 Raw 行顺序跟踪并保存场景结果。 |
| `python -m cordvip_geometry view` | 查看已保存的结果，无需重新推理。 |

完整参数通过各子命令的 `--help` 查看；Python 调用使用 `RGBDFrame`、`SceneTracker` 和 `track_raw`，示例见下方接口文档。

- [实验脚本与任务流程](../../scripts/cordvip/README.md)：任务资产构建、模板冻结、Raw 标注与导出。
- [核心接口与结果格式](docs/usage.md)：Python 示例、初始化、坐标与有效性约定、离线结果查看。
- [HOI 训练与部署](../../docs/object_reconstruction.md#hoi-五阶段流程)：模板与 Checkpoint 契约、实测结论和验证限制。
