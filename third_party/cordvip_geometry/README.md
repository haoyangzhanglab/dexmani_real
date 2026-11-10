# cordvip_geometry

DexMani Real 共用的物体资产与 RGB-D 几何核心，提供 CLI 和 Python API。支持图像分割、网格重建、实测尺度校准、位姿跟踪，供主项目生成手物联合观测。

跟踪要求每个任务对象都有米制 Mesh，使用 FoundationPose 估计位姿并通过语义深度和模型深度一致性门控。手点云由主项目的唯一 FK 路径生成。

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

`python -m cordvip_geometry object` 提供分割、候选选择、网格生成和米制资产准备。

完整参数通过各子命令的 `--help` 查看；Python 调用使用 `RGBDFrame` 和 `SceneTracker`，示例见下方接口文档。

- [实验脚本与任务流程](../../scripts/cordvip/README.md)：任务资产构建、模板冻结、Raw 标注与导出。
- [核心接口与结果格式](docs/usage.md)：Python 示例、初始化、坐标与有效性约定、离线结果查看。
- [HOI 训练与部署](../../docs/object_reconstruction.md#hoi-五阶段流程)：模板与 Checkpoint 契约、实测结论和验证限制。
