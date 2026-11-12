# cordvip_geometry

DexMani Real 共用的物体资产与 RGB-D 几何核心，提供 CLI 和 Python API。支持图像分割、网格重建、实测尺度校准、位姿跟踪，供主项目生成手物联合观测。

跟踪要求每个任务对象都有米制 Mesh，使用 FoundationPose 估计位姿并通过语义深度和模型深度一致性门控。手点云由主项目的唯一 FK 路径生成。

## 安装与命令

在仓库根目录、已有兼容 Torch/CUDA 的 `real_robot` 环境中执行。默认只安装跟踪能力并下载跟踪模型：

```bash
conda activate real_robot
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download
# 单独准备 FoundationPose 权重后检查本地路径
python -m cordvip_geometry check
```

需要单图重建时，在**已完成默认安装的同一环境**中追加以下操作：

```bash
bash third_party/cordvip_geometry/scripts/install_perception.sh reconstruction
python -m cordvip_geometry download --models shape
```

`reconstruction` 是可选增量，只获取 Hunyuan3D 源码并安装重建新增依赖，不自动补做默认安装，也不重复 FoundationPose patch、nvdiffrast 安装或 CMake 构建。两个分支都保留基础 Torch、torchvision、NumPy、transformers 版本约束；缺失源码按脚本中的固定 revision 获取，已有源码目录与本地修改继续复用。脚本可从其他目录调用，使用 `--workspace /path/to/dexmani_real`，且该选项必须位于 `reconstruction` 之前。

`download` 默认只下载 Grounding DINO 和 SAM2.1 Small；`--models shape` 只下载 Hunyuan3D。权重默认保存在 `third_party/weights/`；FoundationPose 权重需单独放入 `third_party/FoundationPose/weights/`。推理只加载本地模型，相关命令可用 `--models-dir` 指定权重目录，GPU 通过 `CUDA_VISIBLE_DEVICES` 选择。

两个 requirements 文件都是对现有研究环境的增量，保留原有依赖版本，分类依据为脚本固定的上游源码导入链：

- [requirements-tracking.txt](requirements-tracking.txt)：SAM2 的 `build_sam → hydra`、`hieradet → iopath`；FoundationPose 的 `estimater → Utils/predict_*` 使用 kornia、warp、ruamel.yaml、transformations；nvdiffrast 的 Torch CUDA 扩展构建使用 ninja，FoundationPose `mycpp/CMakeLists.txt` 使用 Eigen/pybind11。CMake、C++、CUDA 编译器/头文件检查只在默认分支执行。
- [requirements-reconstruction.txt](requirements-reconstruction.txt)：Hunyuan `pipelines`/scheduler/MoE 使用 diffusers，`__init__ → postprocessors` 导入 pymeshlab，`pipelines → autoencoders → surface_extractors` 导入 scikit-image。当前模型配置使用 `FlowMatchEulerDiscreteScheduler`，不加载 `models/diffusion/transport`，因此不安装该路径的 torchdiffeq。FoundationPose 的 scikit-image 用于 BundleSDF 重建路径，不在当前跟踪导入链中。重建分支不执行源码扩展构建。
- 已有环境中的 Open3D/trimesh 仍是共用依赖：FoundationPose 和本项目 `simplify_mesh()` 都使用它们；`simplify_mesh()` 不使用 pymeshlab。

`check` 核对本地源码、解释器、权重和扩展文件路径；实际 CUDA 与模型推理需另行验证。

## 使用入口

`python -m cordvip_geometry object` 提供分割、候选选择、网格生成和米制资产准备。

完整参数通过各子命令的 `--help` 查看；Python 调用使用 `RGBDFrame` 和 `SceneTracker`，示例见下方接口文档。

- [实验脚本与任务流程](../../scripts/cordvip/README.md)：任务资产构建、模板冻结、Raw 标注与导出。
- [核心接口与结果格式](docs/usage.md)：Python 示例、初始化、坐标与有效性约定、离线结果查看。
- [HOI 训练与部署](../../docs/object_reconstruction.md#hoi-五阶段流程)：模板与 Checkpoint 契约、实测结论和验证限制。
