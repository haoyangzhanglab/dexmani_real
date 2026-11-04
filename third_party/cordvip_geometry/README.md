# cordvip_geometry

物体资产准备、Raw 逐行处理与宿主 RGB-D 跟踪共用同一个几何核心。Grounding DINO 提供首帧候选，人工确认后由 SAM2.1 传播；有米制 mesh 的对象使用 FoundationPose++ 估计位姿，无资产的对象输出分割与实测点云。

## 安装与命令

在 workspace 根目录、已有兼容 Torch/CUDA 的环境中执行：

```bash
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download --models grounding sam
python -m cordvip_geometry check
```

源码固定放在 `third_party/Grounded-SAM-2`、`FoundationPose`、`Hunyuan3D-2.1`；权重默认位于 `third_party/weights`，可用 `--models-dir` 指定。SAM 默认 small，可指定 large 或完整自训配置/权重配对；`--sam-python` 选择独立解释器。模型只从本地加载，GPU 通过 `CUDA_VISIBLE_DEVICES` 选择。

常用脚本是[物体重建](../../scripts/cordvip/reconstruct_object.py)、[体积点数分配](../../scripts/cordvip/allocate_points.py)和[Raw 追踪](../../scripts/cordvip/track_object.py)。命令与数据示例见[脚本说明](../../scripts/cordvip/README.md)，细节见[使用文档](docs/usage.md)。

## Python 调用

```python
from dexmani_real.data.raw import RawReader
from cordvip_geometry import SceneTracker, track_raw

with (
    RawReader(episode) as reader,
    SceneTracker.open(assets, initial_selections=selections) as tracker,
):
    for result in track_raw(reader, tracker, with_hand=True):
        points_camera = tracker.model_points(result, "object")
```

`assets` 为对象名到资产目录的映射，`selections` 为对象名到首帧 PNG/NPY mask 文件或选择目录的映射；目录仅需 `selection.json` 的 `mask_file`，可选 `source_rgbd.npz` 存在时核对首帧实际数据。宿主持有相机时，用 `RGBDFrame.from_camera(camera, row.camera)` 构造帧，再调用 `tracker.step(frame)`。也可省略 `initial_selections`，改在首帧传入原 RGB 网格的 `initial_masks`；两种初始化输入不能混用。实时重复样本复用结果，Raw 每行均独立处理。点云按需重建为米制 `float32 (N, 3)`，可用 `T_target_camera` 变换到基座坐标系。

`SceneTracker.open` 一次加载资产与模型，并负责资源收尾。API 返回内存结果；SAM 日志默认输出到宿主 stderr，指定 `log_dir` 可保存日志。CLI 使用同一工厂，并写入可独立查看的结果目录。接口和返回语义见 [Python API](docs/usage.md#python-api)。

## 几何约定

- 深度为 color-camera Z，单位米；0 表示缺测。物体资产保留原点和朝向，固定点与法向按原始顺序使用。
- 生成 mesh 在导出前简化到最多 20,000 面；已有高面数资产在跟踪初始化时简化一次，位姿与模型深度比较共用该网格。
- 模型深度只查询当前实测像素。深度支持不足时 pose 失效，独立实测点仍保留；无效 pose 的模型点为空。
- 点数分配脚本按闭合实体体积采样，总计 1024 点；当前示例在 `pnp_apple_banana/sampled_1024`，分配为 432 / 324 / 268。

上游源码、模型授权和安装前提见[安装与权重](docs/usage.md#安装与权重)。
