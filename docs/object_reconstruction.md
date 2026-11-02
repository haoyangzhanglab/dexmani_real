# 物体重建与 1024 点点云

流程：**物体照片 → Grounding DINO 检测 → SAM2 分割 → 确认掩膜 → Hunyuan3D-2.1 生成并简化网格 → 校准尺度 → 表面采样**。

输入为单张物体照片和至少一个实测尺寸。照片应包含完整物体，尽量减少遮挡、反光和背景干扰。单图不可见部分由模型推断，真实尺度必须另行提供。

每个物体独立保存参考图、点云和元数据，网格与模型权重仅保留在本地：

```text
assets/pointcloud_template/object/pnp_apple_banana/
  apple/                         # banana、bowl 使用相同结构
    reference.jpg                # 参考图、以下元数据和点云纳入 Git
    asset.json
    points.npy
    normals.npy
    pointcloud_1024.ply
    mesh/mesh_m.obj              # 仅本地，Git 忽略
  preview.png
third_party/weights/             # 仅本地，Git 忽略
```

克隆后可直接读取 NPY 或 PLY 点云。FoundationPose 位姿估计需要本地网格，应先在新目录完成重建和尺度校准，再将完整资产目录传给 `--asset`；缺少网格时会明确报错。

## 1. 环境与权重

以下命令均在仓库根目录执行，使用已安装重建依赖的环境；分割和重建需要 CUDA GPU。示例中间结果写入 `/tmp/cordvip/`；需要长期保存时自行指定输出路径。

```bash
conda activate real_robot
python -m pip install --no-deps -e third_party/cordvip_geometry
python -m cordvip_geometry download --models grounding sam shape
```

权重已保存在 `third_party/weights/`。文件齐全时上述下载命令直接复用，不联网下载：

| 路径（相对权重目录） | 用途 |
| --- | --- |
| `grounding-dino-base/` | 文本检测及配套配置 |
| `sam2.1_hiera_small.pt` | 物体掩膜 |
| `Hunyuan3D-2.1/hunyuan3d-dit-v2-1/` | 三维形状生成 |

通过 `--models-dir` 统一指定权重目录，保持表中的相对布局。源码固定放在 workspace 的 `third_party/`；GPU 通过 `CUDA_VISIBLE_DEVICES` 选择，推理使用可见的第 0 张卡。

## 2. 分割并生成网格

```bash
python scripts/cordvip/reconstruct_object.py \
  --image assets/pointcloud_template/object/pnp_apple_banana/apple/reference.jpg \
  --prompt 'apple.' \
  --output /tmp/cordvip/apple_build --preview
```

在分割窗口用 `n/p` 切换候选、Enter 确认。检查掩膜包含完整物体，排除桌面、手和阴影。确认后生成网格并打开预览；默认随机种子为 0、生成步数为 50、网格提取分辨率为 384。生成网格若超过 20,000 面，会在导出前简化到此上限；尺度、原点和朝向保持不变，不新增命令行参数。

- `segmentation/foreground_rgba.png`：实际送入重建模型的透明背景图。
- `generation/mesh_unscaled.obj`：已简化、尚未标定尺度的网格。
- `generation/generation_result.json`：模型和参数、简化前的 `source_faces`、`mesh_simplification`，以及导出网格的 `faces`/`vertices`。

所有输出目录必须是新目录，重复运行时更换目录名。此阶段不会保存最终点云；先检查简化后网格的外形、内腔和坐标轴。面数简化不包含孔洞修补等通用网格清理。

## 3. 标定尺度并采样

以下仅为轴向标定示例：**确认网格 x 轴的包围盒宽度对应实测 8 cm 后**，运行：

```bash
python -m cordvip_geometry object prepare \
  --mesh /tmp/cordvip/apple_build/generation/mesh_unscaled.obj \
  --size-axis x --size-meters 0.08 \
  --num-points 1024 --seed 0 \
  --output assets/pointcloud_template/object/apple_demo --preview
```

直径、斜向跨度等尺寸不一定等于某个坐标轴的包围盒长度。这时按相同定义测量网格尺寸，计算 `缩放系数 = 实测尺寸（米） / 网格尺寸`，用 `--scale-to-meters 缩放系数` 替代 `--size-axis` 和 `--size-meters`。已有米制网格使用 `--scale-to-meters 1`。

输出采用物体局部坐标系，保留网格朝向，不自动居中：

| 文件 | 内容 |
| --- | --- |
| `mesh/mesh_m.obj` | 米制网格，仅本地保存 |
| `points.npy` | `(1024, 3)`、float32 的 XYZ 坐标，单位米 |
| `normals.npy` | 对应的单位表面法向，来自采样三角面 |
| `asset.json` | 缩放系数、采样参数、坐标系和来源 |

prepare 从生成阶段导出的网格校准尺度后采样，不回到简化前的高面数结果。`source_mesh` 仅作来源记录，加载不依赖重建中间文件。直接读取 `points.npy`、`normals.npy` 不依赖网格。采样按三角面面积分配概率，在面内均匀取点；固定 seed 可复现。点云覆盖完整网格表面，不仅是照片可见部分。

新生成的网格已符合面数上限，跟踪时直接复用。已有资产若仍超过 20,000 面，跟踪初始化会在内存中简化一次，保留源资产、固定采样点及法向不变。

## 4. 检查并导出 PLY（可选）

```python
from pathlib import Path
import numpy as np
import open3d as o3d

asset = Path("assets/pointcloud_template/object/apple_demo")
points = np.load(asset / "points.npy", allow_pickle=False)
normals = np.load(asset / "normals.npy", allow_pickle=False)
assert points.shape == normals.shape == (1024, 3)
assert np.isfinite(points).all() and np.isfinite(normals).all()
assert np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.normals = o3d.utility.Vector3dVector(normals)
if not o3d.io.write_point_cloud(str(asset / "pointcloud_1024.ply"), cloud):
    raise OSError("PLY 点云写入失败")
```

## 无图形窗口时

分阶段执行，先查看 `candidates/candidates.jpg`，再显式选择实例。下面的 `0` 应替换为实际确认的候选编号：

```bash
python -m cordvip_geometry object segment \
  --image assets/pointcloud_template/object/pnp_apple_banana/apple/reference.jpg \
  --prompt 'apple.' --list-only --output /tmp/cordvip/apple_headless/candidates

python -m cordvip_geometry object select \
  --segmentation-dir /tmp/cordvip/apple_headless/candidates \
  --instance 0 --headless --output /tmp/cordvip/apple_headless/selection

python -m cordvip_geometry object generate \
  --segmentation-dir /tmp/cordvip/apple_headless/selection \
  --output /tmp/cordvip/apple_headless/generation
```

之后按第 3 步处理新的网格路径，并去掉 `--preview`。

现有苹果、香蕉、碗资产位于 [pnp_apple_banana](../assets/pointcloud_template/object/pnp_apple_banana)，分别按约 8 cm 直径、18 cm 两端直线距离、17 cm 碗口外径标定，每个物体 1024 点。具体尺度定义见各自的 `asset.json`；[预览图](../assets/pointcloud_template/object/pnp_apple_banana/preview.png)可用于快速检查。

多个物体需要合计 1024 点时，使用 [allocate_points.py](../scripts/cordvip/allocate_points.py)，默认生成 task 内的 `sampled_1024/`。按闭合实体体积分配，更新固定点、法向、PLY 与资产元数据；具体命令见[脚本说明](../scripts/cordvip/README.md#为多物体分配总计-1024-点)。
