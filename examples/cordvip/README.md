# 物体重建与手—物点云示例

这里保留两个薄入口；模型、几何和可视化实现在 [cordvip_geometry](../../third_party/cordvip_geometry/README.md)。建议先跑下面的 toy 回放，再看库的 Python 接口，最后按[完整文档](../../third_party/cordvip_geometry/docs/usage.md)准备新物体。

在仓库根目录激活环境：

```bash
conda activate real_robot
```

## 1. RGB 图像 → mesh 与点云

Grounded-SAM2 分割目标，确认候选后用 Hunyuan3D-2.1 生成形状，并采样点云：

```bash
python examples/cordvip/reconstruct_object.py \
  --image /path/to/object.jpg --text 'white basket.' \
  --output-dir reports/cordvip_demo/object_build --preview
```

`generation/` 保存 `mesh_unscaled.obj`、`points_unscaled.npy`、`normals.npy` 和来源记录。默认采样 1024 点，可用 `--num-points` 修改；`--preview` 显示 mesh、点云和坐标轴。单图生成结果没有真实尺度。

检查生成模型的轴后，用实测尺寸准备追踪资产。例如，**仅当模型 z 轴对应实测 12 cm 高度**：

```bash
python -m cordvip_geometry prepare \
  --mesh reports/cordvip_demo/object_build/generation/mesh_unscaled.obj \
  --size-axis z --size-meters 0.12 \
  --output-dir reports/cordvip_demo/object_asset --preview
```

已知尺度时也可直接为 `reconstruct_object.py` 提供 `--size-axis/--size-meters` 或 `--scale-to-meters`，生成后的米制资产放在 `asset/`。已有 mesh 可直接 `prepare`，跳过单图生成。

## 2. RGB-D 视频流 → 位姿与实时点云

### Toy + 篮子：连续分割、双位姿与手点云

已准备的两个资产、首帧选择由一个 JSON 指定；SAM2.1 small 共享编码两物体，每个物体独立维护 FoundationPose++ 2D 引导状态，网络权重共享。直接在窗口查看已有录像：

```bash
python examples/cordvip/track_object.py \
  --scene reports/cordvip_multiobject/toy_basket.json \
  --episode episodes/pick_place_toy/episode_20260827_165923 \
  --with-hand --replay-fps 16 \
  --output reports/cordvip_demo/toy_basket
```

`v` 切换实测/模型图层，`s` 截图，`q` 退出。Toy 为青色，篮子为紫色，FK 手为橙色。窗口显示每个物体的实测点数与位姿状态。首次初始化后，默认只处理最新帧，处理不及时会明确跳帧；离线加 `--all-frames` 则逐帧运行，不按墙钟节拍等待。加 `--headless --all-frames --save-frames --save-video` 可完整离线导出，`--max-frames N` 可限制验收长度。

场景文件格式如下，路径相对于 workspace：

```json
{
  "objects": [
    {"id": "toy", "role": "object", "asset": "reports/cordvip_toy/asset",
     "initial_selection": "reports/cordvip_toy/first_frame_selected"},
    {"id": "basket", "role": "reference", "asset": "reports/cordvip_basket/tracking_asset",
     "initial_selection": "reports/cordvip_basket/first_frame_selected"}
  ]
}
```

**`asset` 可省略**：该实例只输出连续语义分割和当前深度实测点，不加载其位姿模型，也不伪造 6D pose。所有实例都不提供 asset 时，不要求 FoundationPose 或生成模型。初始化选择只用于完全相同的首帧；新录像/相机删去 `initial_selection`，可填写 `prompt`（如 `white basket.`），在窗口逐个确认候选，或手动画多边形。新选择保存在输出的 `initial_selections/`，同一源下次可直接复用 `replay_scene.json`；相机新帧禁止套用历史 mask。

语义实测点独立于 mesh，模型支持子集与残差另外保存。单个物体 pose 失效不会抹掉另一个物体或当前语义深度点；该 pose 保持失效，重新运行并显式选择才能注册。分割仍可能覆盖手边缘或篮孔后面的桌面，几何一致性也不等于物理精度。生成模型和当前实测点不做跨帧融合。

`real_robot` 是示例环境名；父进程按实际能力检查依赖，SAM 子解释器须满足固定 Python/依赖版本，可通过 `--sam-python` 指定。新工作站按[安装说明](../../third_party/cordvip_geometry/docs/usage.md)准备环境，再运行 `python -m cordvip_geometry download --models grounding sam`。默认 SAM2.1 small，不编译；Grounding DINO 只负责首帧文本候选，开始跟踪后释放。已有确认 mask 不要求检测权重。large 对照须同时指定匹配的 `--sam-config/--sam-checkpoint`。当前位姿仅采用 2D guidance，没有 Kalman。本地示例资产和 Raw 不随仓库分发。

### 单物体位姿模式

已有 toy 资产和首帧分割可直接复用，Grounded-SAM2 从该 mask 开始连续传播。单物体与多物体共用 SceneTracker，单物体 ID 为 object。下面只读 Raw，在窗口显示物体模型点、实测目标点与同一行实测关节生成的 XHand 点：

```bash
python examples/cordvip/track_object.py \
  --episode episodes/pick_place_toy/episode_20260827_165923 \
  --asset reports/cordvip_toy/asset \
  --initial-selection reports/cordvip_toy/first_frame_selected \
  --with-hand --replay-fps 16 --save-frames --save-video \
  --output reports/cordvip_demo/toy_interaction
```

输出目录必须尚不存在。加 `--headless` 可无窗口运行；省略两个 `--save-*` 可减少落盘开销。离线源先停在首帧，注册通过后按指定速率供给最新帧，结束自动退出；处理不及时会跳帧，保存结果不补齐。`--initial-selection` 仅接受已人工确认且 RGB-D、内参、身份、时间与首帧完全匹配的选择，不能套用到新相机帧。

`--with-hand` 当前仅支持 `--episode`；`--hand-points` 默认 1024，`--experiment-config` 可指定当前手安装配置。输出包含 robot-base 下的物体/手点、法向、link IDs 和 19 维实测关节。Toy 的历史 Raw 只有名义时间：可看交互几何，不能证明手/相机同步。**几何有效不等于策略输入的时序已合格**，未知时间差为 NaN。

实时相机使用米制资产、RGB-D 和内参进行 FoundationPose++ 2D 引导追踪。**以下命令会连接 RealSense，目前输出物体点云，不读取机器人关节**：

```bash
python examples/cordvip/track_object.py \
  --asset reports/cordvip_demo/object_asset --object-prompt 'white basket.' \
  --serial CAMERA_SERIAL --output reports/cordvip_demo/live_run
```

模型加载后冻结首帧并分割目标，选择候选后 Enter 确认，`s` 保存截图，`q` 退出。追踪窗口显示已处理源 RGB 与同帧叠加图。实测点来自当前帧深度反投影，不做跨帧表面融合；当前入口需要深度，不能直接输入普通 RGB 视频。

之后接入遥操作/策略时，应复用其已有 RGB-D 与实测关节观测，调用库的内存接口；不要另开一套机器人/相机连接。手点来自固定 URDF 表面采样与 FK，物体完整点来自生成模型与追踪位姿，均不等同于实测接触。默认 1024 手点 + 1024 物体点及 XArm7 + XHand 的 19 关节是本地表示，训练适配放在 `dexmani_policy`。

两个脚本均支持 `--help`；`track_object.py --check --object-prompt 'white basket.'` 只检查依赖，不连接设备。拍照、分阶段生成和回放等辅助操作使用 `python -m cordvip_geometry {capture,object,replay,...}`。

`python -m cordvip_geometry replay` 提供无模型输入预览。运行和诊断记录保存在 `reports/`；真实相机与实时手—物接口须单独验收。
