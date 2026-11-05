# 物体资产与 Raw 离线几何

入口分别负责[单图建模](reconstruct_object.py)、[点数分配](allocate_points.py)和[逐行追踪](track_object.py)。算法及安装说明见 [cordvip_geometry](../../third_party/cordvip_geometry/README.md) 和[使用文档](../../third_party/cordvip_geometry/docs/usage.md)。以下命令在仓库根目录、已准备依赖的环境中执行；示例 Raw、网格和模型权重不随仓库分发，参考图和点云资产位于 [pnp_apple_banana](../../assets/pointcloud_template/object/pnp_apple_banana)。

## 准备米制资产

按[物体重建与 1024 点点云](../../docs/object_reconstruction.md)完成分割、重建与尺度校准。最终资产目录包含米制 `mesh/mesh_m.obj`、固定采样的 `points.npy`、`normals.npy` 和 `asset.json`。参考图 `reference.jpg`、元数据、NPY 与导出的 PLY 点云纳入 Git；`mesh/` 和 `third_party/weights/` 仅保留本地。克隆后可直接使用点云；FoundationPose 需要先在本地重建完整资产，再将该目录传给下方 `--asset`，缺失网格会明确报错。

## 为多物体分配总计 1024 点

```bash
python scripts/cordvip/allocate_points.py \
  assets/pointcloud_template/object/pnp_apple_banana \
  --output /tmp/cordvip/pnp_apple_banana_1024
```

脚本按各物体闭合 mesh 的实体体积分配点数；每个物体至少 1 点，剩余点按体积比例分配并用最大余数法补齐。碗按壁和底的实体体积计算。采样 seed 固定为 0，总计 1024 点，不包含手点云。

输出保留各物体的 mesh、标定和参考图，重建 `points.npy`、`normals.npy`、`pointcloud.ply`，并同步 `asset.json`；`allocation.json` 记录体积与配额。原资产保留，已有输出目录拒绝覆盖；省略 `--output` 时默认写入 `task/sampled_1024`。示例分配为 apple 432、banana 324、bowl 268。

追踪时将 scene 或 `SceneTracker.open` 的资产路径指向新目录中的各物体即可。所有对象位姿有效时，拼接 `model_points` 得到 `(1024, 3)`；跟踪失败的对象仍按原语义返回空点。

## 确认首帧并逐行追踪

单物体示例：

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 \
  --asset /tmp/cordvip/object_asset --prompt 'white basket.' \
  --with-hand --output /tmp/cordvip/scene_001
```

提供 `--prompt`，由 Grounding DINO 和 SAM2 生成冻结 Raw 首帧上的候选 mask，再人工选择实例；或用 `--initial-selection PATH` 加载已有首帧 mask。初始 mask 优先，提供时忽略 prompt，允许 prompt 缺省、`null` 或为空；明确指定的 mask 无效时直接报错，不回退到 prompt。没有初始 mask 时必须提供非空文本 prompt，无候选时明确报错。新选择保存到输出的 `initial_selections/`；`scene.json` 记录可复用的对象与选择路径。PNG/NPY mask 可直接使用，选择目录只需 `selection.json` 的 `mask_file` 指向目录内的 mask。两者均应用于当前指定首帧原始 RGB 网格，附带来源文件不参与校验；不自动缩放或猜测坐标网格，已有输入保持原样。

多物体由一个场景 JSON 声明，路径相对仓库根目录：

```json
{"objects": [
  {"id": "toy", "asset": "/tmp/cordvip/toy_asset",
   "initial_selection": "/tmp/cordvip/toy_first_frame"},
  {"id": "basket", "asset": "/tmp/cordvip/basket_asset",
   "initial_selection": "/tmp/cordvip/basket_first_frame"}
]}
```

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 --scene scene.json \
  --with-hand --headless --output /tmp/cordvip/scene_002
```

scene 只使用 `id`、`asset`、`initial_selection` 和 `prompt`。路径字段 `asset`、`initial_selection` 省略或为 `null` 时表示未提供，指定时须为非空文本。省略某个 `asset` 时，该对象只要求有效 RGB 与去畸变后非空 mask，首帧和后续帧均允许零深度或少量有效深度；输出 `OBSERVED_ONLY`、实际实测点与 `observed_valid`，无深度时点集为空，不估计 pose。全部省略 asset 时不要求 FoundationPose。有 asset 的对象保留位姿所需的采样点数、至少 4 个 `depth >= 0.001 m` 深度像素及模型匹配要求；交互深度预检查也只用于该对象，不影响混合场景中的无 asset 对象。无窗口运行要求每个实例提供当前首帧的有效 mask，可直接使用 mask 文件。

默认顺序处理全部 Raw 行，不按墙钟跳帧，也不要求 observation 时间为正或递增。`--max-frames N` 显式限制处理前缀，终端打印实际处理行数。输出目录必须尚不存在，可以是 `rollouts/derived/objects` 等独立结果目录；实际输入、Raw episode 内部和已有输出仍受保护。处理前将各资产的固定点/法向写入 `model_templates/<object_id>.npz` 一次，每行结果保存为 `frames/000000.npz` 等；`row_index` 是实际 Raw 行号，`timestamp_ns` 原样保留该行 `camera_ns`（包括 0），不自动生成 `source_id` 或推断 `clock`。逐帧保存有效/候选 pose 与实测数据，模型点/法向从模板按需重建。mask-only 不生成模板。查看结果是独立步骤：

```bash
python -m cordvip_geometry view /tmp/cordvip/scene_001
# 只看二维结果，或无窗口读取检查
python -m cordvip_geometry view /tmp/cordvip/scene_001 --no-3d
python -m cordvip_geometry view /tmp/cordvip/scene_001 --headless
```

查看器用 `n`/Space 前进一行，`v` 只更新 observed/model 点云，`q`/`Q`/Esc 或关窗退出；首次 model 切换加载输出内模板，本次查看复用。静止时仅轮询事件和渲染。`--no-3d` 不读取模板或解压点云，`view --headless` 检查 RGB 并打印已有来源说明，不加载 OpenCV/Open3D。查看使用固定调色板，无需原资产、Raw、模型环境或运行报告。

实测点来自当前 mask 与深度，模型点由模板与有效 pose 重建；深度支持参与 `POSE_LOST` 判定。正常 pose 丢失仍保留独立语义实测点，以及已计算的支持掩码/残差和门控前候选位姿；候选不等于有效位姿，无效 pose 和 mask-only 在 model 模式显示空点。模型或 renderer 的技术异常直接抛出并非零退出，保留已写入前缀。`--with-hand` 直接使用同一 Raw 行的 19 维实测关节生成 FK 手点，不使用 action，不证明物理同步，也不代表通过训练时序准入。

模板的 `points_object`、`normals_object` 保持资产载入后的顺序、float32 和米制 object frame；camera 模型点按原公式变换为 float32，法向矩乘保留自然 dtype。同一帧 NPZ 保留 `K`，手几何启用时还保留 camera 手点/法向、`T_base_camera`、link IDs 与关节状态。NumPy 重建示例见[使用文档](../../third_party/cordvip_geometry/docs/usage.md#输出与查看)。

默认 `--sam-variant small`；large 对照先下载对应权重，再指定 `--sam-variant large`。自训模型须同时指定 `--sam-config` 与 `--sam-checkpoint`。SAM 在独立解释器运行；模型回归、显存及吞吐需在实际 GPU 上验证。

环境路径检查使用 `python -m cordvip_geometry check`（可加 `--json`）。该命令只接收本地模型路径和型号选择，不接收检测阈值；检查不加载模型，不证明 CUDA 推理或 ABI 可用。
