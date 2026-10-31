# 物体资产与 Raw 离线几何

两个入口分别负责[单图建模](reconstruct_object.py)和[逐行追踪](track_object.py)。算法及安装说明见 [cordvip_geometry](../../third_party/cordvip_geometry/README.md) 和[使用文档](../../third_party/cordvip_geometry/docs/usage.md)。以下命令在仓库根目录、已准备依赖的环境中执行；示例 Raw、资产和模型不随仓库分发。

## 准备米制资产

先分割并人工确认目标，再生成未定尺度 mesh：

```bash
python scripts/cordvip/reconstruct_object.py \
  --image /path/to/object.jpg --text 'white basket.' \
  --output-dir outputs/object_build --preview
```

`generation/` 只保存 `mesh_unscaled.obj` 与请求、结果记录。预览点临时生成。单图没有真实尺度；检查模型轴后，若 z 轴确实对应实测 12 cm 高度：

```bash
python -m cordvip_geometry prepare \
  --mesh outputs/object_build/generation/mesh_unscaled.obj \
  --size-axis z --size-meters 0.12 \
  --output-dir outputs/object_asset --preview
```

最终 `mesh_m.obj`、`points.npy`、`normals.npy` 和 `asset.json` 使用米制 object frame，保持原点与朝向。固定点和法向只在最终尺度按 seed 生成一次。也可向建模入口直接提供尺寸或 `--scale-to-meters`，米制资产会保存到 `asset/`；已有 CAD/扫描 mesh 可直接 prepare。

## 确认首帧并逐行追踪

单物体示例：

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 \
  --asset outputs/object_asset --object-prompt 'white basket.' \
  --with-hand --output outputs/scene_001
```

在冻结的 Raw 首帧上确认文本候选，或省略 prompt 手动画 mask。选择保存到输出的 `initial_selections/`；`scene.json` 记录可复用的对象与选择路径。再次运行可指定 `--initial-selection DIR`。复用仍严格核对首帧 RGB-D、内参、畸变、来源时间和行身份，不把历史 mask 套到另一段录像。

多物体由一个场景 JSON 声明，路径相对仓库根目录：

```json
{"objects": [
  {"id": "toy", "role": "object", "asset": "outputs/toy_asset",
   "initial_selection": "outputs/toy_first_frame"},
  {"id": "basket", "role": "reference", "asset": "outputs/basket_asset",
   "initial_selection": "outputs/basket_first_frame"}
]}
```

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 --scene scene.json \
  --with-hand --headless --output outputs/scene_002
```

`role` 是一次保存的标签，不改变算法。省略某个 `asset` 时，该对象只输出连续 mask 与实测深度点，不估计 pose；全部省略时不要求 FoundationPose。无窗口运行要求每个实例都有当前首帧的确认选择，不会自动接受最高分候选。

默认顺序处理全部 Raw 行，不按墙钟跳帧。`--max-frames N` 显式限制处理前缀，结果标为 partial。输出目录必须尚不存在；每行结果只保存为 `frames/000000.npz` 等，源 row ID、时间和时钟保留；不再生成 `records.jsonl` 或默认分析报告。查看结果是独立步骤：

```bash
python -m cordvip_geometry view outputs/scene_001
# 只看二维结果，或无窗口读取检查
python -m cordvip_geometry view outputs/scene_001 --no-3d
python -m cordvip_geometry view outputs/scene_001 --headless
```

查看器用 `n`/Space 前进一行，`v` 切换 observed/model，`q` 退出。追踪不再提供实时相机、latest 持续发布或通用 Frame 宿主；旧 notebook 应改为读取逐行结果。

实测点来自当前 mask 与深度，模型点来自固定资产与有效 pose；深度支持参与 `POSE_LOST` 判定。正常 pose 丢失仍保留独立语义实测点，以及已计算的支持掩码/残差和门控前候选位姿；候选不等于有效位姿。模型或 renderer 的技术异常结束任务为 failed，并保留已写入前缀。`--with-hand` 直接使用同一 Raw 行的 19 维实测关节生成 FK 手点，不使用 action，不证明物理同步，也不代表通过训练时序准入。法向继续保存。

默认 `--sam-variant small`；large 对照先下载对应权重，再指定 `--sam-variant large`。自训模型须同时指定 `--sam-config` 与 `--sam-checkpoint`。SAM RPC 隔离继续保留；模型回归、显存及吞吐需在实际 GPU 上验证。

环境路径检查使用 `python -m cordvip_geometry check`（可加 `--json`）；追踪不再接受 `--check`。该命令只接收本地模型路径和型号选择，不接收检测阈值；检查不加载模型，不证明 CUDA 推理或 ABI 可用。
