# L515 边缘飞线(edge flying-lines)深度滤波方法工程研究报告

面向 `dexmani_real` 当前管线(xArm7 + XHand,静态桌面场景,L515 640×480 @ 0.3–2.5 m,`remove_flying_pixels_at_edges` + 可选 3D radius-outlier)

---

## 1. TL;DR + 推荐方案

**核心结论(一句话):** L515 的边缘飞线不是随机噪声,而是 ToF 混合像素(mixed-pixel)在深度不连续处产生的、从前景到背景的**连续、稠密、高置信、低噪声的 3D 斜坡(ramp)**。因此只有**删除类(DELETE)**方法能干净去除它;任何**平滑类(SMOOTH)**滤波(SDK `spatial_filter`、`bilateral`、`hole_filling`)都会在斜坡中段"看不到边"而跨边平均,把前景和背景**焊接**成实心桥,使飞线更严重 —— 这正是你们观察到"spatial 让飞线更糟"的原因。

**给本仓库的优先推荐(按性价比排序):**

| 优先级 | 动作 | 落点 | 预期收益 |
|---|---|---|---|
| **P0** | 在源头加 L515 device-level 配置:`visual_preset=SHORT_RANGE(5)`、`noise_filtering` 从中到高 sweep、`confidence_threshold=1–2`、`min_distance` 读设备后按最近物体设定 | `test_pointcloud_process.py:61`(`depth_sensor` 之后) | 从固件层减少飞线**生成量**,零主机开销;后续删除少吃真实面 |
| **P0** | 修正当前 `remove_flying_pixels_at_edges` 的 **k=1 采样污染**:两侧采样从 ramp 外开始(加 `gap` 死区参数,`k_start=2–3`),并把 `margin` 改为**深度自适应** `margin(z)=α·z` | `pointcloud_utils.py:454`(kernel 采样起点)、`:543-552`(删除判据) | 直接抬高你们文档里记录的"no-eat 天花板",同一 `margin` 不再近处过删/远处漏删 |
| **P1** | 新增一个**深度自适应角距删除**(organized angular test,Chen et al. 2022)作为主删除器,与现有 ramp-detector 二选一或串联 | `pointcloud_utils.py` 新增函数 + `test_...py:169-175` 替换/串联 | 一套阈值覆盖 0.3–2.5 m 全程,几乎免调参 |
| **P1** | 用对齐的 **IR 边缘掩膜**与深度梯度做**AND 门控**(两个独立线索同时命中才删) | 在 §3 unproject 之前,读 `RS2_STREAM_INFRARED` 对齐到 depth | Intel 官方对 L515 边缘 tails 的建议路线(#7117),显著降低吃真实面风险 |
| **P2** | 重新启用 3D `remove_radius_outlier` 作为**收尾**清扫残余孤立飞点(不是主力),并把 `nb_points=40, radius=0.01` 放宽 | `test_...py:267`(`USE_OUTLIER_REMOVAL`) | 只清 2D 删除后的零散残点;连成线的稠密 ramp 它删不掉 |

**一句话行动:** 保留你们已经调到"no-eat"的 ramp-detector,先从**设备源头 + 修 k=1 采样 + 深度自适应阈值**三处低风险改动入手,再决定是否引入角距删除和 IR 门控。**永远不要**用 `hole_filling` 或宽 `smooth_delta` 的 `spatial` 去"去飞线"。

---

## 2. 根因:L515 边缘飞线的物理机制

L515 是**扫描式固态 LiDAR**(MEMS 反射镜 + 单光电二极管,每点一次 ToF 测量,datasheet Rev002)。在一个深度不连续边界处:

1. 激光足迹(footprint)+ 像素瞬时视场(IFOV)会**同时跨过**前景轮廓线和其后的背景。
2. 返回波形因此包含**两个不同飞行时间的回波**。器件只上报**一个深度值**,它是两个回波按幅度/相位加权的混合(Mask-ToF 相位混合模型:`ĥφ = atan2(a·sin φ + a'·sin φ', a·cos φ + a'·cos φ')`),落在**前景与背景之间**。
3. 当足迹扫过边界时,混合权重在若干像素上**连续变化**,于是这些错误深度在 3D 中形成一条从表面伸向背景的**平滑斜坡** —— 也就是"飞线"。

两个关键性质决定了正确的滤波类别:

- **局部平滑、逐像素跳变小**:斜坡中段相邻像素的深度差很小,所以任何"靠局部深度差判边"的边缘保持型平滑器(bilateral / joint-bilateral / RealSense `spatial`)在斜坡中段**根本检测不到边**,于是跨斜坡平均,把前景背景焊成实心桥。
- **高置信、低噪声**:L515 的 confidence map 在这些假边上给出**高置信**(librealsense #7117 已确认),所以 `confidence_threshold=3` 也删不掉它们;而且斜坡是稳定的,对**静态桌面**做时间滤波(temporal)也会把它平均保留下来。

**推论:** 飞线是一条**稠密连通的斜坡**,不是孤立散点。因此:
- 平滑器 → 会加重(焊桥);
- 稀疏离群点删除(SOR/ROR/DBSCAN)→ 对连通稠密斜坡基本无效,只能当收尾;
- **正解 = 删除/分割不连续带**:方向导数/梯度、窗口深度跨度、视线-法向夹角(mixed-pixel)、IR/RGB 边缘掩膜 —— 把飞线像素置 0/NaN,而不是把它们混进平均。

---

## 3. 为什么平滑类滤波会加重飞线,删除类才是正解

这是整份报告的**判据核心(the crux)**,也直接解释你们"开 spatial 反而更糟"的现象。

**DELETE vs SMOOTH 的机制差异:**

- **SMOOTH(平滑/插值)** —— `spatial_filter`、`temporal_filter`、`hole_filling_filter`、bilateral、guided filter、point-cloud bilateral。它们**从不置无效**,只是用邻域加权平均替换深度值。
  - `spatial_filter` 是 1D domain-transform 边缘保持平滑器(Gastal & Oliveira)。它的 `smooth_delta` 只是"跨大台阶时停止平滑"的边缘阈,但飞线斜坡中段的相邻差**小于** `smooth_delta`,于是被当作"同一表面"平均掉,把前景拉向背景 —— **这就是你们看到飞线变糟的直接原因**。
  - `hole_filling_filter` 默认 mode 1 = `farest_from_around`,用**最远(背景)**邻居填洞,等于在物体和墙之间**主动制造**一条斜坡。它是本任务里**最有害**的一个;如果你先用删除打了洞,它会把洞重新填成桥。
  - `temporal_filter` 对静态斜坡无能为力(稳定的斜坡被平均保留);高 persistence 还会**凭空持续**只出现在少数帧的飞点。

- **DELETE(删除)** —— 方向梯度、窗口跨度/SAD、法向-视线夹角、形态学腐蚀、连通域删除、IR 边缘掩膜。它们把飞线像素**置无效(0/NaN)**,**不做平均**,所以**不可能焊桥**。

**和你们已有观察的对账:**
> 你们观察到"spatial 让飞线更糟" —— 这不是参数没调好,而是**方法类别选错了**。`spatial` 属于 SMOOTH,对高置信、局部平滑的斜坡天生会跨边平均。即使把 `smooth_delta` 调到最小、`holes_fill=0`,它最多**不加重**,也**不可能删除**斜坡中段像素。你们现有的 `remove_flying_pixels_at_edges` 走的正是 DELETE 路线(方向梯度 + 两侧差异门 + "圆心落在两面之间则置 0"),方向是对的。

**代价的诚实说明:** 任何 DELETE 方法都会在每条真实边缘吃掉 1–2 px 的真实面(真实物体轮廓本身也是低密度、掠射、高梯度,与飞线**签名相同**)。Intel 官方的立场也是"宁可少留有效像素,也不要用捏造深度去填补"。在**面向相机的桌面**场景,真实面大多正对相机,这点附带损失很小 —— 这正是删除在此处是正确选择的原因。

---

## 4. 方法全景与对比

四大家族横向对比。**删除还是平滑**一列是选型的第一决策变量。

### 4.1 设备级 / SDK 源头

| 方法 | 机制 | 删除/平滑 | 对边缘飞线效果 | 吃真实面风险 | 计算成本 | 对 L515 适用性 |
|---|---|---|---|---|---|---|
| `visual_preset=SHORT_RANGE(5)` + `laser_power` 微调 | 降激光功率/增益,减 IR 饱和与 blooming | 预防(减少生成) | 中(减少来源) | 低(功率过低会丢暗面) | **必配**;0.3–2.5 m 桌面用 Short Range(min-Z 25 cm),勿用 Max Range(min-Z 50 cm) |
| `noise_filtering`(L515 独有,范围 **0–6**) | 固件内边缘/背景噪声抑制 | DELETE(推测,固件内部未公开是否删/平滑,**待经验验证**) | 中-高(最对症的设备旋钮) | 中(过高会削薄真实边缘) | 直接适用;从中值向高 sweep,盯住物体轮廓别消失 |
| `confidence_threshold`(1/2/3) | 丢低置信像素 | DELETE | **对边缘飞线基本无效**(#7117:假边是高置信) | 中(高阈会在暗/镜面真实面打洞) | 设 1–2 做通用清理,**别指望删飞线** |
| `threshold_filter`(`min_distance`/`max_distance`) | 范围门,越界置 0 | DELETE | 只删越界的远背景 tails,带内斜坡删不掉 | 极低 | 第一级粗筛,设到工作体积 |
| `min_distance`(设备近端门) | 固件内删近端 blooming | DELETE | 只近端 | 低 | 设到最近物体之下 |
| `hole_filling_filter` | 用邻居填洞(默认 mode 1=背景) | **SMOOTH/插值** | **有害,制造飞线** | 极高 | **禁用** |
| `spatial_filter` | domain-transform 边缘保持平滑 | **SMOOTH** | **删不掉,易加重** | —(跨边焊桥) | 仅删除后对平坦面做轻度降噪,`holes_fill=0` |
| `temporal_filter` | 逐像素跨帧 EMA | **SMOOTH(时间)** | 对静态稳定斜坡无效 | 高 persistence 会凭空造飞点 | 静态可用低 persistence 压闪烁飞点 |
| `disparity_transform` | 深度↔视差域变换 | 域变换 | — | — | **L515 跳过**(ToF 无真视差,这是 D4xx 概念) |

> **纠错记录(来自对抗验证):**
> - **接收增益的选项名与绑定版本相关(本机实测已更正)**:本机安装的 pyrealsense2 上存在的是 `rs.option.receiver_gain`,而 `rs.option.avalanche_photo_diode` **不存在**(`hasattr` 实测)。较新的 librealsense 头文件枚举名为 `RS2_OPTION_AVALANCHE_PHOTO_DIODE`,`receiver_gain` 是旧名 / Viewer 标签。**以本机为准用 `receiver_gain`**;跨机器前先 `hasattr(rs.option, ...)` 探测。
> - **不存在内建的 "modified median" SDK 滤波块**。librealsense 的内建块只有 decimation/spatial/temporal/hole-filling/rotation(+threshold/disparity/hdr-merge)。"modified median"是 Intel D400 白皮书里的**概念**,需自己实现。
> - `threshold_filter` 的选项名是 `rs.option.min_distance` / `rs.option.max_distance`(**不是** `filter_min_distance`);Python 构造默认 0.15/4.0 m。
> - `noise_filtering` 的确切数值范围/默认由**固件**在运行时提供,`0–6` 为文档值,建议在真机上 `get_option_range` 读取。
> - `visual_preset` 枚举**已确认**:CUSTOM=0, DEFAULT=1, NO_AMBIENT=2, LOW_AMBIENT=3, MAX_RANGE=4, **SHORT_RANGE=5**, AUTOMATIC=6;优先用枚举名 `RS2_L500_VISUAL_PRESET_SHORT_RANGE`。注意设完其它选项会把 preset 翻成 CUSTOM,须在 pipeline start 后再设,避免 Vga/Xga 不兼容报错(#8369)。

### 4.2 2D 深度图删除(在 unproject 之前的 organized depth 上)

| 方法 | 机制 | 删除/平滑 | 对边缘飞线效果 | 吃真实面风险 | 计算成本 | 对 L515 适用性 |
|---|---|---|---|---|---|---|
| **窗口深度跨度 / SAD 删除** | WxW 窗内 `max(Z)-min(Z)` 或 SAD 超阈则删,再膨胀 1 px | DELETE | 好(直接抓斜坡中段) | 中(削薄掠射真实面) | 低 | **首选主删除器**;5×5,`spread > max(12mm, 0.03·Z)`,膨胀 1 px |
| **方向导数 / 梯度 + 同号判据** | 左右/上下差都超阈且**同号**=斜坡(删);异号=噪声 | DELETE | 好(与窗口跨度互补) | 中(啃 1–2 px 真实锐边) | O(N),极低 | 阈值随深度 `kTh~max(10mm,0.02·Z)` |
| **法向-视线夹角(mixed-pixel)** | 局部法向与视线夹角≈90°(掠射)则删 | DELETE | 好(物理最对症) | **高(掠射真实斜面误删)** | 中(需法向,边缘处法向本身噪) | **勿单用**;须与深度跳变 AND 门控 |
| **深度自适应角距(Chen 2022,organized)** | `η=k·2d·sin(β/2)`,k=4,与图像邻接比较,超阈删 | DELETE | 好(89% 边缘精度@28fps,论文已核实) | 中(会削真实轮廓) | 低,实时 | **强推**:一套阈覆盖全程,β≈0.0019 rad |
| **Intel modified-median(自实现)** | 与窗中值差超阈则**排除**(非替换) | DELETE | 好 | 中(fill-factor 降,~10% 为估计值) | 低 | 与设备 confidence/preset 配合 |
| **IR/RGB 边缘掩膜** | Scharr/Sobel 在对齐 IR 上求边,膨胀成带,带内深度删 | DELETE | 几乎删净所有边缘飞点 | **高(每条边删一整条带,~60% 有效像素**,FedCSIS) | 低 | Intel 对 L515 的官方建议路线(#7117);与深度梯度 AND 降误删 |
| **形态学腐蚀 valid mask** | 腐蚀 1–2 px 去边缘晕带 | DELETE | 好(清残余 1–2 px 晕) | 高(等量吃真实边) | 极低 | 收尾,cross-3,1 iter |
| **连通域删除** | `|dz|<d_thr` 判连通,删小块 | DELETE | 好(扫多像素飞点条) | 中(删小真实物) | 中 | 收尾,阈须小于最小真实物 |
| **bilateral / joint-bilateral / guided** | 邻域加权平均 / 局部线性模型 | **SMOOTH** | **删不掉,易焊桥** | — | 中(guided O(N)) | **勿用于去飞线**;仅删除后对平坦面降噪 |

> **选型要点(Sabov & Krueger, SCCG 2008,已核实):** 他们直接对比了 neighbour-distance(窗口跨度)、surface-normal、viewing-ray-angle 三种打分,发现**窗口跨度(s1)最鲁棒**;法向打分(s2)**完全失效**(找不到可用阈值);视线夹角(s3)**在掠射看到的平面上失败**(会吃真实斜面)。所以:**窗口跨度做主删除器,夹角只做辅助且必须与深度跳变 AND**。

> **你们当前 `remove_flying_pixels_at_edges` 属于哪一类?** 它是"方向梯度候选 + 两侧均值/方差 + '圆心落在两面之间则删'"的组合,本质是 DELETE 的方向导数 + 窗口跨度混合体,方向正确。它的**天花板问题在采样几何**(见 §5),不在方法类别。

### 4.3 3D 点云删除(unproject 之后)

| 方法 | 机制 | 删除/平滑 | 对边缘飞线效果 | 吃真实面风险 | 计算成本 | 对 L515 适用性 |
|---|---|---|---|---|---|---|
| **SOR** `remove_statistical_outlier` | 均值 kNN 距 > μ+σ_ratio·σ 则删 | DELETE | **弱**(斜坡局部稠密,均距接近真实面) | 中(削远/掠射稀疏真实面) | 中(kd-tree) | **仅收尾**清散点;先 voxel 2–3mm 均衡密度 |
| **ROR** `remove_radius_outlier` | 半径 r 内邻居 < n 则删 | DELETE | 弱-中(仅删稀疏过渡区飞点) | 中-高(固定半径是深度陷阱) | 低 | 收尾;半径≈2.5–3× 局部间距 |
| **深度自适应组织化 ROR** | 图像邻接 + `η=k·2d·sin(β/2)` | DELETE | 好 | 中 | 低 | 同 4.2 角距,本质在 organized 上跑 |
| **DBSCAN** | 密度聚类,删 label=-1 与小簇 | DELETE | **弱**(斜坡把前景/背景连成一个大簇全保留) | 中(删小真实物) | 中 | 仅扫孤立离散簇 |
| **LOF** | 密度对比,LOF≫1 删 | DELETE | 中(比 SOR 好,抓结构性稀疏飞点) | 高(误删真实轮廓) | **高(离线)** | 不能进 50Hz 环;离线批处理 |
| **法向一致/视线夹角(3D)** | PCA 法向 ⊥ 视线则删 | DELETE | 中-好(抓连通斜坡) | 高(掠射真实面) | 中(需法向) | 与组织化角距互补;须验证不削真实斜面 |
| **point-cloud bilateral(Digne 2017)** | 沿法向加权移动点 | **SMOOTH** | **删不掉,易加厚斜坡** | — | 中 | **仅最后**对平坦面降噪,小 σ |

> **为什么 3D 家族整体在边缘飞线上不给力:** 飞线在 3D 里是**局部稠密、连通**的径向斜坡,长得像真实面。unproject 会**丢掉规则像素邻接**,而这正是删除斜坡所需的信息。因此**优先在 2D/organized 深度上删**,3D 只当第二级扫散点。固定半径方法(ROR/DBSCAN eps)在 0.3–2.5 m 上是深度陷阱(L515 侧向间距 ≈ d·β:0.5m≈1mm,1m≈1.9mm,2.5m≈4.8mm)。

### 4.4 学习类 / 颜色引导修复

| 方法 | 机制 | 删除/平滑 | 对边缘飞线效果 | 吃真实面风险 | 计算成本 | 对 L515 适用性 |
|---|---|---|---|---|---|---|
| **Color-Guided FP Correction**(arXiv:2410.08084 / IEEE MMSP 2024,已核实为真) | SAD 检出 top~5% → 约束深度沿视线 + 颜色加权 LS,把飞点**吸附**回前景或背景面 | **删错误深度后沿视线重投影(非跨边平均)** | 好,且**保留点密度** | 低-中(同色前后景会失败) | 高(迭代,需对齐 RGB) | L515 有对齐 RGB,可用;比纯删除重。论文在 Kinect/Oyla 上评测,L515 为合理外推(RMSE 106→54 为示例值,未独立核实) |
| **Mask-ToF**(CVPR 2021) | 学习微透镜掩膜做硬件级校正 | — | 机理参考 | — | — | 仅作根因理解,不可直接部署 |

---

## 5. 针对当前管线的落地建议(代码级)

下面按你们四个集成点给出具体改法与起始参数。文件均为绝对路径引用:`/home/zhanghaoyang/Desktop/dexmani_real/examples/real/test_pointcloud_process.py` 与 `/home/zhanghaoyang/Desktop/dexmani_real/dexmani_real/utils/pointcloud_utils.py`。

### (a) 在源头加 L515 device-level 配置 —— **值得做,P0**

**落点:** `test_pointcloud_process.py:61`(`depth_sensor = device.first_depth_sensor()` 之后)。

```python
# 在 depth_sensor 拿到之后、pipeline 逻辑里设置(注意 preset 要在 start 后设,避免 Vga/Xga 报错)
depth_sensor.set_option(rs.option.visual_preset,
                        int(rs.l500_visual_preset.short_range))   # =5,枚举优先
depth_sensor.set_option(rs.option.confidence_threshold, 2)        # 1–2;别指望删边缘飞线(#7117)
# noise_filtering:范围由固件给,先读再 sweep
nf_range = depth_sensor.get_option_range(rs.option.noise_filtering)
depth_sensor.set_option(rs.option.noise_filtering, nf_range.max)  # 从高往下退,盯物体轮廓
# 近端门:读默认再按最近物体设
depth_sensor.set_option(rs.option.min_distance, 0)               # 需要 <25cm 才置 0
# 接收增益:本机绑定用 receiver_gain(实测 avalanche_photo_diode 不存在);跨机先 hasattr 探测
# depth_sensor.set_option(rs.option.receiver_gain, 18)   # 弱返回面才调高
```

**要点与坑:**
- `noise_filtering` 是**唯一直接针对边缘 tails 的固件旋钮**,范围固件给(文档 0–6),必须 `get_option_range` 读取后经验 sweep;它内部是删还是平滑**未公开**,请用输出 validity mask 验证。
- `confidence_threshold` **不要**指望删飞线(高置信假边,#7117 已确认),设 1–2 只做通用清理。
- 接收增益:**本机用 `rs.option.receiver_gain`**(实测存在;`avalanche_photo_diode` 在本机不存在)。跨机器前 `hasattr` 探测。
- 这一步不改动 §3 之后的任何逻辑,是**零风险、先减源头**的改动。

### (b) 加一个删除类 2D pass —— **推荐:深度自适应角距,而非单独法向夹角**

对抗验证明确指出:**单独的法向-视线夹角会吃真实斜面(Sabov s3 在掠射平面失败)**。所以**不建议**单独加法向夹角 pass;更稳的是加**深度自适应角距删除**(Chen 2022,organized),或把法向夹角**与深度跳变 AND 门控**后再用。

**落点:** `pointcloud_utils.py` 新增函数并加入 `__all__`(`utils/__init__.py:19`),在 `test_...py:169-175` 替换或串联现有调用。

```python
# organized 深度上的角距删除:η = k·2d·sin(β/2),k=4,β≈0.0019 rad(70°/640)
# 对每个有效像素,与 4/8 邻接像素的 3D 距离中位数 > η(z) 则置 0
# η@0.5m≈3.8mm, @1m≈7.6mm, @2.5m≈19mm —— 一套阈覆盖全程
BETA = 0.0019
K = 4.0
def remove_flying_pixels_angular(depth_m, K_intrinsic, k=K, beta=BETA):
    # 用 K_inv 反投影成 (H,W,3),对图像邻接算 3D 距离,超 η 置 0
    ...
```

**若要用法向夹角,务必 AND 门控:** 仅当"法向≈⊥视线(>80–85°)" **且** "相邻存在大深度跳变"时才删 —— 这个交集既删斜坡飞线,又放过平坦斜面。

### (c) 保留现有 ramp-detector 并抬高其天花板 —— **最高性价比,P0**

你们文档(`test_...py:152-163`)记录的"no-eat 天花板"根源在**采样几何**,不在方法本身。三处外科级修改可直接抬高天花板:

1. **修 k=1 采样污染(最关键)。** 当前 kernel(`pointcloud_utils.py:454`)从 `k=1`(紧邻边缘)开始沿方向采样,而 `k=1..3` 恰好落在 1–3 px 的**斜坡内部**,污染了两侧的 mean/std —— 这正是你们不得不把参数调得比库默认大(`noise_threshold` 0.005→0.015、`sample_radius` 5→6)的原因。**加一个 `gap` 死区参数,让采样从 `k_start = gap+1`(gap≈2–3 px)开始**,跳过斜坡本体:
   ```
   # kernel 内:for k in range(k_start, sample_radius+1):  # k_start = gap+1
   ```
   修好后,`noise_threshold` 可回落到更严的 ~0.008,`margin` 也能压小而不吃面。

2. **`margin` 深度自适应。** 把 `:543-552` 的固定 `margin`(0.009)改为 `margin(z)=max(0.008, α·z)`(α≈0.02–0.03)。ToF 噪声随距离增长,固定 margin 在近处过删、远处漏删。同理 `edge_threshold` 也可随 z 缩放。

3. **让紧邻空洞的边缘也能被检测。** 当前 `neighbour_count != 9`(3×3 窗触及无效像素)就把 `grad_mag` 置 0(`pointcloud_utils.py` Sobel 段),导致**空洞旁的边缘完全不测** —— 而飞线恰恰爱长在空洞边。改成"允许 ≥6 个有效邻居时仍计算梯度"可覆盖这类边。

4. **删除后膨胀掩膜 1 px** 清残余晕带(与 §4.2 形态学腐蚀同理)。

这些都是**手术级改动**,每一行都能追溯到"抬高 no-eat 天花板"这一目标,不引入新抽象。

### (d) 何时重新启用 3D outlier pass —— **收尾用,放宽参数,P2**

**落点:** `test_...py:267`(`USE_OUTLIER_REMOVAL`),当前 `nb_points=40, radius=0.01`。

- **定位:** 只当**收尾**清 2D 删除后的**零散残点**,**不是主力**(连通稠密斜坡它删不掉)。
- **当前参数偏激进且是深度陷阱:** `radius=0.01, nb_points=40` 在 2.5 m 处(点距 ~4.8mm,半径内本就点少)会误删真实远面。
- **更稳的收尾:** 先 `remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)`,再(可选)`remove_radius_outlier(nb_points=8, radius=0.02)`;或在 §7 voxel_down(0.003)**之后**再跑,密度已均衡。
- **只有在 2D 删除已把主斜坡清掉、仍见零散飞点时才开**;否则保持 `False`。

---

## 6. 分优先级行动清单

### P0(本周,低风险高收益)
1. **设备源头配置**(§5a):`visual_preset=SHORT_RANGE`、`noise_filtering` sweep、`confidence_threshold=1–2`、`min_distance`。改 `test_...py:61`。验收:侧视点云,边缘 tails 明显减少,真实物体轮廓不消失。
2. **修 k=1 采样污染 + `margin` 深度自适应**(§5c-1、5c-2):改 `pointcloud_utils.py` kernel 采样起点与删除判据。验收:在同一场景把 `margin` 压到 0.006–0.008 而**不吃真实面**(对照你们文档里"0.006 ate / 0.008 ate@sr7"的旧记录,现应能安全通过)。
3. **确认不用平滑器去飞线**:代码中若有 `spatial`/`hole_filling` 用于去飞线,移除或仅置于删除之后、`holes_fill=0`。

### P1(下一步,中等工作量)
4. **加深度自适应角距删除**(§5b):新增 `remove_flying_pixels_angular`,加入 `__all__`,在 `test_...py:169-175` 与现有 ramp-detector 串联或 A/B 对比。验收:一套阈值在 0.3/1.0/2.5 m 三档都不吃真实面。
5. **IR 边缘掩膜 AND 门控**(Intel 官方路线 #7117):读对齐 IR,Scharr 求边,与深度梯度 AND 后才删。验收:纹理/反照率边(平面上的花纹)不再被误删。

### P2(按需,飞线仍残留时)
6. **重启 3D outlier 收尾**(§5d):`statistical` 优先,参数放宽,置于 voxel_down 之后。
7. **(可选,若要保点密度)颜色引导修复**(§4.4):用对齐 RGB 做 detect-then-reproject,把飞点吸附回真实面而非删除。工程量大,仅在"删除损失的边缘像素影响下游"时考虑。

---

## 7. 参考来源

**物理机制 / 学习类:**
- Mask-ToF: Flying Pixel Correction in ToF (CVPR 2021) — https://openaccess.thecvf.com/content/CVPR2021/html/Chugunov_Mask-ToF_Learning_Microlens_Masks_for_Flying_Pixel_Correction_in_Time-of-Flight_CVPR_2021_paper.html
- Color-Guided Flying Pixel Correction in Depth Images (arXiv:2410.08084 / IEEE MMSP 2024) — https://ar5iv.labs.arxiv.org/html/2410.08084 · https://ieeexplore.ieee.org/abstract/document/10743790
- Kadambi et al., Customizing ToF Modulation Codes to Resolve Mixed Pixels (SIGGRAPH 2013) — https://history.siggraph.org/wp-content/uploads/2022/12/2013-Poster-33-Kadambi_Customizing-Time-of-Flight-Modulation-Codes-to-Resolve-Mixed-Pixels.pdf

**L515 硬件 / SDK:**
- Intel RealSense L515 Datasheet Rev002 — https://www.mouser.com/datasheet/2/612/Intel_RealSense_LiDAR_L515_Datasheet_Rev002-1713847.pdf
- librealsense `rs_option.h`(noise_filtering / avalanche_photo_diode / visual_preset 枚举) — https://raw.githubusercontent.com/IntelRealSense/librealsense/master/include/librealsense2/h/rs_option.h
- librealsense Post-Processing Filters 文档(spatial/temporal/hole-filling/threshold 参数与顺序) — https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md
- librealsense #7117(边缘深度插值/L515 假边高置信;Intel 建议用 IR 定位边缘) — https://github.com/IntelRealSense/librealsense/issues/7117
- librealsense #7029(confidence map CNF4 4-bit 0–15) — https://github.com/IntelRealSense/librealsense/issues/7029
- librealsense #8244(L515 min_distance / enable_max_usable_range) — https://github.com/IntelRealSense/librealsense/issues/8244
- librealsense #6816(L515 IR 饱和:laser_power / 接收增益 / confidence max=3) — https://github.com/IntelRealSense/librealsense/issues/6816
- librealsense #8369(Vga/Xga CUSTOM preset 不兼容) — https://github.com/realsenseai/librealsense/issues/8369
- Intel RealSense Depth Post-Processing 白皮书(modified median 概念) — https://realsenseai.com/download/9979/

**2D 删除算法 / 专利:**
- Sabov & Krueger, Identification and correction of flying pixels (SCCG 2008) — https://doi.org/10.1145/1921264.1921293
- Chen Hao et al., 基于梯度聚类的组织化点云边缘提取(仪器仪表学报 2022,`η=k·2d·sin(β/2)`) — DOI 10.19650/j.cnki.cjsi.J2209126
- EP2538242A1 / CN103814306B(方向导数同号飞点判据;SoftKinetic,同一专利族) — https://patents.google.com/patent/EP2538242A1
- CN108961294A(行/列梯度双超阈删"拖点") — https://patentimages.storage.googleapis.com/33/be/22/ca3e8e801faebf/CN108961294A.pdf
- CN111833370A / US20240412393A1(视线射线几何删飞点) — https://patents.google.com/patent/US20240412393A1
- FedCSIS 2019(IR Scharr+Harris 边缘掩膜删深度) — https://annals-csis.org/proceedings/2019/pliks/fedcsis.pdf
- He, Sun, Tang, Guided Image Filtering (ECCV 2010) — https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf
- e-con Systems, What is a flying pixel — https://www.e-consystems.com/blog/camera/technology/what-is-flying-pixel-and-how-can-it-be-mitigated-in-3d-imaging-for-time-of-flight-cameras/

**3D 点云 / 工具:**
- Open3D 离群点去除教程(statistical / radius) — https://www.open3d.org/docs/0.6.0/tutorial/Advanced/pointcloud_outlier_removal.html
- PCL StatisticalOutlierRemoval — https://pointcloudlibrary.gitlab.io/documentation/classpcl_1_1_statistical_outlier_removal.html
- scikit-learn LocalOutlierFactor — https://scikit-learn.org/1.6/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
- Digne & de Franchis, The Bilateral Filter for Point Clouds (IPOL 2017) — https://www.ipol.im/pub/art/2017/179/
- ifm3d Mixed Pixel Filter(角度/距离模式,mixedPixelThresholdRad 默认 0.15 rad) — https://ifm3d.com/v1.1.30/Technology/3D/ProcessingParams/mixedPixelFilter.html
- JeremyBYU/polylidar-realsense(L515 organized cloud + 后处理) — https://github.com/JeremyBYU/polylidar-realsense/
- librealsense #11124(L515 稀疏背面点云噪声) — https://github.com/IntelRealSense/librealsense/issues/11124

---

**报告可信度说明(诚实标注不确定项):**
- `noise_filtering` 的确切范围/默认、以及它内部"删还是平滑",由固件提供且未公开 —— 请在真机 `get_option_range` 读取并用 validity mask 验证。
- L515 `min_distance` 默认(~190 mm)未从一手来源核实。
- Chen et al. 2022 的 `k=4`、89%/28fps 与 EP2538242A1 的闭式法向公式为**机理可信、精确常数需按你们场景实测**。
- Color-Guided 修复的 RMSE 106→54 为示例值,未独立核实;该论文在 Kinect/Oyla 上评测,L515 适用性为合理外推。
- 所有 mm 级阈值均为**起点**,须在你们实际 L515 桌面捕获上 sweep(Sabov 原文即强调阈值"determined empirically")。
