# Intel RealSense L515 安装、验证与机器人 3D 模仿学习使用规范

> 适用对象：Ubuntu 22.04 + Linux kernel 6.8.x + Intel RealSense L515 + Conda Python 环境。  
> 已验证组合：`librealsense v2.54.2` + `pyrealsense2 2.54.2` + `Firmware 1.5.8.1` + `Python 3.10` + `RSUSB backend`。  
> 本文档按实际调试过程反思整理，目标是避免 `apt / DKMS / conda-forge / pip wheel / 源码安装` 混用导致的版本污染。

---
## L515参数参考
```text
https://github.com/JS-RML/Learning-to-Grasp-by-Digging/blob/main/real/640X480_L_short_default.json
```

## 0. 最终结论

推荐固定使用这套方案：

```text
Ubuntu 22.04 + kernel 6.8.x
L515 firmware 1.5.8.1
librealsense v2.54.2
FORCE_RSUSB_BACKEND=ON
Conda env: real_robot
Python: 3.10
```

不要再混用以下安装方式：

```bash
sudo apt install librealsense2-*
conda install -c conda-forge librealsense
pip install pyrealsense2
pip install pyrealsense
```

原因：

1. `apt + DKMS` 在 Ubuntu 22.04 + kernel 6.8 上容易遇到 DKMS / uvcvideo patch 问题。
2. conda-forge 可能自动选择不匹配的 Python build 或 CUDA 变体，曾在 `Python 3.10` 环境中安装 `py314` 版本。
3. PyPI 的 `pyrealsense2` 是二进制 wheel，不一定与本机 L515 / SDK 版本 / RSUSB backend 组合一致。
4. L515 是较老型号，实测 `v2.57.7` 无法识别 L515；切到 `v2.54.2` 后可正常识别和采集。

---

## 1. 已验证成功状态

最终验证输出应接近：

```text
Python 3.10.20
pyrealsense2: 2.54.2
devices: 1
name: Intel RealSense L515
firmware: 1.5.8.1
usb: 3.2
```

Python 采集测试应能输出：

```text
depth: 1024 768
color: 1920 1080
center distance: <合理距离，单位 m>
```

---

## 2. 清理旧安装

### 2.1 拔掉相机

安装和清理前先拔掉 L515。

```bash
conda deactivate 2>/dev/null || true
```

---

### 2.2 清理 apt 版 RealSense

```bash
sudo apt-mark unhold \
  librealsense2 \
  librealsense2-gl \
  librealsense2-utils \
  librealsense2-dev \
  librealsense2-dkms \
  librealsense2-udev-rules 2>/dev/null || true

sudo apt purge -y 'librealsense2*'
sudo apt autoremove -y
sudo apt clean

sudo rm -f /etc/apt/sources.list.d/librealsense.list
sudo rm -f /etc/apt/sources.list.d/realsense-public.list
sudo rm -f /etc/apt/keyrings/librealsenseai.gpg

sudo apt update
```

确认：

```bash
dpkg -l | grep -Ei "librealsense|realsense" || echo "apt 层 RealSense 已清理"
```

---

### 2.3 清理 udev 规则

```bash
sudo rm -f /etc/udev/rules.d/*realsense*
sudo rm -f /lib/udev/rules.d/*realsense*

sudo udevadm control --reload-rules
```

如需触发规则，优先只触发 USB 子系统：

```bash
sudo udevadm trigger --action=add --subsystem-match=usb
sudo udevadm settle
```

不建议频繁执行裸命令：

```bash
sudo udevadm trigger
```

裸触发会扫描大量系统虚拟设备，可能出现与 RealSense 无关的 `Permission denied` 提示。

---

### 2.4 清理 `/usr/local` 源码安装残留

```bash
sudo rm -rf \
  /usr/local/lib/cmake/realsense2 \
  /usr/local/lib/cmake/realsense2-gl \
  /usr/local/lib/cmake/pyrealsense2 \
  /usr/local/include/librealsense2 \
  /usr/local/include/librealsense2-gl \
  /usr/local/lib/librealsense2.so \
  /usr/local/lib/librealsense2.so.* \
  /usr/local/lib/librealsense2-gl.so \
  /usr/local/lib/librealsense2-gl.so.*

sudo rm -f \
  /usr/local/bin/realsense-viewer \
  /usr/local/bin/rs-enumerate-devices \
  /usr/local/bin/rs-fw-update \
  /usr/local/bin/rs-depth-quality \
  /usr/local/bin/rs-capture \
  /usr/local/bin/rs-convert \
  /usr/local/bin/rs-save-to-disk

sudo ldconfig
```

确认：

```bash
sudo find /usr/local -iname "*realsense*" -o -iname "*pyrealsense2*" -o -iname "librealsense2*"
```

---

### 2.5 不要污染主机器人环境

如果已有主环境，例如 `base_robot`，不要在其中试 RealSense。RealSense 使用独立环境：

```text
real_robot
```

如果某个环境被 conda-forge RealSense 污染，可用：

```bash
conda list --revisions
conda install --revision <安装前的 revision>
```

---

## 3. 安装系统依赖

```bash
sudo apt update
sudo apt install -y \
  git wget cmake build-essential pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev \
  libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
  python3-dev v4l-utils
```

说明：`v4l-utils` 不是核心 SDK 依赖，但 `setup_udev_rules.sh` 会调用 `v4l2-ctl` 做检查，缺失时会提示安装。

---

## 4. 创建 Conda 环境

```bash
conda create -n real_robot python=3.10 numpy opencv -y
conda activate real_robot

python -V
which python
echo "$CONDA_PREFIX"
```

预期：

```text
/home/zhy/anaconda3/envs/real_robot/bin/python
/home/zhy/anaconda3/envs/real_robot
```

---

## 5. 下载 librealsense v2.54.2

```bash
mkdir -p ~/src
cd ~/src

rm -rf librealsense
git clone --depth 1 --branch v2.54.2 https://github.com/IntelRealSense/librealsense.git
cd librealsense

git describe --tags
```

预期：

```text
v2.54.2
```

---

## 6. 安装 udev 权限规则

```bash
cd ~/src/librealsense
./scripts/setup_udev_rules.sh

sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=usb
sudo udevadm settle
```

拔掉 L515，等待 5 秒，重新插入主板 USB3 口。

检查：

```bash
lsusb | grep -Ei "realsense|intel|8086|03e7"
lsusb -t
```

预期可见：

```text
8086:0b64 Intel RealSense 515
5000M 或 USB 3.x
```

---

## 7. 编译 SDK + Viewer + Python Binding

```bash
conda activate real_robot

cd ~/src/librealsense
rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_GRAPHICAL_EXAMPLES=ON \
  -DBUILD_TOOLS=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCHECK_FOR_UPDATES=OFF \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib;$ORIGIN/../../..'

make -j"$(nproc)"
make install
```

注意：不要使用 `sudo make install`。安装目标是当前 conda 环境，使用 `sudo` 可能导致权限混乱。

---

## 8. 修复 Python binding 安装路径

有时 CMake 会把 Python binding 错装到：

```text
$CONDA_PREFIX/OFF/
```

如果出现：

```text
ModuleNotFoundError: No module named 'pyrealsense2'
```

执行：

```bash
conda activate real_robot

PY_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

echo "$PY_SITE"

rm -f "$PY_SITE"/pyrealsense2*.so*
rm -f "$PY_SITE"/pybackend2*.so*

cp -avL ~/src/librealsense/build/Release/pyrealsense2.cpython-310-x86_64-linux-gnu.so* "$PY_SITE/"
cp -avL ~/src/librealsense/build/Release/pybackend2.cpython-310-x86_64-linux-gnu.so* "$PY_SITE/"
```

验证：

```bash
python - <<'PY'
import pyrealsense2 as rs

print("pyrealsense2:", rs.__version__)
ctx = rs.context()
devices = ctx.query_devices()
print("devices:", len(devices))

for d in devices:
    print("name:", d.get_info(rs.camera_info.name))
    print("serial:", d.get_info(rs.camera_info.serial_number))
    print("firmware:", d.get_info(rs.camera_info.firmware_version))
    print("usb:", d.get_info(rs.camera_info.usb_type_descriptor))
PY
```

成功后可删除错误目录：

```bash
rm -rf "$CONDA_PREFIX/OFF"
```

---

## 9. 验证工具链

```bash
conda activate real_robot

which rs-enumerate-devices
which realsense-viewer
ldd "$(which rs-enumerate-devices)" | grep realsense
```

预期：

```text
/home/zhy/anaconda3/envs/real_robot/bin/rs-enumerate-devices
/home/zhy/anaconda3/envs/real_robot/bin/realsense-viewer
librealsense2.so.2.54 => /home/zhy/anaconda3/envs/real_robot/lib/librealsense2.so.2.54
```

检查相机：

```bash
rs-enumerate-devices -c | head -n 40
```

打开 Viewer：

```bash
realsense-viewer
```

---

## 10. 固件更新到 1.5.8.1

下载 L500 Series 固件包后，目录应包含：

```text
Signed_Image_UVC_1_5_8_1.bin
```

进入目录：

```bash
conda activate real_robot
cd ~/Downloads/L500_Series_FW_1_5_8_1
ls -lh Signed_Image_UVC_1_5_8_1.bin
```

确认工具和设备：

```bash
which rs-fw-update
rs-fw-update -l
```

更新：

```bash
rs-fw-update -f Signed_Image_UVC_1_5_8_1.bin
```

如果需要指定序列号：

```bash
rs-fw-update -s f1382055 -f Signed_Image_UVC_1_5_8_1.bin
```

更新时注意：

```text
不要打开 realsense-viewer
不要运行 Python 采集脚本
不要拔线
不要使用 USB HUB
只保留一台 L515 插着
```

更新完成后，拔插相机并验证：

```bash
rs-enumerate-devices -c | grep -E "Name|Serial Number|Firmware Version|Recommended Firmware Version|Usb Type"
```

预期：

```text
Firmware Version              : 1.5.8.1
Recommended Firmware Version  : 1.5.8.1
Usb Type Descriptor           : 3.2
```

---

## 11. Python 采集测试

```bash
conda activate real_robot

python - <<'PY'
import pyrealsense2 as rs

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

profile = pipe.start(cfg)

try:
    for _ in range(30):
        frames = pipe.wait_for_frames()

    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    print("depth:", depth.get_width(), depth.get_height())
    print("color:", color.get_width(), color.get_height())
    print("center distance:",
          depth.get_distance(depth.get_width() // 2, depth.get_height() // 2),
          "m")
finally:
    pipe.stop()
PY
```

---

## 12. 最小保存 RGB-D 帧脚本

```python
# save_l515_frame.py
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from pathlib import Path

out_dir = Path.home() / "l515_test"
out_dir.mkdir(parents=True, exist_ok=True)

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

profile = pipe.start(cfg)
align = rs.align(rs.stream.color)

try:
    for _ in range(30):
        frames = pipe.wait_for_frames()

    frames = align.process(frames)
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    depth_np = np.asanyarray(depth.get_data())
    color_np = np.asanyarray(color.get_data())

    cv2.imwrite(str(out_dir / "color.png"), color_np)
    cv2.imwrite(str(out_dir / "aligned_depth.png"), depth_np)

    intr = color.profile.as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    meta = {
        "color_width": intr.width,
        "color_height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
        "depth_scale": depth_scale,
        "center_depth_m": float(depth.get_distance(depth.get_width() // 2, depth.get_height() // 2)),
    }

    with open(out_dir / "camera.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("saved to", out_dir)
    print(meta)
finally:
    pipe.stop()
```

运行：

```bash
python save_l515_frame.py
ls -lh ~/l515_test
```

---

# 13. L515 不适合的材质、光照和对象

L515 是 LiDAR depth camera，适合反射较稳定、表面偏漫反射的物体。它不适合所有材料和光照环境。

## 13.1 不适合的材质

| 材质/表面                        | 问题                                   | 建议                                             |
| -------------------------------- | -------------------------------------- | ------------------------------------------------ |
| 透明玻璃、透明塑料、亚克力、水杯 | 深度可能穿透、缺失或落到背景上         | 不作为初期主训练对象；可贴哑光胶带或换哑光替代物 |
| 镜面金属、亮面陶瓷、反光桌面     | 入射光不一定反射回相机，深度空洞或跳变 | 改用哑光表面，调整视角                           |
| 黑色/深色低反射物体              | 反射信号弱，深度噪声增加               | 换浅色物体，增加辅助标记                         |
| 高光塑料包装、保鲜膜、塑料袋     | 透明 + 高光，深度和分割都不稳定        | 不建议作为早期学习对象                           |
| 毛巾、衣物、软袋                 | 形变大，3D 状态难定义                  | 需要单独的柔性物体策略                           |
| 细杆、细线、薄片边缘             | 点云稀疏，边缘容易断裂                 | 近距离、多视角、避免作为关键抓取依据             |

## 13.2 不适合的光照

| 场景光               | 问题                             | 建议                                      |
| -------------------- | -------------------------------- | ----------------------------------------- |
| 太阳直射、窗边强日光 | 环境红外会降低深度质量           | 避开窗户，拉窗帘，使用稳定室内光          |
| LED/PWM 灯频闪       | RGB 画面可能出现水波纹、滚动暗带 | 设置 Power Line Frequency 为 50Hz 或 60Hz |
| 强背光               | RGB 曝光和白平衡漂移             | 光源放在相机同侧或侧前方                  |
| 反光背景             | 深度和 RGB 都不稳定              | 使用哑光背景板和哑光桌面                  |
| 频繁变化光照         | 训练数据分布漂移                 | 固定灯光、曝光、白平衡                    |

## 13.3 不适合的对象

| 对象             | 问题                          | 建议                         |
| ---------------- | ----------------------------- | ---------------------------- |
| 透明杯子、玻璃瓶 | 深度不可依赖                  | 初期不要作为主要操作对象     |
| 镜面工具、金属杯 | 点云跳变、空洞                | 贴哑光胶带或换物体           |
| 黑色小零件       | 深度缺失概率高                | 换颜色或增加视觉标记         |
| 高反光包装盒     | 分割和深度都不稳定            | 使用哑光替代物               |
| 软袋、毛巾、衣物 | 状态空间复杂                  | 后期单独建模                 |
| 细小零件         | L515 点云分辨率和空洞影响定位 | 近距离、多视角、提高质量筛选 |

L515 有最小有效深度距离，过近物体不可靠。实际机器人操作中建议让目标处于约 `0.35 m ~ 1.2 m` 的稳定工作范围。

---

# 14. 从机器人 3D 模仿学习角度的注意事项

## 14.1 固定相机-机器人关系

必须稳定保存并复用：

```text
camera intrinsics
camera extrinsics: camera_to_robot_base 或 camera_to_ee
depth scale
RGB-D 对齐方式
分辨率
FPS
桌面高度
物体初始区域
光照配置
```

建议每次任务开始前保存一次相机状态，采集中每帧保存时间戳。

---

## 14.2 不要让策略学习坏深度

坏深度会导致：

```text
抓取点错误
物体中心漂移
点云空洞
接触前距离估计错误
轨迹回放碰撞
策略学到错误 affordance
```

建议过滤：

```text
depth == 0 的点
< 0.25 m 的点
> 任务最大距离的点，例如 > 1.5 m
confidence 低的点
mask 外的点
反光边缘离群点
```

---

## 14.3 固定 RGB 设置

为了减少视觉分布漂移，建议固定：

```text
Power Line Frequency: 中国大陆通常 50Hz；北美通常 60Hz
White Balance Auto: Off
White Balance: 固定值
Auto Exposure: 视场景决定，稳定采集时建议固定
Color resolution: 固定
Depth resolution: 固定
FPS: 固定
```

如果 RGB 像水波一样波动，优先检查 `Power Line Frequency`，而不是 SDK 安装。

---

## 14.4 数据目录结构建议

```text
dataset/
  episode_000001/
    color/
      000000.png
    aligned_depth/
      000000.png
    mask/
      000000.png
    confidence/
      000000.png
    camera.json
    robot_state.jsonl
    action.jsonl
    metadata.json
```

`camera.json` 至少保存：

```json
{
  "fx": 0,
  "fy": 0,
  "ppx": 0,
  "ppy": 0,
  "depth_scale": 0,
  "width": 0,
  "height": 0,
  "camera_to_base": []
}
```

`robot_state.jsonl` 建议每帧保存：

```json
{
  "timestamp": 0.0,
  "joint_positions": [],
  "joint_velocities": [],
  "ee_pose_base": [],
  "gripper_width": 0.0,
  "gripper_state": "open"
}
```

`action.jsonl` 建议保存：

```json
{
  "timestamp": 0.0,
  "target_ee_pose_base": [],
  "delta_ee_pose": [],
  "gripper_command": 0.0
}
```

---

## 14.5 采集任务从易到难

建议顺序：

```text
阶段 1：哑光、浅色、刚体、大物体
阶段 2：不同颜色、不同形状刚体
阶段 3：轻微反光物体
阶段 4：小物体、细长物体、多物体遮挡
阶段 5：透明、高反光、软体物体
```

早期不要直接使用：

```text
透明杯子
玻璃瓶
镜面金属
黑色小零件
塑料袋
毛巾衣物
```

否则很难区分是感知失败还是策略失败。

---

## 14.6 保留失败样本，但要标注失败原因

失败原因建议分为：

```text
depth_missing
segmentation_error
gripper_slip
collision
occlusion
operator_error
object_moved
lighting_failure
calibration_drift
```

训练时可以先只用成功样本；调试和后续提升时再使用失败样本做筛选、分类或数据增强。

---

## 14.7 单视角和多视角建议

单个 L515 推荐放置在：

```text
斜上方 30°~60°
能看到桌面、目标物体和夹爪
不正对窗户
不正对镜面反光面
尽量不被机械臂长期遮挡
```

如果条件允许，建议加一个辅助相机：

```text
主视角：策略输入
辅助视角：标注、调试、遮挡补偿
```

---

# 15. 常见问题

## 15.1 v2.57.7 为什么看不到 L515？

本机实测：USB 层能看到 `8086:0b64 Intel RealSense 515`，但 librealsense `v2.57.7` 工具无法识别；切换到 `v2.54.2` 后正常识别。L515 属于旧型号，因此本项目固定 `v2.54.2`。

---

## 15.2 `v4l2-ctl not found` 怎么办？

安装：

```bash
sudo apt install -y v4l-utils
```

然后重新执行：

```bash
./scripts/setup_udev_rules.sh
```

---

## 15.3 `Cannot open device /dev/video0` 是否严重？

如果最后出现：

```text
udev-rules successfully installed
```

且 `rs-enumerate-devices -c` 能看到 L515，则可以忽略。

---

## 15.4 `control_transfer returned error ... error: Success` 是否严重？

如果 `rs-fw-update -l` 最后能列出设备，通常可以继续。重点看是否能看到：

```text
Name: Intel RealSense L515
firmware version
USB type: 3.2
```

---

## 15.5 `Digital Gain get_xu failed Resource temporarily unavailable` 是什么？

一般是 viewer 查询 L515 某个控制项时的瞬时 USB/UVC 控制传输失败。如果 depth、color、Python 采集正常，可以先忽略。不要频繁拖动 `Digital Gain`。

---

## 15.6 RGB 画面像水波一样不稳定怎么办？

在 viewer 里设置：

```text
RGB Camera → Power Line Frequency → 50Hz 或 60Hz
White Balance Auto → Off
必要时 Auto Exposure → Off
```

中国大陆通常使用 `50Hz`，北美通常使用 `60Hz`。

---

## 15.7 clone conda env 后还能用吗？

同机 clone 后可能可用，但不保证。测试：

```bash
conda activate cloned_env
python -c "import pyrealsense2 as rs; print(rs.__version__)"
rs-enumerate-devices -c
```

如果失败，在 clone env 里重新执行编译安装，或至少重新复制 `pyrealsense2*.so` 和 `pybackend2*.so` 到当前 env 的 `site-packages`。

---

# 16. 最终使用方式

每次使用：

```bash
conda activate real_robot
realsense-viewer
```

或：

```bash
conda activate real_robot
python your_l515_script.py
```

不要再执行：

```bash
sudo apt install librealsense2-*
conda install -c conda-forge librealsense
pip install pyrealsense2
pip install pyrealsense
```

当前固定组合：

```text
librealsense v2.54.2
pyrealsense2 2.54.2
L515 firmware 1.5.8.1
Conda env real_robot
RSUSB backend
```

---

# 17. 参考资料

- Intel RealSense L515 Datasheet Rev003: https://realsenseai.com/wp-content/uploads/2025/06/Intel_RealSense_LiDAR_L515_Datasheet_Rev003.pdf
- Optimizing the RealSense LiDAR Camera L515 Range: https://www.realsenseai.com/news-insights/optimizing-the-lidar-camera-l515-range/
- librealsense Python wrapper README: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md
- pyrealsense2 PyPI: https://pypi.org/project/pyrealsense2/
- RealSense Firmware Update Tool: https://dev.realsenseai.com/docs/firmware-update-tool/
- RealSense L500 Firmware Releases: https://dev.realsenseai.com/docs/firmware-releases-l500/
- robomimic observation modalities: https://robomimic.github.io/docs/tutorials/observations.html
- robomimic dataset overview: https://robomimic.github.io/docs/datasets/overview.html