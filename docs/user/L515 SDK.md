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
Conda env: real
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

> **相关文档**：通用 RealSense 相机驱动与点云工具见 [realsense.md](realsense.md)。L515 材质/光照/采集建议已迁移至该文档。

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

安装和清理前先拔掉 L515。

```bash
# 清理 apt 版 RealSense
sudo apt-mark unhold librealsense2 librealsense2-gl librealsense2-utils librealsense2-dev librealsense2-dkms librealsense2-udev-rules 2>/dev/null || true
sudo apt purge -y 'librealsense2*' && sudo apt autoremove -y && sudo apt clean
sudo rm -f /etc/apt/sources.list.d/librealsense.list /etc/apt/sources.list.d/realsense-public.list /etc/apt/keyrings/librealsenseai.gpg
sudo apt update
dpkg -l | grep -Ei "librealsense|realsense" || echo "apt 层 RealSense 已清理"

# 清理 udev 规则
sudo rm -f /etc/udev/rules.d/*realsense* /lib/udev/rules.d/*realsense*
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=usb
sudo udevadm settle

# 清理 /usr/local 源码安装残留
sudo rm -rf /usr/local/lib/cmake/realsense2 /usr/local/lib/cmake/realsense2-gl /usr/local/lib/cmake/pyrealsense2
sudo rm -rf /usr/local/include/librealsense2 /usr/local/include/librealsense2-gl
sudo rm -f /usr/local/lib/librealsense2.so /usr/local/lib/librealsense2.so.* /usr/local/lib/librealsense2-gl.so /usr/local/lib/librealsense2-gl.so.*
sudo rm -f /usr/local/bin/realsense-viewer /usr/local/bin/rs-enumerate-devices /usr/local/bin/rs-fw-update /usr/local/bin/rs-depth-quality /usr/local/bin/rs-capture /usr/local/bin/rs-convert /usr/local/bin/rs-save-to-disk
sudo ldconfig
sudo find /usr/local -iname "*realsense*" -o -iname "*pyrealsense2*" -o -iname "librealsense2*"
```

RealSense 使用独立 conda 环境（`real`），不要在主环境中安装。如果某环境被 conda-forge RealSense 污染，用 `conda list --revisions` + `conda install --revision <N>` 回滚。

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
conda create -n real python=3.10 numpy opencv -y
conda activate real

python -V
which python
echo "$CONDA_PREFIX"
```

预期：

```text
/home/zhy/anaconda3/envs/real/bin/python
/home/zhy/anaconda3/envs/real
```

---

## 5. 下载 librealsense + 安装 udev 规则

```bash
mkdir -p ~/src && cd ~/src
rm -rf librealsense
git clone --depth 1 --branch v2.54.2 https://github.com/IntelRealSense/librealsense.git
cd librealsense && git describe --tags   # 预期: v2.54.2

# 安装 udev 规则
./scripts/setup_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=usb
sudo udevadm settle
```

拔掉 L515，等待 5 秒，重新插入主板 USB3 口。检查：

```bash
lsusb | grep -Ei "realsense|intel|8086|03e7"   # 预期: 8086:0b64 Intel RealSense 515
lsusb -t                                          # 预期: 5000M 或 USB 3.x
```

---

## 6. 编译 SDK + Python Binding + 修复路径

```bash
conda activate real
cd ~/src/librealsense
rm -rf build && mkdir build && cd build

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

make -j"$(nproc)" && make install
```

注意：不要使用 `sudo make install`。安装目标是当前 conda 环境，使用 `sudo` 可能导致权限混乱。

如果 CMake 把 Python binding 错装到 `$CONDA_PREFIX/OFF/` 导致 `ModuleNotFoundError`，手动复制：

```bash
PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -f "$PY_SITE"/pyrealsense2*.so* "$PY_SITE"/pybackend2*.so*
cp -avL ~/src/librealsense/build/Release/pyrealsense2.*.so* "$PY_SITE/"
cp -avL ~/src/librealsense/build/Release/pybackend2.*.so* "$PY_SITE/"
rm -rf "$CONDA_PREFIX/OFF"
```

---

## 7. 验证工具链

```bash
conda activate real

which rs-enumerate-devices
which realsense-viewer
ldd "$(which rs-enumerate-devices)" | grep realsense
```

预期：

```text
/home/zhy/anaconda3/envs/real/bin/rs-enumerate-devices
/home/zhy/anaconda3/envs/real/bin/realsense-viewer
librealsense2.so.2.54 => /home/zhy/anaconda3/envs/real/lib/librealsense2.so.2.54
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

## 8. 固件更新到 1.5.8.1

下载 L500 Series 固件包后，目录应包含：

```text
Signed_Image_UVC_1_5_8_1.bin
```

进入目录：

```bash
conda activate real
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

## 9. 采集测试

最小采集与保存测试：

```bash
conda activate real

python - <<'PY'
import json, numpy as np, cv2, pyrealsense2 as rs
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

    print("depth:", depth.get_width(), depth.get_height())
    print("color:", color.get_width(), color.get_height())
    print("center distance:", depth.get_distance(depth.get_width() // 2, depth.get_height() // 2), "m")

    cv2.imwrite(str(out_dir / "color.png"), np.asanyarray(color.get_data()))
    cv2.imwrite(str(out_dir / "aligned_depth.png"), np.asanyarray(depth.get_data()))

    intr = color.profile.as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    with open(out_dir / "camera.json", "w") as f:
        json.dump({"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy, "depth_scale": depth_scale}, f, indent=2)
    print("saved to", out_dir)
finally:
    pipe.stop()
PY
```

---

# 10. 材质、光照与采集建议

L515 不适合的材质/光照/对象选择建议、3D 模仿学习数据采集注意事项（固定相机参数、坏深度过滤、数据目录结构、采集任务分级等）已迁移至 [realsense.md](realsense.md) Section 5-6。

---

# 11. 常见问题

## 11.1 v2.57.7 为什么看不到 L515？

本机实测：USB 层能看到 `8086:0b64 Intel RealSense 515`，但 librealsense `v2.57.7` 工具无法识别；切换到 `v2.54.2` 后正常识别。L515 属于旧型号，因此本项目固定 `v2.54.2`。

---

## 11.2 `v4l2-ctl not found` 怎么办？

安装：

```bash
sudo apt install -y v4l-utils
```

然后重新执行：

```bash
./scripts/setup_udev_rules.sh
```

---

## 11.3 `Cannot open device /dev/video0` 是否严重？

如果最后出现：

```text
udev-rules successfully installed
```

且 `rs-enumerate-devices -c` 能看到 L515，则可以忽略。

---

## 11.4 `control_transfer returned error ... error: Success` 是否严重？

如果 `rs-fw-update -l` 最后能列出设备，通常可以继续。重点看是否能看到：

```text
Name: Intel RealSense L515
firmware version
USB type: 3.2
```

---

## 11.5 `Digital Gain get_xu failed Resource temporarily unavailable` 是什么？

一般是 viewer 查询 L515 某个控制项时的瞬时 USB/UVC 控制传输失败。如果 depth、color、Python 采集正常，可以先忽略。不要频繁拖动 `Digital Gain`。

---

## 11.6 RGB 画面像水波一样不稳定怎么办？

在 viewer 里设置：

```text
RGB Camera → Power Line Frequency → 50Hz 或 60Hz
White Balance Auto → Off
必要时 Auto Exposure → Off
```

中国大陆通常使用 `50Hz`，北美通常使用 `60Hz`。

---

## 11.7 clone conda env 后还能用吗？

同机 clone 后可能可用，但不保证。测试：

```bash
conda activate cloned_env
python -c "import pyrealsense2 as rs; print(rs.__version__)"
rs-enumerate-devices -c
```

如果失败，在 clone env 里重新执行编译安装，或至少重新复制 `pyrealsense2*.so` 和 `pybackend2*.so` 到当前 env 的 `site-packages`。

---

# 12. 最终使用方式

每次使用：

```bash
conda activate real
realsense-viewer
```

或：

```bash
conda activate real
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
Conda env real
RSUSB backend
```

---

# 13. 参考资料

- Intel RealSense L515 Datasheet Rev003: https://realsenseai.com/wp-content/uploads/2025/06/Intel_RealSense_LiDAR_L515_Datasheet_Rev003.pdf
- Optimizing the RealSense LiDAR Camera L515 Range: https://www.realsenseai.com/news-insights/optimizing-the-lidar-camera-l515-range/
- librealsense Python wrapper README: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md
- pyrealsense2 PyPI: https://pypi.org/project/pyrealsense2/
- RealSense Firmware Update Tool: https://dev.realsenseai.com/docs/firmware-update-tool/
- RealSense L500 Firmware Releases: https://dev.realsenseai.com/docs/firmware-releases-l500/
- robomimic observation modalities: https://robomimic.github.io/docs/tutorials/observations.html
- robomimic dataset overview: https://robomimic.github.io/docs/datasets/overview.html