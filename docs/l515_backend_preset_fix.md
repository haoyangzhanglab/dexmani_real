# L515 深度 preset 无法生效 — 诊断、修复与回滚

## 摘要

- **症状**：L515 任何深度 XU 控制写（`visual_preset` / `serializable_device.load_json`，即
  `load_l515_depth_config`）系统性返回 `Device or resource busy`。相机厂调深度 preset
  （Short Range、激光功率、接收增益等）无法下发，深度只能用固件默认值。
- **根因**：librealsense 当前走 **V4L2 后端**（内核 uvcvideo），而 **kernel 6.17 的 uvcvideo
  未打 RealSense 补丁**，在内核层拒绝 L515 的 XU 控制查询（`UVCIOC_CTRL_QUERY` → EBUSY）。
  已排除：用户态占用（lsof 无持有者）、时序（fresh 进程 10 次重试全失败）、配置错误。
- **影响面**：只影响深度**质量调优**。深度流、内参、点云在固件默认 preset（Max Range）下均正常。
- **一句话建议**：多数情况选 **Option 0（接受默认值，零改动）**；仅当确需 Short Range 深度质量，
  才做 **Option B（源码编译 RSUSB 后端）**。

> 为什么不是"升级到 V4L2"：当前**已经是** V4L2 后端（`backend-v4l2.cpp` 枚举日志 +
> `UVCIOC_CTRL_QUERY` 内核 ioctl 双重确认）。问题出在未打补丁的 uvcvideo，故修复思路是
> **绕开它**（Option B）或**修好它**（Option A/C），而非切后端。

---

## 1. 修复选项对比

| 选项 | 做法 | 收益 | 代价 / 可行性 |
|---|---|---|---|
| **0. 接受默认值**（默认推荐） | 不动环境，深度用固件默认 preset | 零风险；深度已验证正常工作 | 无 Short Range 调优 |
| **B. 改用 RSUSB 后端**（需要 preset 时推荐） | 源码编译 `-DFORCE_RSUSB_BACKEND=ON`，用 libusb 直接收发、绕开 uvcvideo | 与内核版本无关，XU 可写 | 无 frame metadata/硬件时间戳；须源码编译；上线前须实测吞吐 |
| A. 当前内核打补丁 | 对 6.17 跑 `patch-realsense-*.sh` | 保留 V4L2 + metadata | ❌ 补丁不支持 6.17，需手工移植，脆弱 |
| C. 降级内核 + 打补丁 | 装 GA 6.8 内核后再打补丁 | 保留 V4L2 + metadata | ⚠️ 动系统内核，需常驻 6.8 |

---

## 2. 环境快照（本机）

| 项 | 值 |
|---|---|
| OS | Ubuntu 24.04.4 LTS (noble), x86_64 |
| Kernel | `6.17.0-35-generic`，uvcvideo 1.1.1（主线未打补丁）|
| Secure Boot | **disabled**（自编译/DKMS 模块无需 MOK 签名即可加载）|
| Python | conda env `real_robot`，Python 3.10.20 |
| 当前包 | `pyrealsense2==2.53.1.4623`（pip wheel，**V4L2 后端**），位于 `$CONDA_PREFIX/lib/python3.10/site-packages/pyrealsense2/` |
| 构建资源 | 24 核 / 60G RAM；已装 `cmake 3.28`、`build-essential`、`libssl-dev`、`libusb-1.0` |

---

## 3. Option B：源码编译 RSUSB 后端

> 思路：以 `CMAKE_INSTALL_PREFIX=$CONDA_PREFIX` 把 librealsense + python 绑定装进 conda 环境
> **内部**，不污染系统 `/usr`，回滚只需删这些文件 + 还原 pip 包。

### 3.1 备份当前 pip 包（回滚关键，务必先做）

```bash
conda activate real_robot
SP=$CONDA_PREFIX/lib/python3.10/site-packages
tar czf ~/pyrealsense2_pipwheel_backup.tgz -C "$SP" pyrealsense2 pyrealsense2-2.53.1.4623.dist-info
# 移开而非删除（既避免与源码安装冲突，也便于回滚）
mv "$SP/pyrealsense2" "$SP/pyrealsense2.pipbak"
mv "$SP/pyrealsense2-2.53.1.4623.dist-info" "$SP/pyrealsense2-2.53.1.4623.dist-info.pipbak"
```

### 3.2 安装编译依赖

```bash
sudo apt update
sudo apt install -y git pkg-config libusb-1.0-0-dev libudev-dev libssl-dev
# 可选（仅当同时要编译 realsense-viewer）：
# sudo apt install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libgtk-3-dev
```

### 3.3 拉取源码（选能在 24.04/gcc-13 上干净编译的稳定 tag）

```bash
cd ~/Desktop
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
# L515 已停产，官方支持止于 v2.54.1（2.55/2.56 对 L515 标注"未验证"）→ 优先用 2.54.1。
git checkout v2.54.1
# 若 2.54.1 在 gcc-13 上编译报错，再退而用较新的 v2.55.1 / v2.56.x（API 兼容，L515 未官方验证但多数功能可用）。
```

### 3.4 安装 RSUSB 专用 udev 规则（非 root 访问设备必需）

```bash
sudo ./scripts/setup_udev_rules.sh          # 安装 99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 3.5 配置 + 编译 + 安装（关键开关 `FORCE_RSUSB_BACKEND=ON`）

```bash
mkdir build && cd build
cmake .. \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j"$(nproc)"
make install     # librealsense2.so → $CONDA_PREFIX/lib；pyrealsense2 → conda site-packages
```

> 若 `make install` 未把 python 绑定放进 site-packages，从 build 目录手动放置：
> ```bash
> cp -v wrappers/python/pyrealsense2*.so "$SP/"
> ```

---

## 4. 验证（Option B 是"大概率修复"，非 100% 保证，务必验）

先拔插一次 L515（让新 udev 规则对该设备生效），再运行：

```bash
$CONDA_PREFIX/bin/python - <<'PY'
import pyrealsense2 as rs
rs.log_to_console(rs.log_severity.warn)
dev = rs.context().query_devices()[0]
ds  = [s for s in dev.query_sensors() if s.get_info(rs.camera_info.name) == "L500 Depth Sensor"][0]
try:
    ds.set_option(rs.option.visual_preset, 5)   # 5 = Short Range
    print("XU WRITE OK -> visual_preset =", ds.get_option(rs.option.visual_preset), " ✅ 修复成功")
except Exception as e:
    print("XU WRITE STILL FAILS:", e, " ❌ RSUSB 未解决 → 回滚(第 5 节) + 走 Option 0/C")
PY
```

- **成功**：打印 `XU WRITE OK`；再跑 `examples/real/test_pointcloud_process.py`，驱动不再打
  `L515 depth preset not applied` 告警。
- **失败**：立即执行第 5 节回滚。

**RSUSB 后端的已知代价**（成功后仍需知晓）：
- 无 frame metadata / 硬件时间戳（本项目录制用 host_time + 对齐网格，影响有限）。
- 个别 L515 上吞吐/稳定性略逊于 V4L2 —— 上线前跑一段真实采集观察丢帧。
- **RSUSB 自身也可能对"快速连续 set_option"报 busy**（社区 #12986 / #13421）——缓解：放慢控制写调用，
  或调大 device-watcher 轮询间隔。故本节验证脚本必须实跑确认。
- 依赖 udev 规则给非 root 访问权限（已在 3.4 安装）。

**先试的零成本招**（社区反复提到；我们的现象是系统性的、不太像这类，但花不了几秒）：
干净 `pipe.stop()+close()`、把 L515 的 micro-USB 接头翻转 180° 重插、先开一次 `realsense-viewer` 初始化设备。

---

## 5. 回滚（还原到当前 pip-wheel V4L2 状态）

```bash
conda activate real_robot
SP=$CONDA_PREFIX/lib/python3.10/site-packages

# 5.1 删除源码安装进 conda 的 realsense 产物
rm -rf "$SP/pyrealsense2" "$SP"/pyrealsense2*.so "$SP"/pyrealsense2-*.dist-info
rm -f  "$CONDA_PREFIX"/lib/librealsense2.so*        # make install 装入的库
# （也可在 librealsense/build 内执行 `make uninstall`，prefix 指向 conda 时无需 sudo）

# 5.2 还原 pip wheel（二选一）
mv "$SP/pyrealsense2.pipbak" "$SP/pyrealsense2"                                          # (a) 用备份目录
mv "$SP/pyrealsense2-2.53.1.4623.dist-info.pipbak" "$SP/pyrealsense2-2.53.1.4623.dist-info"
# pip install --force-reinstall --no-deps pyrealsense2==2.53.1.4623                      # (b) 或重装原版

# 5.3 （可选）卸载 RSUSB udev 规则
sudo rm -f /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# 5.4 验证还原
python -c "import importlib.metadata as m; print('pyrealsense2', m.version('pyrealsense2'))"
```

拔插 L515 后跑 `examples/real/test_pointcloud_process.py`，确认回到"depth 正常、preset 告警照旧"即完成。

---

## 6. 备选方案 A / C（不推荐，仅存档）

- **C（降级内核 + 打补丁）**：`sudo apt install linux-image-generic-hwe-24.04`（GA 6.8 系）→ grub 选
  6.8 启动 → `git checkout v2.55.x` → `sudo ./scripts/patch-realsense-ubuntu-L4S.sh` → 编译 **V4L2**
  后端（`FORCE_RSUSB_BACKEND=OFF`）。收益：保留 metadata/硬件时间戳。代价：动系统内核，需常驻 6.8。
- **A（当前 6.17 打补丁）**：补丁与 6.17 不兼容，需手工移植，脆弱，不推荐。

---

## 7. 与代码的关系（无需改代码）

- 无论用哪个后端，`dexmani_real/sensor/realsense.py` 的 API 用法不变。本轮已做的
  `connect()` 内参自愈、`_open_pipeline` 泄漏兜底、`camera_process` 卡死重建、以及诚实告警**均继续有效**。
- 若 Option B 成功，`load_l515_depth_config` 会真正生效，`L515 depth preset not applied` 告警自动消失。

---

## 8. 社区参考（同类 issue）

- [#11914](https://github.com/IntelRealSense/librealsense/issues/11914) — L515 + `set_xu xioctl(UVCIOC_CTRL_QUERY) ... Device or resource busy`，与本机逐字一致。
- [#13421](https://github.com/IntelRealSense/librealsense/issues/13421) — `load_json` → `get_xu` busy，用户明确在**未打补丁的 kernel 6.8 + V4L2 后端**（机制同款）。
- [#12300](https://github.com/IntelRealSense/librealsense/issues/12300) — 设 `visual_preset` 时 `get_xu` 报错（L515）。
- [#12986](https://github.com/IntelRealSense/librealsense/issues/12986) — `set_option` busy；RSUSB 对快速连续调用也可能 busy 的讨论。
- [#13314](https://github.com/IntelRealSense/librealsense/issues/13314) / [#13065](https://github.com/IntelRealSense/librealsense/issues/13065) / [#13500](https://github.com/IntelRealSense/librealsense/issues/13500) — Ubuntu 24.04 / kernel 6.8 补丁与 DKMS 缺失；多用户确认 **RSUSB 后端在 24.04 可用**。
- 判别：社区把 busy 分两类 —— (a) 陈旧 pipeline/USB/时序（偶发，重插可解）；(b) 未打补丁内核的 V4L2 无法做 XU（系统性）。本机已证明是 **(b)**，对应 RSUSB 解法（Option B）。
