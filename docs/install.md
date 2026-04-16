## Pytorch相关安装
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu118/
```

## XHand相关安装
```bash
pip install "dex-retargeting<0.5.0"
pip install mediapipe==0.10.20

pip install pybind11==2.13.1    # xhand需求
cd third_party/XHand
pip install xhand_controller-1.1.8-cp310-cp310-linux_x86_64.whl
sudo setcap cap_net_raw+ep $(readlink -f $(which python3))
```

## RealSense相关安装
#### 安装SDK
```bash
sudo rm -f /etc/apt/sources.list.d/librealsense*
sudo apt-get update
sudo apt-get install -y curl gnupg apt-transport-https ca-certificates lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/librealsenseai.gpg > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo jammy main" | \
  sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt-get update

sudo apt remove -y \
  librealsense2 librealsense2-dev librealsense2-utils \
  librealsense2-gl librealsense2-udev-rules

sudo apt install -y --allow-downgrades \
  librealsense2=2.54.2-0~realsense.10773 \
  librealsense2-dev=2.54.2-0~realsense.10773 \
  librealsense2-utils=2.54.2-0~realsense.10773 \
  librealsense2-gl=2.54.2-0~realsense.10773 \
  librealsense2-udev-rules=2.54.2-0~realsense.10773

sudo apt-mark hold \
  librealsense2 librealsense2-dev librealsense2-utils \
  librealsense2-gl librealsense2-udev-rules
```
L515 是退役型号，较新的 librealsense / viewer 版本已经不再支持它
L515 最后一个经过验证的 SDK 版本是 v2.50.0，v2.54.2 仍可支持但不再验证；从 v2.55.1 起，L515/SR300 的支持代码已经被移除
安装完成后执行,查看是否安装成功
```bash
realsense-viewer
```
#### 安装python库
```bash
pip install pyrealsense2==2.54.2.5684   #和SDK版本号最好保持一致
```
#### 安装pytorch3d和open3d
```bash
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install open3d
```

## XArm相关安装
```bash
pip install mplib==0.2.1
```

## VR相关安装
```bash
pip install av
pip install "hand-tracking-sdk[visualization]"
pip install "hand-tracking-sdk[video]"
pip install opencv-python==4.9.0.80
pip install numpy==1.26.4
```

## 2. Quest 3S 与电脑连接教程

### 2.1 网络准备

Quest 3S 和电脑必须连接**同一个局域网**（同一 Wi-Fi 或有线）。

在电脑上获取 IP 地址：

```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
```

找到类似 `192.168.1.100` 的地址，记下来。

### 2.2 安装 Quest 端应用

1. 在 Quest 中打开 **Meta Store**
2. 搜索 **"Hand Tracking Streamer"**（免费应用）
3. 下载安装
4. 也可从 [SideQuest](https://sidequestvr.com/) 安装

### 2.3 配置 Quest 端

1. 戴上 Quest，打开 Hand Tracking Streamer
2. 填入以下配置：

| 设置项 | 值 | 说明 |
|--------|------|------|
| **Protocol** | TCP | 可靠传输，数据不丢失 |
| **IP Address** | 电脑的 IP | 第一步记下的地址 |
| **Port** | 8000 | 与代码默认端口一致 |
| **Hand Mode** | Left/Right/Dual | 选择追踪的手 |

3. 点 **Start Streaming**

### 2.4 启动电脑端接收

### 2.5 常见问题

**连接不上？**

1. 确认 Quest 和电脑在同一 Wi-Fi 下
2. 放行防火墙：`sudo ufw allow 8000/tcp`
3. 确认 Quest 里填的 IP 与电脑实际 IP 一致
4. 用 `ping <电脑IP>` 测试连通性

**帧率低？** 确保使用 5GHz Wi-Fi，信号良好。