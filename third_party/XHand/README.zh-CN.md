# XHandControl Python SDK 使用说明

<p align="center">
  <a target="_blank" href="./README.md">English</a>
</p>

## 介绍
XHandControl Python SDK 是一套软件开发工具包，用于与 XHand 灵巧手进行通信，控制灵巧手的关节和读取传感器数据。

## 安装依赖
```bash
sudo apt update && sudo apt install -y \
    cmake \
    g++ \
    libcurl4-openssl-dev \
    libssl-dev \
    nlohmann-json3-dev
```
## 运行
### 硬件串口权限设置
在使用串口通信前，需要设置串口权限，否则会出现无法打开的
```bash
# 串口 ID 需要根据实际情况进行修改
sudo chmod 666 /dev/ttyUSB0
```

### 【推荐】在 conda 环境中运行
```bash
# ❗❗注意：由于EtherCAT通信需要，需要设置conda环境中Python解释器的权限
# ❗❗注意：该设置会赋予当前conda环境中 Python 进程对原始网络套接字的访问权限

# 创建conda环境（可选）
conda create -n xhand python=3.10.12 -y

# 激活conda环境
conda activate xhand

# 安装 xhand_controller
# 进入 xhand_control_sdk_py/ 目录
pip install -r requirements.txt
pip install xhand_controller-*-cp310-cp310-*.whl

# 设置conda解释器中的权限
sudo setcap cap_net_raw+ep $(readlink -f $(which python3))

# 运行
# 进入 xhand_control_sdk_py/ 目录
python3 xhand_control_example.py
```

### 在本地环境中运行（需要保证系统安装 Python 3.10 版本，Ubuntu 24 默认安装 Python 3.12）
```bash
# 进入 xhand_control_sdk_py/ 目录
pip install --user -r requirements.txt

# ❗❗注意：安装前确定好 Python3 版本
python3 --version

# ❗❗注意：以下二选一
# （选项一）❗❗注意：本地python环境为 Python 3.10 使用如下
pip install xhand_controller-*-cp310-cp310-*.whl

# （选项二）❗❗注意：本地python环境为 Python 3.12 使用如下
pip install xhand_controller-*-cp312-cp312-*.whl

# 添加依赖路径
echo "$(python3 -c 'import os, sys; from importlib import util; spec = util.find_spec("xhand_controller"); print(os.path.dirname(spec.origin) if spec else "")')" | sudo tee /etc/ld.so.conf.d/xhand.conf
sudo ldconfig

# 运行
# 进入 xhand_control_sdk_py/ 目录
sudo -E python3 xhand_control_example.py
```

## 手指关节模式
- 在手指类 **FingerCommand_t** 中，通过修改参数 **mode** 来控制手指关节的模式。
- 手指关节模式以及参数含义如下表所示： 

| 名称   | 数值   | 备注        |
|--------|:--------:|-------------|
| 无力模式  | 0    |    |
| 位控模式  | 3    |    |
| 力控模式  | 5    |    |

## 预设动作列表
- 关节位置: 数值依次为0-12的关节角度位置

|XHAND 类型|动作名|关节位置|备注|
|:---:|---|---|---|
|XHAND 1|Fist|11.85, 74.58, 40, -3.08, 106.02, 110, 109.75, 107.56, 107.66, 110, 109.1, 109.15|握拳|
||Palm|0, 80.66, 33.2, 0.00, 5.11, 0, 6.53, 0, 6.76, 4.41, 10.13, 0|亮掌|
||V|38.32, 90, 52.08, 6.21, 2.6, 0, 2.1, 0, 110, 110, 110, 109.23|比耶|
||OK|45.88, 41.54, 67.35, 2.22, 80.45, 70.82, 31.37, 10.39, 13.69, 16.88, 1.39, 10.55|好的|
|||||
|XHAND 1 Lite|Fist|0, 58, 83, 80, 80, 80, 0, 0, 0, 0, 0, 0|握拳|
| |Palm|0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0|亮掌|
| |V|54, 66, 0, 0, 80, 80, 0, 0, 0, 0, 0, 0|比耶|
| |OK|49, 43, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0|好的|

## 重要声明
用户使用本产品即视为已充分阅读、理解并同意接受用户协议的全部条款约束。在使用过程中，用户应严格遵守该协议约定，合理合法地使用本产品。用户协议详见【<a target="_blank" href="./用户协议.md">**用户协议**</a>】。