# Meta Quest ADB 连接与调试指南

适用对象：Meta Quest / Quest 2 / Quest Pro / Quest 3 / Quest 3S 等基于 Android 的 Quest 设备。  
主要场景：Unity / Native Android / hand-tracking-sdk / logcat / APK 安装 / ADB shell 调试。  
当前验证设备示例：Quest 3S，USB ID `2833:5013 Oculus VR, Inc. Quest 3S`。

---

## 1. 目标状态

连接成功后，电脑端执行：

```bash
adb devices -l
```

应该看到类似：

```text
List of devices attached
340YC10GCD0RZV    device usb:2-1 product:xxx model:xxx device:xxx transport_id:1
```

其中关键字段是：

```text
device
```

常见异常状态：

```text
no permissions   # Linux/udev 权限问题
unauthorized     # Quest 设备没有授权当前电脑的 ADB RSA key
offline          # ADB 状态异常，通常需要重启 adb server 或重新插拔 USB
```

---

## 2. 前置条件

### 2.1 Quest 侧

需要满足：

1. Meta 账号已启用开发者身份。
2. Quest 已开启 Developer Mode。
3. 使用支持数据传输的 USB-C 线，不要只用充电线。
4. 头显保持开机、解锁、亮屏。
5. 首次连接时，在头显中允许 USB debugging。

Quest 设备内常见路径：

```text
Quick Settings / Quick Control
→ Settings
→ Developer
→ MTP Notification: On
```

首次连接电脑后，头显内通常会弹出：

```text
Allow USB debugging?
```

推荐选择：

```text
Always allow from this computer
→ Allow
```

### 2.2 Ubuntu / Debian 侧

安装 ADB 与常见 Android udev rules：

```bash
sudo apt update
sudo apt install adb android-sdk-platform-tools-common
```

确认当前用户在 `plugdev` 组：

```bash
groups
```

如果没有 `plugdev`，执行：

```bash
sudo usermod -aG plugdev $USER
```

然后注销当前 Linux 用户并重新登录。  
注意：用户组变更只有在重新登录后才会生效。

---

## 3. 首次连接标准流程

### Step 1：清理旧的 adb server

```bash
sudo pkill adb || true
adb kill-server
```

如果输出：

```text
cannot connect to daemon at tcp:5037: Connection refused
```

通常不用紧张，只表示 adb server 当前没有在运行。

---

### Step 2：连接 Quest

插上 USB-C 数据线，保持头显亮屏并戴上头显观察弹窗。

---

### Step 3：启动 adb

```bash
adb start-server
adb devices -l
```

可能出现三种结果。

#### 情况 A：成功

```text
340YC10GCD0RZV    device
```

说明 ADB 连接可用。

#### 情况 B：unauthorized

```text
340YC10GCD0RZV    unauthorized
```

说明 Linux 侧权限已经通了，但 Quest 还没有授权当前电脑。

处理方法：

1. 戴上头显；
2. 查看是否有 `Allow USB debugging?` 弹窗；
3. 勾选 `Always allow from this computer`；
4. 点击 `Allow`；
5. 电脑端重新执行：

```bash
adb devices -l
```

#### 情况 C：no permissions

```text
340YC10GCD0RZV    no permissions (user in plugdev group; are your udev rules wrong?)
```

说明 Linux 侧 udev 权限没有匹配到该设备。见第 4 节。

---

## 4. 解决 no permissions：添加 Quest udev rule

### 4.1 找到设备 vendor id

建议用插拔前后对比法：

```bash
lsusb > /tmp/lsusb.before
```

插上 Quest 后：

```bash
lsusb > /tmp/lsusb.after
diff -u /tmp/lsusb.before /tmp/lsusb.after
```

Quest 3S 示例输出：

```text
Bus 002 Device 004: ID 2833:5013 Oculus VR, Inc. Quest 3S
```

解释：

```text
2833 = vendor id
5013 = product id
```

对于 Meta / Oculus 设备，常见 vendor id 是：

```text
2833
```

---

### 4.2 添加本地 udev rule

创建本地规则文件：

```bash
sudo tee /etc/udev/rules.d/51-android-local.rules > /dev/null <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF
```

如果你的 `lsusb` 查到的 vendor id 不是 `2833`，把上面的 `2833` 替换成你的实际 vendor id。

---

### 4.3 重载 udev

```bash
sudo chmod a+r /etc/udev/rules.d/51-android-local.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo systemctl restart udev
```

然后：

```bash
adb kill-server
```

拔掉 Quest，重新插上，再执行：

```bash
adb start-server
adb devices -l
```

预期结果应该从：

```text
no permissions
```

变为：

```text
unauthorized
```

或者直接变为：

```text
device
```

如果变成 `unauthorized`，说明 udev 已经修好，只需要在 Quest 里允许 USB debugging。

---

## 5. 解决 unauthorized：重置 ADB 授权

如果头显里没有弹出授权窗口，或一直停留在 `unauthorized`，执行下面流程。

### 5.1 重启 adb server

```bash
adb kill-server
adb start-server
adb devices -l
```

然后拔插 USB，戴上头显看授权弹窗。

---

### 5.2 删除本机 ADB key

```bash
adb kill-server
rm -f ~/.android/adbkey ~/.android/adbkey.pub
adb start-server
adb devices -l
```

然后重新插拔 USB。  
头显应重新弹出：

```text
Allow USB debugging?
```

---

### 5.3 在 Quest 中撤销旧授权

如果 Quest 设置里有类似选项：

```text
Settings
→ System / Developer
→ Revoke USB debugging authorizations
```

执行撤销后，再重新插拔 USB。

---

## 6. 不要长期使用 sudo adb

临时验证可以使用：

```bash
sudo adb devices
```

如果 `sudo adb devices` 可以识别设备，而普通用户 `adb devices` 不行，说明基本就是 udev / 用户组权限问题。

但不建议长期使用：

```bash
sudo adb
```

原因：

1. 可能启动 root 用户的 adb server；
2. 可能把 `~/.android/adbkey` 权限搞乱；
3. 后续 Python / Unity / SDK 脚本通常使用普通用户调用 adb，容易状态不一致。

清理 root adb server：

```bash
sudo pkill adb || true
adb kill-server
adb start-server
```

修复本地 adb key 权限：

```bash
sudo chown -R $USER:$USER ~/.android
```

---

## 7. 常用检查命令

### 7.1 查看设备列表

```bash
adb devices -l
```

### 7.2 查看设备型号

```bash
adb shell getprop ro.product.model
```

### 7.3 查看 Android / Horizon OS 相关属性

```bash
adb shell getprop | grep -iE "ro.product|ro.build|oculus|meta|vr"
```

### 7.4 进入设备 shell

```bash
adb shell
```

退出：

```bash
exit
```

### 7.5 安装 APK

```bash
adb install path/to/app.apk
```

覆盖安装：

```bash
adb install -r path/to/app.apk
```

降级安装，适合调试旧版本 APK：

```bash
adb install -r -d path/to/app.apk
```

### 7.6 卸载应用

```bash
adb uninstall com.example.package
```

### 7.7 查看 logcat

```bash
adb logcat
```

按关键词过滤：

```bash
adb logcat | grep -i "hand"
```

或者：

```bash
adb logcat | grep -iE "Unity|XR|Oculus|Hand|Tracking"
```

### 7.8 清空 logcat

```bash
adb logcat -c
```

### 7.9 截取当前日志到文件

```bash
adb logcat -d > quest_logcat.txt
```

---

## 8. 多设备场景

如果同时连接多个 Android / Quest 设备，先查看序列号：

```bash
adb devices -l
```

示例：

```text
340YC10GCD0RZV    device
another_device    device
```

指定某个设备执行命令：

```bash
adb -s 340YC10GCD0RZV shell
adb -s 340YC10GCD0RZV install -r app.apk
adb -s 340YC10GCD0RZV logcat
```

---

## 9. hand-tracking-sdk 调试建议

ADB 可用后，再运行 hand-tracking-sdk 或相关部署脚本。

最低检查：

```bash
adb devices -l
```

必须是：

```text
device
```

不要在以下状态继续跑 SDK：

```text
no permissions
unauthorized
offline
```

常用调试组合：

```bash
adb logcat -c
# 启动你的 hand-tracking 示例程序
adb logcat | grep -iE "hand|tracking|xr|oculus|meta"
```

如果 SDK 需要安装 APK：

```bash
adb install -r path/to/hand_tracking_demo.apk
```

如果需要查看包名：

```bash
adb shell pm list packages | grep -i hand
adb shell pm list packages | grep -i oculus
```

---

## 10. 常见问题速查

### 10.1 `adb devices` 显示 no permissions

原因：Linux 用户没有该 USB 设备的访问权限。

处理：

```bash
groups
sudo apt install android-sdk-platform-tools-common
```

如果还不行，添加 Quest vendor id 的 udev rule：

```bash
sudo tee /etc/udev/rules.d/51-android-local.rules > /dev/null <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
sudo systemctl restart udev
```

然后拔插 USB。

---

### 10.2 `adb devices` 显示 unauthorized

原因：Quest 没有确认当前电脑的 RSA debugging key。

处理：

```bash
adb kill-server
adb start-server
adb devices -l
```

然后戴上 Quest，允许 USB debugging。

如果没弹窗：

```bash
adb kill-server
rm -f ~/.android/adbkey ~/.android/adbkey.pub
adb start-server
```

重新插拔 USB。

---

### 10.3 `adb devices` 没有任何设备

可能原因：

1. USB 线只支持充电；
2. Quest 没开机或没解锁；
3. Developer Mode 没开；
4. USB 口或 Hub 有问题；
5. 设备没有被系统识别。

检查：

```bash
lsusb
```

如果 `lsusb` 也看不到 Quest，优先换线、换 USB 口、避免使用 Hub。

---

### 10.4 `adb devices` 显示 offline

处理：

```bash
adb kill-server
adb start-server
```

然后重新插拔 USB。  
必要时重启 Quest。

---

### 10.5 命令行出现 `>` 续行提示

如果输入：

```bash
sudo apt install adb android-sdk-platform-tools-common\
```

末尾的 `\` 会让 shell 进入续行模式，出现：

```text
>
```

正确写法：

```bash
sudo apt install adb android-sdk-platform-tools-common
```

除非你明确需要多行命令，否则不要在命令末尾加 `\`。

---

## 11. 推荐的一键检查流程

新设备第一次连接时，建议直接按下面顺序执行：

```bash
# 1. 安装 adb 和常见 udev rules
sudo apt update
sudo apt install adb android-sdk-platform-tools-common

# 2. 确认用户组
groups

# 如果没有 plugdev，执行：
# sudo usermod -aG plugdev $USER
# 然后注销并重新登录

# 3. 清理 adb
sudo pkill adb || true
adb kill-server

# 4. 插上 Quest 后检查
lsusb
adb start-server
adb devices -l
```

如果出现 `no permissions`：

```bash
sudo tee /etc/udev/rules.d/51-android-local.rules > /dev/null <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF

sudo chmod a+r /etc/udev/rules.d/51-android-local.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo systemctl restart udev

adb kill-server
adb start-server
adb devices -l
```

然后拔插 USB，并在 Quest 中允许 USB debugging。

---

## 12. 判断修复进度

| adb 状态 | 含义 | 下一步 |
|---|---|---|
| 无设备 | 系统没识别到 USB 设备 | 换线、换口、检查 Quest 是否开机 |
| `no permissions` | Linux udev 权限未匹配 | 添加 udev rule |
| `unauthorized` | Quest 未授权 ADB RSA key | 在头显里允许 USB debugging |
| `offline` | adb server / 设备状态异常 | 重启 adb server，重新插拔 |
| `device` | 正常 | 可以跑 SDK、安装 APK、看 logcat |

---

## 13. 参考资料

- Android Developers: Run apps on a hardware device  
  https://developer.android.com/studio/run/device

- Android Developers: Android Debug Bridge  
  https://developer.android.com/tools/adb

- Meta Horizon OS Developers: Device Setup  
  https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/
