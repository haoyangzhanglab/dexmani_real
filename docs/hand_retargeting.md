# XHand 手部重定向

VR 遥操作分别处理腕部和手指运动。腕部位姿映射为机械臂末端目标，人体右手关键点经重定向生成 XHand 关节目标。两路结果由遥操作控制器合并为臂手命令。[VR 遥操作控制器](../dexmani_real/teleop/vr.py)

## DexPilot 手部求解

1. 将 Hand Tracking Streamer 提供的 21 个手部关键点转换到掌坐标系，用标定的骨长和掌宽检查输入的几何一致性。
2. 将人体手指骨段方向投影到 XHand 机构，生成保留手势形态的指尖参考。[掌系与机构投影](../dexmani_real/teleop/retargeting/landmarks.py)
3. 用 DexPilot 拟合指尖之间及腕到指尖的向量，根据人体指尖间距调整对指目标及其权重，在关节位置和速度约束下输出平滑的 12 维关节目标。[DexPilot 求解](../dexmani_real/teleop/retargeting/solver.py)

## 提交语义

求解以实测手部姿态作为初始参考，随后用已确认提交的目标更新参考，使平滑和速度约束跟随实际提交的命令序列。
