# XHand 手部重定向

VR 遥操作分别处理腕部和手指运动。腕部位姿映射为机械臂末端目标，人体右手关键点经重定向生成 XHand 关节目标。两路结果由遥操作控制器合并为臂手命令。[VR 遥操作控制器](../dexmani_real/teleop/vr.py)

## DexPilot 手部求解

1. 将 Hand Tracking Streamer 提供的 21 个手部关键点转换到掌坐标系，用标定的骨长和掌宽检查输入的几何一致性。
2. 将人体手指骨段方向投影为 SDK 顺序的 12 维关节角 `requested_q`。对未裁剪的关节角做一次 FK，在手基座系同时生成指尖目标和每指两段方向目标。[掌系与机构投影](../dexmani_real/teleop/retargeting/landmarks.py)
3. 用 DexPilot 拟合指尖之间及腕到指尖的向量，根据人体指尖间距调整对指目标和权重。完整目标还包括相对已提交姿态的阻尼和弱方向损失。后者对五指的近端到远端、远端到指尖共十段单位向量取 `mean(0.5 * ||u - u*||²)`，权重为 `direction_scale_m² / huber_delta_m`。
4. 将备用几何初值裁入当拍关节位置和速度界，用完整目标分别对它和已提交姿态打分。几何初值严格低分才采用，同分沿用已提交姿态。NLopt 每个进入求解的帧执行一次优化，求解后再做时间相关的 EMA。[目标与求解](../dexmani_real/teleop/retargeting/solver.py)

`HandRetargetingConfig` 默认 `use_geometry_seed=True`，设为 `False` 时直接从已提交姿态开始求解。`direction_scale_m` 默认 `0.005`，设为 `0` 时跳过方向项。录制元数据会保存生效配置和手部标定。[重定向配置](../dexmani_real/config/teleop.py)

## 提交语义

求解以实测手部姿态建立参考，随后在命令提交成功后更新。阻尼、速度界和 EMA 都以该参考为准。`max_solve_time_s` 是 NLopt 的求解预算，`--timing-output` 记录的 `hand_retarget` 时延覆盖目标构建、初值打分、求解和滤波。
