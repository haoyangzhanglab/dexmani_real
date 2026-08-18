# DexMani Sim HDF5/Zarr 读取合同

本文是 Sim HDF5/Zarr 的防御性读取参考，不是本仓库维护的 schema。仓库不包含
Sim producer；文件没有可靠的 schema version，读取器必须检查实际 keys、shape、
dtype 和 episode boundary。

Real v17 不是 Sim 数据的别名。当前约束是 Real 数据只用于真机训练/评测，Sim 数据只
用于仿真训练/评测；不同 domain 的 Zarr 不拼接、不共同拟合 normalizer。需要比较字段
名字或 Policy 读取接口时，先读 [`real_to_sim_mapping.md`](real_to_sim_mapping.md)，不能
把该对照表当作跨 domain 转换许可。

## 1. 两种容器

### HDF5 episode

常见 `.h5` 文件以一个 episode 为单位，逐帧 dataset 位于根层，episode 级信息位于
root attributes。以下 13 个 key 只是目标读取格式的参考；读取前必须枚举实际 keys：

```text
action, action_ee, camera_extrinsic, camera_intrinsic,
contact_force, depth, done, fingertip_points,
imagine_point_cloud, joint_state, point_cloud, rgb, segmentation
```

### Zarr task store

一个 `<task>.zarr` 目录把多个 episode 沿第 0 维拼接：

```text
data/<key>             # 拼接后的数组
meta/episode_ends      # exclusive cumulative end indices
```

此处描述的是 Sim converter 的常见行为：Zarr 可能不保留 HDF5 root attributes、文件名、
seed 或 episode provenance，需要按 Sim producer 的合同管理。Real Policy Zarr 的独立
最小合同见 [`processed_hdf5.md`](processed_hdf5.md)，两者不能混用。

## 2. HDF5 数据语义

令 `N` 为 episode 长度。下表是目标读取格式的 shape/dtype；实际文件不匹配时以
producer 或 manifest 为准，不要静默 reshape 或 cast：

| dataset | shape | dtype | 语义 |
|---|---|---|---|
| `rgb` | `(N,240,320,3)` | `uint8` | RGB 图像 |
| `depth` | `(N,240,320)` | `uint16` | 光轴深度，0 表示无效 |
| `segmentation` | `(N,240,320)` | `uint8` | actor 语义 id，0 为背景 |
| `point_cloud` | `(N,1024,6)` | `float32` | SAPIEN world XYZ[m] + RGB[0,1] |
| `camera_intrinsic` | `(N,9)` | `float32` | row-major 3×3 K |
| `camera_extrinsic` | `(N,12)` | `float32` | row-major 3×4 world→camera |
| `joint_state` | `(N,19)` | `float32` | action 前 qpos，arm 7 + hand 12，rad |
| `contact_force` | `(N,15)` | `float32` | 5 个手指 link 的 world-frame xyz force，N |
| `fingertip_points` | `(N,15)` | `float32` | 5 个 fingertip 的 world-frame xyz，m |
| `imagine_point_cloud` | `(N,512,6)` | `float32` | 手部 mesh 派生点云 |
| `action` | `(N,19)` | `float32` | arm + hand joint target，rad |
| `action_ee` | `(N,21)` | `float32` | EEF position/rot6d + hand target |
| `done` | `(N,)` | `bool` | `env.step(action[t])` 后的 transition 结果 |

`joint_state` 和 `action` 的 19 维顺序为 arm 7 维后接 XHand 12 维；精确名称顺序以 Sim producer 为准。`camera_extrinsic` 是 world→camera，不能和 Real 的 camera→world metadata 直接互换。

时间语义是：`obs[t]` 是执行 `action[t]` 前的状态，`action[t]` 驱动下一状态，`done[t]` 描述该 action 之后的 transition。`action` 是 recorder 收到的目标，不保证等于底层 PD/limit clip 后的实际应用值。

## 3. 相机和坐标

- 常见图像 shape 是 `(240,320,...)`，即 H×W；不要从 Real 的默认分辨率预分配 Sim tensor。
- depth 是相机光轴深度，不是到相机中心的欧氏距离；0 是 invalid。
- point cloud、fingertip 和 `action_ee` 使用 SAPIEN world；robot root 的安装变换已包含在这些 world-frame 值中。
- `camera_extrinsic` 使用 OpenCV RDF camera frame 的 world→camera 3×4 矩阵。
- 文件可能没有把单位和 frame 写成 dataset attributes；读取器必须使用已核对的 producer 或 manifest。

## 4. Zarr 读取规则

令 `T` 为所有 `data/*` 的首维、`E` 为 episode 数：

```python
ends = np.asarray(store["meta/episode_ends"][:], dtype=np.int64)
start = 0
for end in ends:
    episode = store["data/action"][start:end]
    start = int(end)
assert start == store["data/action"].shape[0]
```

读取前检查：

1. `episode_ends` 为一维、非空、严格递增的整数。
2. 最后一个 end 等于每个 `data/*` 的首维。
3. 所有 data array 的 episode 内 tail shape 和 dtype 与实际 schema 一致。
4. sequence sampler 不跨越 `episode_ends` 边界。
5. 不假设任意 Zarr 都有 13 个 key；先枚举实际 arrays。

某些 converter 会根据首个输入文件推导 key/schema，且 Zarr 内可能没有完成标记或
provenance manifest。转换前应在临时目录写入，完成后自行核对 key、shape、边界和来源清单。

## 5. success、failed、done

这些字段不是同义词：

- `done[t]` 是逐 transition 的 post-action 结果。
- root `success` 是 episode/task 级结果，不是某一行的 observation 标签。
- `failed`、`truncated` 和 `success` 分别由 recorder/task lifecycle 定义，不能互相复制。
- 不要用“末帧 `done=True`”“全 False”或 Real recorder 的 `success` 猜造缺失字段。

## 6. 防御性读取最小模板

```python
keys = sorted(actual_data_keys)
required = {"joint_state", "action", "done"}
missing = required - set(keys)
if missing:
    raise ValueError(f"missing Sim datasets: {sorted(missing)}")

for key in keys:
    array = store[f"data/{key}"]
    if array.shape[0] != int(ends[-1]):
        raise ValueError(f"length mismatch: {key}")
```

若需要完整字段表或代码定位，优先查看 Sim producer/converter；本文只冻结读取所需的不变量，不把一次样本审计结果写成永久合同。
