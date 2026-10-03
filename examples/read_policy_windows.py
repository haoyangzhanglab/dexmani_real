#!/usr/bin/env python3
"""离线示例：同一公共缓存按 RGB/触觉各自需要的字段选窗口；不修改数据。

这里只演示连续行窗口，不定义固定频率模型的重采样。真实时间随窗口返回，
时间抖动不删行；缺失所需样本、dispatch 不确定/拒绝和 episode 边界会断开窗口。
"""

import argparse

import numpy as np
import zarr


def policy_windows(root, fields, horizon):
    """小样本示例；返回原始行号，不先压缩有效行再拼接。"""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    data = root["data"]
    ends = root["meta/episode_ends"][:]
    valid = np.ones(int(ends[-1]), dtype=bool)
    for name in (*fields, "action"):
        values = data[name][:]
        valid &= np.isfinite(values).reshape(len(values), -1).all(axis=1)
    # 本示例只用明确 accepted 的两设备动作；CRC 不确定不假称成功执行。
    valid &= (root["row_info/dispatch_status"][:] == 1).all(axis=1)
    windows = []
    start = 0
    for end in ends:
        for index in range(start, int(end) - horizon + 1):
            if valid[index : index + horizon].all():
                windows.append(np.arange(index, index + horizon))
        start = int(end)
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache")
    parser.add_argument("--horizon", type=int, default=2)
    args = parser.parse_args()
    root = zarr.open_group(args.cache, mode="r")
    for name, fields in (
        ("RGB", ("joint_state", "rgb")),
        ("触觉", ("joint_state", "contact_force", "tactile_force")),
    ):
        windows = policy_windows(root, fields, args.horizon)
        print(f"{name}: {len(windows)} 个有效窗口")
        if not windows:
            continue
        # 归一化只统计该策略实际采用的行；不把无效模态填零交给模型。
        used = np.unique(np.concatenate(windows))
        joint = root["data/joint_state"].oindex[used]
        print("  joint mean:", joint.mean(axis=0), "std:", joint.std(axis=0))
        first = windows[0]
        sample = {key: root[f"data/{key}"].oindex[first] for key in (*fields, "action")}
        assert all(np.isfinite(value).all() for value in sample.values())
        print("  原行号:", first)
        print(
            "  实测观测时间(ns，0 为未知):", root["row_info/observation_timestamp_ns"].oindex[first]
        )


if __name__ == "__main__":
    main()
