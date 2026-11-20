"""OSTrack-256 的共享网络和逐物体模板。"""

import math
import sys
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OSTRACK_CONFIG = "experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml"


def mask_to_xyxy(mask):
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class OSTrackAdapter:
    def __init__(self, root, checkpoint):
        root = Path(root).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"OSTrack-256 checkpoint not found: {checkpoint}")
        config_path = root / OSTRACK_CONFIG
        if not config_path.is_file():
            raise FileNotFoundError(f"OSTrack-256 source/config not found: {config_path}")

        sys.path.insert(0, str(root))
        from lib.config.ostrack.config import cfg, update_config_from_file
        from lib.test.tracker.ostrack import OSTrack

        config = deepcopy(cfg)
        update_config_from_file(str(config_path), base_cfg=config)
        params = SimpleNamespace(
            cfg=config,
            checkpoint=str(checkpoint),
            template_factor=config.TEST.TEMPLATE_FACTOR,
            template_size=config.TEST.TEMPLATE_SIZE,
            search_factor=config.TEST.SEARCH_FACTOR,
            search_size=config.TEST.SEARCH_SIZE,
            debug=0,
            save_all_boxes=False,
        )
        self._prototype = OSTrack(params, dataset_name=None)
        self._objects = {}

    def initialize(self, name, rgb, box):
        """用 XYXY 框设置或替换对象的模板。"""
        x1, y1, x2, y2 = box
        height, width = rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if not all(map(math.isfinite, box)) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid OSTrack initialization box for {name}: {box}")
        # 官方 initialize 为当前对象赋予独立模板和状态，共享在推理中只读的网络。
        tracker = copy(self._prototype)
        tracker.initialize(rgb, {"init_bbox": [x1, y1, x2 - x1, y2 - y1]})
        self._objects[name] = tracker

    def track(self, name, rgb):
        """返回可直接用于 SAM2 的 XYXY 框。"""
        box = self._objects[name].track(rgb)["target_bbox"]
        if box is None:
            return None
        x, y, w, h = box
        height, width = rgb.shape[:2]
        return x, y, min(width, x + w), min(height, y + h)

    def clear(self):
        self._objects.clear()
