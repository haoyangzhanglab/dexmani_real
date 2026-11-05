"""场景声明、首帧 mask 输入与交互选择。"""

import json
from pathlib import Path

from ..foundationpose import MIN_DEPTH_PIXELS
from ..tracking import validate_object_name
from .selection import choose


def load_scene(path, *, headless):
    data = json.loads(Path(path).read_text())
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("objects"), list)
    ):
        raise ValueError("scene JSON must contain an objects list")
    if not data["objects"]:
        raise ValueError("scene requires at least one fixed instance")
    objects = {}
    for item in data["objects"]:
        if not isinstance(item, dict):
            raise ValueError("Each scene object must be a dictionary")
        name = item.get("id", "")
        validate_object_name(name)
        if name in objects:
            raise ValueError("Scene object IDs must be unique")
        item = {key: item[key] for key in ("id", "asset", "initial_selection", "prompt") if key in item}
        for key in ("asset", "initial_selection"):
            value = item.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name}: {key} must be a nonempty path string")
        if headless and item.get("initial_selection") is None:
            raise ValueError(
                f"{name}: headless scene requires an initial_selection mask for the first frame"
            )
        if item.get("initial_selection") is None and (
            not isinstance(item.get("prompt"), str) or not item["prompt"].strip()
        ):
            raise ValueError(f"{name}: scene object requires initial_selection or nonempty prompt")
        objects[name] = item
    return objects


def choose_masks(objects, frame, args, output, segmenter):
    import numpy as np

    from ..geometry import RectificationMaps
    from ..io import load_initial_selection, write_selection

    maps = None
    masks, scene_objects = {}, []
    for name, item in objects.items():
        if item.get("initial_selection") is not None:
            masks[name] = load_initial_selection(item["initial_selection"], frame)
            selected = item["initial_selection"]
        else:
            if maps is None:
                maps = RectificationMaps(
                    frame.depth_m.shape, frame.K, frame.distortion_coeffs
                )
            valid_depth = (
                maps.apply_mask(frame.depth_m >= 0.001)
                if item.get("asset") is not None
                else None
            )
            candidates = segmenter.predict(
                frame.rgb,
                item["prompt"],
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            while True:
                index = choose(frame.rgb, candidates, None, headless=False, name=name)
                candidate = candidates[index]
                mask = candidate["mask"]
                rectified_mask = maps.apply_mask(mask)
                if not rectified_mask.any():
                    print(f"{name}: 去畸变后的 mask 不能为空，请重新选择。", flush=True)
                elif valid_depth is not None and np.count_nonzero(
                    rectified_mask & valid_depth
                ) < MIN_DEPTH_PIXELS:
                    print(
                        f"{name}: mask 至少需要 {MIN_DEPTH_PIXELS} 个有效深度像素（>= 0.001 m），请重新选择。",
                        flush=True,
                    )
                else:
                    break
            masks[name] = mask
            selected = output / "initial_selections" / name
            selected.mkdir(parents=True)
            write_selection(
                selected,
                mask,
                {
                    "selected_instance": index,
                    "detection_score": candidate["score"],
                    "mask_score": candidate["mask_score"],
                },
                mask_file="mask.npy",
            )
        scene_item = item.copy()
        for key, path in (
            ("asset", item.get("asset")),
            ("initial_selection", selected),
        ):
            if path is not None:
                scene_item[key] = str(Path(path).expanduser().resolve())
        scene_objects.append(scene_item)
    return masks, scene_objects
