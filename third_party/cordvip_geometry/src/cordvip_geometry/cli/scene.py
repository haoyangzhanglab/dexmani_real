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
        or set(data) != {"objects"}
        or not isinstance(data["objects"], list)
    ):
        raise ValueError("scene JSON must contain an objects list")
    if not data["objects"]:
        raise ValueError("scene requires at least one fixed instance")
    objects = {}
    for item in data["objects"]:
        if not isinstance(item, dict) or set(item) - {
            "id",
            "role",
            "asset",
            "initial_selection",
            "prompt",
        }:
            raise ValueError(
                "Each scene object accepts id, role, asset, initial_selection, prompt"
            )
        name = item.get("id", "")
        validate_object_name(name)
        if name in objects:
            raise ValueError("Scene object IDs must be unique")
        item = item.copy()
        item["role"] = item.get("role", "object")
        if item["role"] not in {"object", "reference"}:
            raise ValueError("Scene role must be object or reference")
        if headless and not item.get("initial_selection"):
            raise ValueError(
                f"{name}: headless scene requires an initial_selection mask for the first frame"
            )
        if item.get("prompt") is not None and (
            not isinstance(item["prompt"], str) or not item["prompt"].strip()
        ):
            raise ValueError("Scene prompt must be nonempty text")
        if not item.get("initial_selection") and item.get("prompt") is None:
            raise ValueError(f"{name}: scene object requires initial_selection or prompt")
        objects[name] = item
    return objects


def choose_masks(objects, frame, args, output, segmenter, maps):
    import numpy as np

    from ..io import load_initial_selection, write_json, write_selection
    from ..rgbd import save_frame

    _, depth, _ = maps.apply(frame.rgb, frame.depth_m)
    masks, scene_objects = {}, []
    for name, item in objects.items():
        if item.get("initial_selection"):
            masks[name] = load_initial_selection(item["initial_selection"], frame)
            selected = item["initial_selection"]
        else:
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
                if np.count_nonzero(rectified_mask & (depth > 0)) >= MIN_DEPTH_PIXELS:
                    break
                print(
                    f"{name}: mask 至少需要 {MIN_DEPTH_PIXELS} 个有效深度像素，请重新选择。",
                    flush=True,
                )
            masks[name] = mask
            selected = output / "initial_selections" / name
            selected.mkdir(parents=True)
            save_frame(selected / "source_rgbd.npz", frame)
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
    write_json(output / "scene.json", {"objects": scene_objects})
    return masks
