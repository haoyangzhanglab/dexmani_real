"""场景声明和首帧实例确认；保留来源一致的选择产物。"""

import json
import re
from pathlib import Path


def load_scene(path, workspace, *, headless):
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
            raise ValueError("Each scene object accepts id, role, asset, initial_selection, prompt")
        name = item.get("id", "")
        if (
            not isinstance(name, str)
            or not re.fullmatch("[a-z][a-z0-9_]*", name)
            or "__" in name
            or name in objects
        ):
            raise ValueError(
                "Scene object IDs must be unique lowercase identifiers without double underscores"
            )
        item = item.copy()
        item["role"] = item.get("role", "object")
        if item["role"] not in {"object", "reference"}:
            raise ValueError("Scene role must be object or reference")
        for key in ("asset", "initial_selection"):
            if item.get(key) is not None:
                candidate = Path(item[key]).expanduser()
                item[key] = (
                    candidate if candidate.is_absolute() else workspace / candidate
                ).resolve()
                if not item[key].is_dir():
                    raise ValueError(f"{name}: missing {key} directory {item[key]}")
        if headless and not item.get("initial_selection"):
            raise ValueError(f"{name}: headless scene requires an exact-source initial_selection")
        if item.get("prompt") is not None and (
            not isinstance(item["prompt"], str) or not item["prompt"].strip()
        ):
            raise ValueError("Scene prompt must be nonempty text")
        objects[name] = item
    return objects


def select_mask(rgb, candidates, name):
    """只在冻结首帧选候选或画多边形；没有后台查看/模型任务。"""
    import cv2
    import numpy as np

    from .object_selection import display_available, overlay

    display_available()
    title = f"Select {name}: 1-9/[ ] candidate | p polygon | Enter confirm | q cancel"
    header = 54
    polygon = []
    index = 0
    manual = not candidates

    def mouse(event, x, y, flags, param):
        nonlocal index
        y -= header
        if not (0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if manual:
                polygon.append((x, y))
            else:
                index = next((i for i, candidate in enumerate(candidates) if candidate["mask"][y, x]), index)
        elif event == cv2.EVENT_RBUTTONDOWN and manual and polygon:
            polygon.pop()

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            image = rgb.copy() if manual else np.asarray(overlay(rgb, candidates[index], index)).copy()
            if polygon:
                cv2.polylines(image, [np.asarray(polygon, dtype=np.int32)], len(polygon) >= 3, (255, 220, 30), 2)
            canvas = np.zeros((rgb.shape[0] + header, rgb.shape[1], 3), dtype=np.uint8)
            canvas[header:] = image[..., ::-1]
            cv2.putText(canvas, "Polygon: left click / right undo" if manual else "Candidate: [ / ] or click", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(canvas, "Enter confirms; p draws polygon; q cancels", (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt("Initial scene selection cancelled")
            if key == ord("p"):
                manual = True
                polygon.clear()
            elif not manual and key in (ord("["), ord("]")):
                index = (index + (1 if key == ord("]") else -1)) % len(candidates)
            elif not manual and ord("1") <= key <= ord("9") and key - ord("1") < len(candidates):
                index = key - ord("1")
            elif key in (10, 13):
                if manual:
                    if len(polygon) < 3:
                        continue
                    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
                    mask = mask.astype(bool)
                else:
                    mask = candidates[index]["mask"].copy()
                return mask, None if manual else index
    finally:
        cv2.destroyWindow(title)


def choose_masks(objects, frame, args, output, segmenter, maps):
    import numpy as np

    from ..io import load_initial_selection, write_confirmed_selection, write_json
    from ..rgbd import rectify_frame, save_frame

    masks, scene_objects = {}, []
    for name, item in objects.items():
        if item.get("initial_selection"):
            masks[name] = load_initial_selection(item["initial_selection"], frame, maps)
            selected = item["initial_selection"]
        else:
            candidates = segmenter.predict(
                frame.rgb, item["prompt"], box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                frame_key=dict(source_id=frame.source_id, row_index=frame.row_index,
                               timestamp_ns=frame.timestamp_ns, clock=frame.clock),
            ) if item.get("prompt") else []
            while True:
                mask, index = select_mask(frame.rgb, candidates, name)
                _, depth, _, rectified_mask = rectify_frame(frame, mask, maps=maps)
                if np.count_nonzero(rectified_mask & (depth > 0)) >= 4:
                    break
                print(f"{name}: mask 至少需要四个有效深度像素，请重新选择。", flush=True)
            masks[name] = rectified_mask
            selected = output / "initial_selections" / name
            selected.mkdir(parents=True)
            save_frame(selected / "source_rgbd.npz", frame)
            candidate = candidates[index] if index is not None else None
            write_confirmed_selection(selected, mask, dict(
                selection="gui_confirmed", selected_instance=index,
                mask_grid="original_rgb_grid; not_automatically_rectified",
                source_id=frame.source_id, timestamp_ns=frame.timestamp_ns, clock=frame.clock,
                prompt=item.get("prompt"),
                detection_score=candidate["score"] if candidate else None,
                mask_score=candidate["mask_score"] if candidate else None,
                segmentation_model=segmenter.provenance,
            ), mask_file="mask.npy")
        scene_item = item.copy()
        for key, path in (("asset", item.get("asset")), ("initial_selection", selected)):
            if path is not None:
                scene_item[key] = str(Path(path).expanduser().resolve())
        scene_objects.append(scene_item)
    write_json(output / "scene.json", {"objects": scene_objects})
    return masks
