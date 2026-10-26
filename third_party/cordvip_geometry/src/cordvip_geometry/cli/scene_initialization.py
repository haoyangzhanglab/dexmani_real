"""场景声明和首帧实例确认；保留来源一致的选择产物。"""

import json
import re
from pathlib import Path


def load_scene(path, workspace, *, offline, headless):
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
        if item.get("initial_selection") and not offline:
            raise ValueError("Historical initial_selection cannot initialize a new camera frame")
        if headless and not item.get("initial_selection"):
            raise ValueError(f"{name}: headless scene requires an exact-source initial_selection")
        if item.get("prompt") is not None and (
            not isinstance(item["prompt"], str) or not item["prompt"].strip()
        ):
            raise ValueError("Scene prompt must be nonempty text")
        objects[name] = item
    return objects


def choose_masks(objects, frame, args, viewer, output, worker):
    import numpy as np

    from ..io import load_initial_selection, write_confirmed_selection, write_json
    from ..live_workers import wait_worker
    from ..rgbd import rectify_frame, save_frame

    masks, details, replay_objects = {}, {}, []
    try:
        for name, item in objects.items():
            candidates = []
            detection_metrics = None
            if item.get("initial_selection"):
                masks[name], details[name] = load_initial_selection(
                    item["initial_selection"], frame
                )
                replay_objects.append({**item, "initial_selection": str(item["initial_selection"])})
                continue
            # Confirm on the original RGB grid so the same source-verified selection can be reused.
            viewer.begin_selection(frame.rgb, manual=not item.get("prompt"))
            if item.get("prompt"):
                if not worker.submit(
                    "predict",
                    frame,
                    item["prompt"],
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                ):
                    raise RuntimeError("Scene worker was not ready for selection")

                def selecting():
                    key = viewer.update(
                        frame.rgb,
                        frame.rgb,
                        lines=[f"Grounded-SAM2: finding {name}"],
                        status_text="Text segmentation",
                        help_text="q: cancel",
                    )
                    if key in (27, ord("q")):
                        raise KeyboardInterrupt("Selection cancelled")

                result = wait_worker(worker, selecting)
                if result.frame is not frame or result.operation != "predict":
                    raise RuntimeError("Rejected mismatched detection result")
                candidates = result.value
                detection_metrics = dict(worker.segmenter.last_metrics)
                viewer.set_candidates(candidates)
                if not candidates:
                    viewer.use_polygon()
            while True:
                key = viewer.update(
                    frame.rgb,
                    frame.rgb,
                    lines=[f'Select {name} ({item["role"]}), then Enter. q cancels the scene.'],
                    status_text="Confirm every instance on the same frozen frame",
                    help_text="1-9 / [ ] / click: candidate | p: polygon | Enter: confirm | q: quit",
                )
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt("Scene selection cancelled")
                if key == ord("p"):
                    viewer.use_polygon()
                viewer.choose_candidate(key)
                if key in (10, 13):
                    mask = viewer.selected_mask()
                    if mask is not None and np.count_nonzero(mask & (frame.depth_m > 0)) >= 4:
                        _, selected_depth, _, selected_mask = rectify_frame(frame, mask)
                        if np.count_nonzero(selected_mask & (selected_depth > 0)) < 4:
                            continue
                        masks[name] = selected_mask
                        details[name] = dict(
                            method="gui_confirmed",
                            prompt=item.get("prompt"),
                            selected_instance=viewer.selected_candidate_index,
                            detection_metrics=detection_metrics,
                        )
                        selected_index = viewer.selected_candidate_index
                        candidate = (
                            candidates[selected_index] if selected_index is not None else None
                        )
                        scores = dict(
                            detection_score=candidate["score"] if candidate is not None else None,
                            mask_score=candidate["mask_score"] if candidate is not None else None,
                            score_kind="grounding_dino" if candidate is not None else None,
                            mask_score_kind="sam2_predicted_iou" if candidate is not None else None,
                            box_threshold=args.box_threshold,
                            text_threshold=args.text_threshold,
                        )
                        details[name].update(scores)
                        selected = output / "initial_selections" / name
                        selected.mkdir(parents=True)
                        save_frame(selected / "source_rgbd.npz", frame)
                        write_confirmed_selection(
                            selected,
                            mask,
                            dict(
                                selection="gui_confirmed",
                                selected_instance=viewer.selected_candidate_index,
                                mask_grid="original_rgb_grid; not_automatically_rectified",
                                source_id=frame.source_id,
                                timestamp_ns=frame.timestamp_ns,
                                clock=frame.clock,
                                prompt=item.get("prompt"),
                                detection_metrics=detection_metrics,
                                **scores,
                                segmentation_model=(
                                    worker.metadata["segmentation"] if item.get("prompt") else None
                                ),
                            ),
                            mask_file="mask.npy",
                        )
                        replay_objects.append({**item, "initial_selection": str(selected)})
                        break
            viewer.cancel_selection()
        save_frame(output / "initial_source_rgbd.npz", frame)
        for name, mask in masks.items():
            path = output / f"initial_{name}_rectified.npy"
            np.save(path, mask, allow_pickle=False)
        write_json(
            output / "initialization.json",
            dict(
                source_id=frame.source_id,
                timestamp_ns=frame.timestamp_ns,
                clock=frame.clock,
                row_index=frame.row_index,
                objects=details,
            ),
        )
        write_json(
            output / "replay_scene.json",
            {
                "objects": [
                    {k: str(v) if isinstance(v, Path) else v for k, v in item.items()}
                    for item in replay_objects
                ]
            },
        )
        return masks
    finally:
        if viewer is not None:
            viewer.cancel_selection()
