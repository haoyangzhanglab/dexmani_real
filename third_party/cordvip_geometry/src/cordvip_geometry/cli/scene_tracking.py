"""确认 Raw 首帧实例，顺序处理每行并保存可独立查看的离线结果。"""

import itertools
import json
import logging
import math
from pathlib import Path

from .arguments import parse_tracking_args
from .scene_initialization import choose_masks, load_scene

logger = logging.getLogger(__name__)


def run(args):
    objects = load_scene(args.scene, args.workspace, headless=args.headless) if args.scene else {
        "object": dict(
            id="object", role="object", asset=args.asset,
            initial_selection=args.initial_selection, prompt=args.object_prompt,
        )
    }
    need_pose = any(item.get("asset") for item in objects.values())
    need_grounding = any(
        item.get("prompt") and not item.get("initial_selection") for item in objects.values()
    )
    if need_pose and (
        not math.isfinite(args.depth_tolerance_m) or args.depth_tolerance_m <= 0
        or args.min_matched_pixels < 1 or not 0 < args.min_matched_fraction <= 1
    ):
        raise ValueError("位姿深度容差、匹配像素数必须为正，匹配比例必须位于 (0,1]")
    if need_grounding and not (0 < args.box_threshold < 1 and 0 < args.text_threshold < 1):
        raise ValueError("首帧检测阈值必须位于 (0,1)")
    if args.check:
        from ..sam2_config import check_sam_environment
        from .check_tracking import check_tracking_environment, print_tracking_report

        report = check_tracking_environment(
            args.foundationpose_root, need_pose=need_pose, need_gui=False,
        )
        report["checks"].extend(check_sam_environment(args, require_grounding=need_grounding))
        report["ok"] = all(item["ok"] for item in report["checks"])
        print_tracking_report(report)
        return 0 if report["ok"] else 2

    from dexmani_real.data.raw import RawReader

    from ..geometry import RectificationMaps
    from ..integrations.dexmani import RawHandCloudProvider, iter_raw_rows
    from ..io import load_asset, new_output, write_json, write_scene_frame
    from ..rgbd import rectify_frame
    from ..scene_tracking import SceneTracker, scene_arrays
    from ..segmentation import GroundedSAM2Segmenter

    protected = [args.episode, args.scene]
    for item in objects.values():
        protected.extend((item.get("asset"), item.get("initial_selection")))
    output = new_output(args.output, workspace=args.workspace, protected=protected)
    (output / "frames").mkdir()
    print(f"Scene output: {output}", flush=True)
    reader = segmenter = tracker = records = None
    failure = None
    cleanup_errors = []
    processed = source_rows = 0
    try:
        reader = RawReader(args.episode)
        camera = reader.read_camera()
        source_rows = reader.num_frames
        rows = iter_raw_rows(reader, camera)
        first = next(rows, None)
        if first is None:
            raise ValueError("Raw episode contains no RGB-D rows")
        assets, asset_metadata = {}, {}
        for name, item in objects.items():
            if item.get("asset"):
                mesh, points, normals, metadata = load_asset(item["asset"])
                assets[name], asset_metadata[name] = (mesh, points, normals), metadata
        hand_provider = RawHandCloudProvider(reader, camera, num_points=args.hand_points) if args.with_hand else None
        palette = [(50, 220, 230), (220, 90, 230), (80, 220, 100), (240, 220, 70)]
        write_json(output / "run.json", dict(
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            assets=asset_metadata,
            colors_rgb={name: palette[index % len(palette)] for index, name in enumerate(objects)},
            units="m", coordinate_frame="color_camera",
            processing="all Raw rows in order, unpaced; max_frames explicitly limits a partial run",
            source_clock="original source timestamps; row association does not prove physical synchronization",
            observed_semantics="semantic mask + measured depth, independent of mesh; segmentation may be wrong",
            model_semantics="fixed generated surface + pose; depth support participates in pose validity",
            hand=hand_provider.metadata if hand_provider else None,
        ))
        segmenter = GroundedSAM2Segmenter(
            args.sam_python, args.sam_root, args.sam_checkpoint, sam_config=args.sam_config,
            dino_weights=args.dino_weights, log_path=output / "sam2_worker.log",
        )
        segmenter.start()
        initial_masks = choose_masks(objects, first[0], args, output, segmenter)
        tracker = SceneTracker(
            segmenter, assets, foundationpose_root=args.foundationpose_root,
            debug_dir=output / "foundationpose", pixel_stride=args.pixel_stride,
            depth_tolerance_m=args.depth_tolerance_m, min_matched_pixels=args.min_matched_pixels,
            min_matched_fraction=args.min_matched_fraction,
        )
        records = (output / "records.jsonl").open("x", encoding="utf-8")
        maps = RectificationMaps()
        for frame, timing in itertools.chain((first,), rows):
            rgb, depth, K = rectify_frame(frame, maps=maps)[:3]
            current = tracker.step(frame, rgb, depth, K, initial_masks=initial_masks if processed == 0 else None)
            hand = hand_provider.for_row(frame, timing) if hand_provider else None
            write_scene_frame(output / "frames" / f"{frame.row_index:06d}.npz", scene_arrays(current, hand))
            record = dict(
                source_id=frame.source_id, row_index=frame.row_index,
                timestamp_ns=frame.timestamp_ns, clock=frame.clock,
                operation=current["operation"],
                objects={name: dict(
                    state=item["state"], error=item["error"], pose_valid=item["pose_valid"],
                    observed_valid=item["observed_valid"], diagnostics=item["cloud"].diagnostics,
                    diagnostic_T_camera_object=(
                        item["diagnostic_pose"].tolist() if item["diagnostic_pose"] is not None else None
                    ),
                ) for name, item in current["objects"].items()},
            )
            records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            processed += 1
            if args.max_frames is not None and processed >= args.max_frames and processed < source_rows:
                break
        write_json(output / "models.json", dict(
            segmentation=segmenter.provenance,
            poses={name: tracker.provenance for name, tracker in tracker.trackers.items()},
        ))
    except BaseException as exc:
        failure = exc
    finally:
        for name, resource in (
            ("tracker", tracker), ("segmenter", segmenter if tracker is None else None),
            ("records", records), ("reader", reader),
        ):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    logger.exception("scene %s close failed", name)
                    cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
                    failure = failure or exc
        try:
            write_json(output / "result.json", dict(
                status=("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed") if failure else (
                    "completed" if processed == source_rows else "partial"
                ),
                error=f"{type(failure).__name__}: {failure}" if failure else None,
                cleanup_errors=cleanup_errors, processed_frames=processed, source_frames=source_rows,
            ))
        except BaseException as exc:
            logger.exception("scene result write failed")
            failure = failure or exc
    if failure is not None:
        raise failure
    print(f"Processed {processed}/{source_rows} Raw rows: {output}", flush=True)
    return 0


def main(argv=None, *, workspace=None):
    try:
        return run(parse_tracking_args(argv, workspace=workspace))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
