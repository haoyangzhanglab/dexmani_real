"""确认 Raw 首帧实例，顺序处理每行并保存可独立查看的离线结果。"""

import itertools
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
    reader = segmenter = tracker = None
    failure = None
    cleanup_errors = []
    processed = source_rows = 0
    model_sources: dict[str, object] = dict(segmentation=None, poses={})
    try:
        reader = RawReader(args.episode)
        camera = reader.read_camera()
        source_rows = reader.num_frames
        rows = iter_raw_rows(reader, camera)
        first = next(rows, None)
        if first is None:
            raise ValueError("Raw episode contains no RGB-D rows")
        maps = RectificationMaps(first[0].depth_m.shape, first[0].K, first[0].distortion_coeffs)
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
        model_sources["segmentation"] = segmenter.provenance
        initial_masks = choose_masks(objects, first[0], args, output, segmenter, maps)
        tracker = SceneTracker(
            segmenter, assets, foundationpose_root=args.foundationpose_root,
            debug_dir=output / "foundationpose", pixel_stride=args.pixel_stride,
            depth_tolerance_m=args.depth_tolerance_m, min_matched_pixels=args.min_matched_pixels,
            min_matched_fraction=args.min_matched_fraction,
        )
        model_sources["poses"] = {name: item.provenance for name, item in tracker.trackers.items()}
        write_json(output / "models.json", model_sources)
        for frame, timing in itertools.chain((first,), rows):
            rgb, depth, _, _ = rectify_frame(frame, maps=maps)
            current = tracker.step(frame, rgb, depth, initial_masks=initial_masks if processed == 0 else None)
            hand = hand_provider.for_row(frame, timing) if hand_provider else None
            write_scene_frame(output / "frames" / f"{frame.row_index:06d}.npz", scene_arrays(current, hand))
            processed += 1
            if args.max_frames is not None and processed >= args.max_frames and processed < source_rows:
                break
    except BaseException as exc:
        failure = exc
    finally:
        # Initialization can fail after SAM is ready; preserve those model facts before closing.
        if failure is not None and not (output / "models.json").exists():
            try:
                write_json(output / "models.json", model_sources)
            except BaseException:
                logger.exception("scene model sources write also failed")
        for name, resource in (
            ("tracker", tracker), ("segmenter", segmenter), ("reader", reader),
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
