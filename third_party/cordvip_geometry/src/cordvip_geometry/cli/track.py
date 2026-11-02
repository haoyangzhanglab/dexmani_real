"""确认 Raw 首帧实例，顺序保存每行位姿与几何。"""

import itertools
import logging
from pathlib import Path

from .arguments import parse_tracking_args
from .scene import choose_masks, load_scene

logger = logging.getLogger(__name__)


def run(args):
    from dexmani_real.data.raw import RawReader

    from ..geometry import RectificationMaps
    from ..io import new_output, write_json, write_npz
    from ..raw import RawHandCloud, iter_raw_rows
    from ..tracking import SceneTracker, scene_arrays

    objects = (
        load_scene(args.scene, headless=args.headless)
        if args.scene
        else {
            "object": {
                "id": "object",
                "role": "object",
                "asset": args.asset,
                "initial_selection": args.initial_selection,
                "prompt": args.prompt,
            }
        }
    )
    for item in objects.values():
        for key in ("asset", "initial_selection"):
            if item.get(key) is not None:
                item[key] = (args.workspace / Path(item[key]).expanduser()).resolve()
    protected = [args.episode, args.scene]
    for item in objects.values():
        protected.extend((item.get("asset"), item.get("initial_selection")))
    output = new_output(args.output, workspace=args.workspace, protected=protected)
    (output / "frames").mkdir()
    print(f"Scene output: {output}", flush=True)
    failure = None
    processed = source_rows = 0
    model_sources: dict[str, object] = {"segmentation": None, "poses": {}}
    try:
        with RawReader(args.episode) as reader:
            camera = reader.read_camera()
            source_rows = reader.num_frames
            rows = iter_raw_rows(reader, camera)
            first = next(rows, None)
            if first is None:
                raise ValueError("Raw episode contains no RGB-D rows")
            maps = RectificationMaps(
                first[0].depth_m.shape, first[0].K, first[0].distortion_coeffs
            )
            hand_source = (
                RawHandCloud(reader, camera, args.hand_points)
                if args.with_hand
                else None
            )
            with SceneTracker.open(
                {name: item.get("asset") for name, item in objects.items()},
                workspace=args.workspace,
                models_dir=args.models_dir,
                sam_python=args.sam_python,
                sam_checkpoint=args.sam_checkpoint,
                sam_config=args.sam_config,
                log_dir=output,
                pixel_stride=args.pixel_stride,
                depth_tolerance_m=args.depth_tolerance_m,
                min_matched_pixels=args.min_matched_pixels,
                min_matched_fraction=args.min_matched_fraction,
            ) as tracker:
                segmenter = tracker.segmenter
                model_sources["segmentation"] = segmenter.provenance
                model_sources["poses"] = {
                    name: item.provenance for name, item in tracker.trackers.items()
                }
                if tracker.assets:
                    (output / "model_templates").mkdir()
                    for name, (_, points, normals) in tracker.assets.items():
                        write_npz(
                            output / "model_templates" / f"{name}.npz",
                            {
                                "points_object": points,
                                "normals_object": normals,
                            },
                        )
                palette = [
                    (50, 220, 230),
                    (220, 90, 230),
                    (80, 220, 100),
                    (240, 220, 70),
                ]
                write_json(
                    output / "run.json",
                    {
                        "config": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                        },
                        "assets": tracker.asset_metadata,
                        "colors_rgb": {
                            name: palette[index % len(palette)]
                            for index, name in enumerate(objects)
                        },
                        "units": "m",
                        "coordinate_frame": "color_camera",
                        "hand": hand_source.metadata if hand_source else None,
                    },
                )
                initial_masks = choose_masks(
                    objects, first[0], args, output, segmenter, maps
                )
                write_json(output / "models.json", model_sources)
                for frame, timing in itertools.chain((first,), rows):
                    current = tracker.step(
                        frame,
                        initial_masks=initial_masks if processed == 0 else None,
                    )
                    if hand_source is not None:
                        current["hand"] = hand_source.for_row(frame, timing)
                    write_npz(
                        output / "frames" / f"{frame.row_index:06d}.npz",
                        scene_arrays(current),
                    )
                    processed += 1
                    if (
                        args.max_frames is not None
                        and processed >= args.max_frames
                        and processed < source_rows
                    ):
                        break
    except BaseException as error:
        failure = error
        raise
    finally:
        reports: dict[str, dict[str, object]] = {
            "result.json": {
                "status": (
                    "interrupted"
                    if isinstance(failure, KeyboardInterrupt)
                    else "failed"
                )
                if failure
                else ("completed" if processed == source_rows else "partial"),
                "error": f"{type(failure).__name__}: {failure}" if failure else None,
                "processed_frames": processed,
                "source_frames": source_rows,
            }
        }
        if not (output / "models.json").exists():
            reports["models.json"] = model_sources
        for name, report in reports.items():
            try:
                write_json(output / name, report)
            except BaseException:
                if failure is None:
                    raise
                logger.exception("scene report write also failed: %s", name)
    print(f"Processed {processed}/{source_rows} Raw rows: {output}", flush=True)
    return 0


def main(argv=None, *, workspace=None):
    try:
        return run(parse_tracking_args(argv, workspace=workspace))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
