#!/usr/bin/env python3
"""从 RGB-D 视频流实时估计物体位姿，显示逐帧实测点云与模型点云；默认使用 RealSense。"""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import math
from pathlib import Path
# Mesa-enabled native imports can load system Expat before the model worker's
# XML dependencies. Bind Python's matching library before importing cv2/Torch.
import pyexpat
import sys
import time

from .check_tracking import check_tracking_environment, print_tracking_report


def parse_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument('--asset', type=Path, help='reconstruct_object.py 或 prepare 生成的米制物体资产目录')
    parser.add_argument('--scene', type=Path, help='多物体 JSON：objects 中逐个声明 id/role/asset/initial_selection/prompt')
    parser.add_argument('--all-frames', action='store_true', help='多物体离线模式逐帧处理，不按回放速率丢帧或等待')
    parser.add_argument('--max-frames', type=int, help='多物体模式最多处理的帧数；用于显式的部分验收')
    parser.add_argument('--foundationpose-root', type=Path, default=workspace / 'third_party/FoundationPose')
    parser.add_argument('--object-prompt', help='Grounded-SAM-2 文本目标，例如 "red mug."；省略时手动圈选')
    parser.add_argument('--sam-root', type=Path, default=workspace / 'third_party/Grounded-SAM-2')
    parser.add_argument('--dino-weights', type=Path,
                        default=workspace / 'reports/cordvip_models/grounding-dino-base')
    parser.add_argument('--sam-checkpoint', type=Path,
                        help='本地权重；多物体默认 SAM2.1 small，单物体初始化默认 large')
    parser.add_argument('--sam-config', help='与 SAM checkpoint 配套的 config')
    parser.add_argument('--box-threshold', type=float, default=.3)
    parser.add_argument('--text-threshold', type=float, default=.25)
    parser.add_argument('--check', action='store_true', help='只检查环境，不加载模型、创建窗口或连接相机')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--episode', type=Path, help='用只读 Raw 替代相机，按 replay-fps 演示异步流程')
    source.add_argument('--rgbd-dir', type=Path, help='用 RGB-D NPZ 替代相机')
    parser.add_argument('--replay-fps', type=float, default=30)
    parser.add_argument('--serial', help='RealSense serial；省略时由 SDK 选择相机')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--no-3d', action='store_true', help='仅显示双 RGB 窗口；仍生成两类物体点云')
    parser.add_argument('--initial-selection', type=Path,
                        help='复用已确认的 select 目录；必须与离线源首帧完全对应')
    parser.add_argument('--headless', action='store_true', help='无窗口离线回放；需要 --initial-selection')
    parser.add_argument('--with-hand', action='store_true', help='读取同一 Raw 行的实测关节，附加 XHand FK 点云')
    parser.add_argument('--experiment-config', type=Path, help='手安装变换使用的当前实验配置 YAML')
    parser.add_argument('--hand-points', type=int, default=1024, help='固定手表面采样点数')
    parser.add_argument('--output', type=Path, help='reports/ 下的新输出目录')
    parser.add_argument('--save-inputs', action='store_true', help='保存所有送入推理的源 RGB-D；注册帧总是保存')
    parser.add_argument('--save-frames', action='store_true', help='逐次保存实际处理结果到 frames/；不补齐跳过的源帧')
    parser.add_argument('--save-video', action='store_true', help='保存处理帧叠加视频；播放率为 replay-fps')
    parser.add_argument('--max-age-ms', type=float, default=500, help='源帧最大有效年龄，毫秒')
    parser.add_argument('--pixel-stride', type=int, default=2, help='物体实测点云的像素采样步长')
    parser.add_argument('--depth-tolerance-m', type=float, default=.015)
    parser.add_argument('--min-matched-pixels', type=int, default=30)
    parser.add_argument('--min-matched-fraction', type=float, default=.15)
    args = parser.parse_args(argv)
    if args.sam_checkpoint is None:
        size = 'small' if args.scene else 'large'
        args.sam_checkpoint = workspace / f'reports/cordvip_models/sam2.1_hiera_{size}.pt'
    if args.sam_config is None:
        size = 's' if args.scene else 'l'
        args.sam_config = f'configs/sam2.1/sam2.1_hiera_{size}.yaml'
    if not args.check and args.asset is None and args.scene is None:
        parser.error('追踪需要 --asset 或 --scene；只检查环境可用 --check')
    if args.scene and (args.asset or args.initial_selection or args.object_prompt):
        parser.error('--scene 内逐个指定资产、选择和 prompt，不能混用单物体参数')
    if (args.all_frames or args.max_frames is not None) and args.scene is None:
        parser.error('--all-frames/--max-frames 仅用于 --scene')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames 必须为正整数')
    if args.serial and (args.episode or args.rgbd_dir):
        parser.error('--serial 只用于实时相机')
    offline = args.episode is not None or args.rgbd_dir is not None
    if args.all_frames and not offline:
        parser.error('--all-frames 仅用于离线源')
    if args.initial_selection is not None and not offline:
        parser.error('--initial-selection 仅用于离线源，不能将历史 mask 用于新的相机帧')
    if args.headless and (not offline or (args.initial_selection is None and args.scene is None)):
        parser.error('--headless 需要离线源和 --initial-selection')
    if args.with_hand and args.episode is None:
        parser.error('--with-hand 需要 --episode，以同一 Raw 行的实测关节生成手点')
    if args.experiment_config is not None and not args.with_hand:
        parser.error('--experiment-config 需要 --with-hand')
    if args.object_prompt is not None and not args.object_prompt.strip():
        parser.error('--object-prompt 不能为空')
    for name in ('box_threshold', 'text_threshold'):
        if not math.isfinite(getattr(args, name)) or not 0 < getattr(args, name) < 1:
            parser.error(f'{name} 必须位于 (0,1)')
    for name in ('width', 'height', 'fps', 'pixel_stride', 'min_matched_pixels', 'hand_points'):
        if getattr(args, name) < 1:
            parser.error(f'{name} 必须为正整数')
    for name in ('replay_fps', 'max_age_ms', 'depth_tolerance_m'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} 必须为有限正数')
    if not 0 < args.min_matched_fraction <= 1:
        parser.error('--min-matched-fraction 必须位于 (0,1]')
    return args


def check_segmentation_environment(args, *, need_grounding=True):
    """Check local GS2 inputs without importing model modules or opening devices."""
    checks = []

    def file_check(name, path):
        path = Path(path).expanduser().resolve()
        try:
            ok = path.is_file() and path.stat().st_size > 0
            detail = str(path) if ok else f'缺失或为空：{path}'
            if ok and path.suffix in {'.pt', '.safetensors'}:
                with path.open('rb') as stream:
                    pointer = stream.read(80).startswith(b'version https://git-lfs.github.com/spec/')
                if pointer or Path(str(path) + '.aria2').exists():
                    ok, detail = False, f'权重尚未下载完成：{path}'
        except OSError as error:
            ok, detail = False, str(error)
        checks.append(dict(name=name, ok=ok, detail=detail))

    sam_package = args.sam_root.expanduser().resolve() / 'sam2'
    source_files = ('build_sam.py', 'sam2_video_predictor.py')
    if need_grounding:
        source_files += ('sam2_image_predictor.py',)
    for relative in source_files:
        file_check(f'SAM 2 源码 {relative}', sam_package / relative)
    sam_config = (sam_package / args.sam_config).resolve()
    if not sam_config.is_relative_to(sam_package):
        checks.append(dict(name='SAM 2 config', ok=False, detail='config 必须位于源码的 sam2/ 内'))
    else:
        file_check('SAM 2 config', sam_config)
    file_check('SAM 2.1 本地权重', args.sam_checkpoint)
    modules = ['torch', 'torchvision', 'hydra', 'iopath', 'omegaconf', 'numpy', 'PIL', 'tqdm']
    if need_grounding:
        dino = args.dino_weights.expanduser().resolve()
        for name in ('config.json', 'preprocessor_config.json'):
            file_check(f'Grounding DINO {name}', dino / name)
        tokenizer = next((dino / name for name in ('tokenizer.json', 'vocab.txt')
                          if (dino / name).is_file()), dino / 'tokenizer.json')
        file_check('Grounding DINO tokenizer', tokenizer)
        checkpoints = sorted(dino.glob('*.safetensors'))
        if not checkpoints:
            checks.append(dict(name='Grounding DINO 本地权重', ok=False, detail=f'缺少 safetensors：{dino}'))
        for path in checkpoints:
            file_check(f'Grounding DINO {path.name}', path)
        modules.extend(('transformers', 'safetensors', 'tokenizers', 'huggingface_hub'))
    for name in modules:
        try:
            spec = importlib.util.find_spec(name)
            ok, detail = spec is not None, (spec.origin or 'namespace package') if spec else '当前环境缺少模块'
        except (ImportError, ValueError, OSError) as error:
            ok, detail = False, str(error)
        checks.append(dict(name=f'分割依赖 {name}', ok=ok, detail=detail))
    return checks


def rectify_frame(frame, geometry, np):
    model = frame.distortion_model.removeprefix('distortion.')
    coefficients = np.asarray(frame.distortion_coeffs)
    if model not in {'none', 'brown_conrady'}:
        raise ValueError(f'Unsupported distortion model: {frame.distortion_model}')
    if model == 'none' and np.any(coefficients != 0):
        raise ValueError('none distortion requires zero coefficients')
    rgb, depth_m, K, _ = geometry.rectify_rgbd(frame.rgb, frame.depth_m, frame.K, coefficients)
    return rgb, depth_m, K


def load_initial_selection(directory, frame, geometry, np, cv2):
    """Reuse only an explicitly selected mask for the exact original RGB-D frame."""
    from ..io import _mask, _sha256

    directory = directory.expanduser().resolve()
    metadata = json.loads((directory / 'selection.json').read_text())
    if metadata.get('selection') not in {'explicit_headless_instance', 'gui_confirmed'}:
        raise ValueError('initial-selection must contain an explicitly confirmed selection')
    if metadata.get('mask_grid') != 'original_rgb_grid; not_automatically_rectified':
        raise ValueError('initial-selection mask must use the original source RGB grid')
    expected = dict(rgb=frame.rgb, depth_m=frame.depth_m, K=frame.K,
                    distortion_model=frame.distortion_model, distortion_coeffs=frame.distortion_coeffs,
                    source_id=frame.source_id, timestamp_ns=frame.timestamp_ns, clock=frame.clock)
    with np.load(directory / 'source_rgbd.npz', allow_pickle=False) as saved:
        for key, value in expected.items():
            if key not in saved.files or not np.array_equal(saved[key], np.asarray(value)):
                raise ValueError(f'initial-selection {key} differs from the actual first source frame')
        if 'row_index' in saved.files and not np.array_equal(saved['row_index'], frame.row_index):
            raise ValueError('initial-selection row_index differs from the first Raw row')
    for key in ('source_id', 'timestamp_ns', 'clock'):
        if metadata.get(key) != expected[key]:
            raise ValueError(f'initial-selection metadata {key} differs from the source frame')
    mask_path = (directory / metadata['mask_file']).resolve()
    if not mask_path.is_relative_to(directory) or _sha256(mask_path) != metadata['mask_sha256']:
        raise ValueError('initial-selection mask path/hash does not match the confirmed selection')
    mask = _mask(mask_path, frame.depth_m.shape, np, cv2)
    # select writes both formats; reject an edited convenience NPY as well.
    if (directory / 'mask.npy').exists():
        if not np.array_equal(mask, _mask(directory / 'mask.npy', frame.depth_m.shape, np, cv2)):
            raise ValueError('initial-selection mask.npy differs from the confirmed mask')
    rgb, depth, K, mask = geometry.rectify_rgbd(
        frame.rgb, frame.depth_m, frame.K, frame.distortion_coeffs, mask=mask)
    if np.count_nonzero(mask & (depth > 0)) < 4:
        raise ValueError('initial-selection has fewer than four valid rectified depth samples')
    return (rgb, depth, K), mask, dict(
        method='reused_confirmed_selection', directory=str(directory),
        selection_sha256=_sha256(directory / 'selection.json'),
        mask_sha256=metadata['mask_sha256'], source_check='exact_rgb_depth_K_distortion_identity_time_clock',
        selected_instance=metadata['selected_instance'],
    )


def source_age_ms(result, now_ns, *, live):
    # Offline clocks may belong to a different process/boot. Never subtract them
    # from this process's monotonic clock or call replay timing camera latency.
    if result is None or not live:
        return None
    return (now_ns - result.frame.timestamp_ns) / 1e6


def result_state(mode, result, now_ns, *, live, max_age_ms):
    if mode != 'TRACKING' or result is None:
        return mode, False
    age = source_age_ms(result, now_ns, live=live)
    if age is not None and not 0 <= age <= max_age_ms:
        return 'STALE', False
    return ('TRACKING' if live else 'REPLAY'), True


def publish_latest(path, *, state, valid, result, cloud, model_points, max_age_ms, np,
                   hand=None, model_normals=None):
    """One atomic snapshot; invalid states contain no usable pose or cloud.

    Readers must check valid AND the original timestamp against their current
    monotonic clock. A killed process cannot publish its own invalidation.
    """
    from ..observation import build_observation, write_observation

    payload = build_observation(
        state=state, valid=valid, result=result, cloud=cloud, model_points=model_points,
        max_age_ms=max_age_ms, hand=hand, model_normals=model_normals,
    )
    write_observation(path, payload)


def result_overlay(result, cloud, model_points, cv2, np, *, hand_points=None):
    from ..viewer import project_points

    canvas = result.rgb.copy()
    mask = cloud.observed_mask
    canvas[mask] = (canvas[mask].astype(float) * .3 + np.array([40, 240, 70]) * .7).astype(np.uint8)
    pixels, good = project_points(model_points, result.K)
    h, w = canvas.shape[:2]
    good &= (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    for x, y in np.rint(pixels[good]).astype(int):
        cv2.circle(canvas, (int(x), int(y)), 1, (0, 200, 255), -1)
    if hand_points is not None:
        pixels, good = project_points(hand_points, result.K)
        good &= (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        for x, y in np.rint(pixels[good]).astype(int):
            cv2.circle(canvas, (int(x), int(y)), 1, (255, 120, 40), -1)
    axes = np.array([[0, 0, 0], [.04, 0, 0], [0, .04, 0], [0, 0, .04]])
    axes = axes @ result.pose[:3, :3].T + result.pose[:3, 3]
    pixels, good = project_points(axes, result.K)
    if good.all() and np.abs(pixels).max() < 1e6:
        pixels = np.rint(pixels).astype(int)
        for i, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), 1):
            cv2.line(canvas, tuple(pixels[0]), tuple(pixels[i]), color, 2)
    return canvas


def run(args):
    if args.scene is not None:
        from .scene_tracking import run_scene
        return run_scene(args)
    live = not (args.episode or args.rgbd_dir)
    report = check_tracking_environment(args.foundationpose_root, need_camera=live,
                                        need_gui=not args.check and not args.headless)
    if args.object_prompt and not args.headless:
        report['checks'].extend(check_segmentation_environment(args))
        report['ok'] = all(item['ok'] for item in report['checks'])
    print_tracking_report(report)
    if args.check:
        return 0 if report['ok'] else 2
    if not report['ok']:
        return 2

    # None of the heavy libraries or device code are loaded by --help / --check.
    import cv2
    import numpy as np
    from .. import geometry
    from ..foundationpose import FoundationPoseTracker
    from ..object_cloud import ObjectCloudProjector
    from ..io import _asset, _new_output, _save_input, load_asset_normals
    from ..live_workers import LatestFrameReader, PoseWorker
    from ..segmentation_worker import SegmentationWorker
    from ..rgbd import iter_rgbd_directory
    from ..integrations.dexmani import iter_live_realsense, iter_raw_episode, RawHandCloudProvider

    mesh, points_local, asset_metadata = _asset(args.asset, geometry)
    normals_local = load_asset_normals(args.asset, points_local)
    hand_provider = (RawHandCloudProvider(args.episode, config_path=args.experiment_config,
                                          num_points=args.hand_points) if args.with_hand else None)
    projector = ObjectCloudProjector(mesh, pixel_stride=args.pixel_stride,
                                    depth_tolerance_m=args.depth_tolerance_m)
    output = _new_output(args.output, workspace=args.workspace,
                         protected=(args.episode, args.rgbd_dir, args.asset, args.initial_selection))
    (output / 'inputs').mkdir()
    if args.save_frames:
        (output / 'frames').mkdir()
    print(f'Output: {output}', flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output / 'run.json').write_text(json.dumps(dict(
        config=config, asset=asset_metadata, environment_check=report,
        coordinate_frame='color_camera', units='m',
        pose_semantics='T_camera_object maps original metric mesh coordinates into the aligned color camera',
        observed_cloud='RGB-D samples near the first rendered mesh surface; geometric filter, not semantic segmentation',
        quality='depth consistency is not calibrated confidence or ground truth',
        replay=not live, source_policy='latest frame only; no inference backlog',
        replay_initialization='first source frame held until successful registration',
        hand=hand_provider.metadata if hand_provider is not None else None,
        frames_semantics='processed source frames only; no replacement or interpolation of skipped rows',
        video=dict(enabled=args.save_video, fps=args.replay_fps,
                   content='processed frames only; playback timing is not original source timing'),
    ), indent=2, ensure_ascii=False))

    if args.episode:
        source_factory = lambda: iter_raw_episode(args.episode)
    elif args.rgbd_dir:
        source_factory = lambda: iter_rgbd_directory(args.rgbd_dir)
    else:
        source_factory = lambda: iter_live_realsense(serial=args.serial, width=args.width,
                                                   height=args.height, fps=args.fps)
    viewer = worker = reader = segmenter = video = None
    current = cloud = model_points = model_normals = hand = overlay = None
    source_snapshot = latest_frame = latest_rectified = selection = None
    selection_generation = 0
    segmentation_pending = False
    selection_details = None
    mode = 'LOADING'
    message = 'Loading FoundationPose; camera starts only after the estimator is ready.'
    seen_version = 0
    last_submitted_id = None
    publish_key = None
    input_index = processed = 0
    rate_start_ns = time.monotonic_ns()
    rate_start_version = 0
    capture_fps = 0.
    result_times = deque(maxlen=30)
    failure = None
    shutdown = {}
    completed_replay = False
    initial_selection_pending = args.initial_selection is not None
    records = (output / 'records.jsonl').open('x')

    def log(event, **values):
        records.write(json.dumps(dict(event=event, host_ns=time.monotonic_ns(), **values),
                                 ensure_ascii=False, allow_nan=False) + '\n')
        records.flush()

    def submit(frame, rectified, mask=None, mask_metadata=None):
        nonlocal input_index, last_submitted_id
        rgb, depth_m, K = rectified
        if not worker.submit(frame, rgb, depth_m, K, mask=mask):
            return False
        # Each mask is saved with its exact original source frame and rectified K.
        path = output / 'inputs' / f'{input_index:06d}.npz'
        if args.save_inputs or mask is not None:
            _save_input(path, frame, np)
        mask_path = None
        if mask is not None:
            mask_path = f'mask_{input_index:06d}_rectified.npy'
            np.save(output / mask_path, mask, allow_pickle=False)
        log('submit', index=input_index, source_id=frame.source_id,
            timestamp_ns=frame.timestamp_ns, clock=frame.clock, mask_file=mask_path,
            input_file=str(path.relative_to(output)) if args.save_inputs or mask is not None else None,
            operation='register' if mask is not None else 'track', K=K.tolist(),
            mask_selection=mask_metadata,
            generation=worker.snapshot().generation)
        input_index += 1
        last_submitted_id = (frame.source_id, frame.row_index)
        return True

    try:
        publish_latest(output / 'latest.npz', state=mode, valid=False, result=None, cloud=None,
                       model_points=None, max_age_ms=args.max_age_ms, np=np)
        if not args.headless:
            from ..live_viewer import LiveViewer
            viewer = LiveViewer(show_3d=not args.no_3d)
        if args.object_prompt and not args.headless:
            segmenter = SegmentationWorker(
                output / 'segmentation', dino_weights=args.dino_weights,
                sam_root=args.sam_root, sam_checkpoint=args.sam_checkpoint,
                sam_config=args.sam_config, box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
        worker = PoseWorker(lambda: FoundationPoseTracker(
            args.foundationpose_root, mesh, debug_dir=output / 'foundationpose')).start()
        while True:
            tick_ns = time.monotonic_ns()
            worker_status = worker.snapshot()
            if worker_status.error:
                raise RuntimeError(worker_status.error)
            if mode == 'LOADING' and worker_status.ready:
                reader = LatestFrameReader(source_factory, replay_interval_s=0 if live else 1 / args.replay_fps,
                                           wait_on_first=not live).start()
                mode = 'PREVIEW'
                rate_start_ns = time.monotonic_ns()
                message = ('Press i/r to detect the prompted object, p for polygon. Keep object still until tracking.'
                           if segmenter else 'Press i/r, outline the target, Enter to register. Keep object still until tracking.')
            if reader is not None:
                source_snapshot = reader.snapshot()
                if source_snapshot.error:
                    raise RuntimeError(source_snapshot.error)
                if source_snapshot.version != seen_version:
                    seen_version = source_snapshot.version
                    latest_frame = source_snapshot.frame
                    latest_rectified = rectify_frame(latest_frame, geometry, np)
                if tick_ns - rate_start_ns >= 1_000_000_000:
                    capture_fps = (seen_version - rate_start_version) * 1e9 / (tick_ns - rate_start_ns)
                    rate_start_ns, rate_start_version = tick_ns, seen_version
                if source_snapshot.finished and live:
                    raise RuntimeError('Live camera stream ended')
                if source_snapshot.finished and latest_frame is None:
                    raise ValueError('Offline RGB-D source contains no frames')
            if initial_selection_pending and latest_frame is not None:
                rectified, mask, details = load_initial_selection(
                    args.initial_selection, latest_frame, geometry, np, cv2)
                if submit(latest_frame, rectified, mask=mask, mask_metadata=details):
                    initial_selection_pending = False
                    mode = 'REGISTERING'
                    message = 'Registering the verified first-frame selection.'

            result = worker.poll()
            if result is not None:
                processed += 1
                current = result
                cloud = model_points = model_normals = hand = None
                overlay = result.rgb
                projection_ms = hand_ms = 0.
                if result.error:
                    mode = 'LOST'
                    message = result.error + ' | Press r to re-register.'
                    worker.reset()
                else:
                    geometry_start = time.monotonic_ns()
                    cloud = projector.project(result.rgb, result.depth_m, result.K, result.pose)
                    model_points = geometry.transform_points(points_local, result.pose)
                    model_normals = normals_local @ result.pose[:3, :3].T
                    projection_ms = (time.monotonic_ns() - geometry_start) / 1e6
                    if hand_provider is not None:
                        hand_start = time.monotonic_ns()
                        hand = hand_provider.for_frame(result.frame)
                        hand_ms = (time.monotonic_ns() - hand_start) / 1e6
                    if viewer is not None or args.save_video:
                        overlay = result_overlay(result, cloud, model_points, cv2, np,
                                                 hand_points=hand['points_camera'] if hand else None)
                    diagnostics = cloud.diagnostics
                    fraction = diagnostics['matched_fraction']
                    matched = diagnostics['matched_pixel_count']
                    if matched < args.min_matched_pixels or fraction is None or fraction < args.min_matched_fraction:
                        mode = 'LOST'
                        message = 'Depth/model mismatch; inspect scale, mask, occlusion. Press r to re-register.'
                        worker.reset()
                    elif mode != 'PAUSED':
                        mode = 'TRACKING'
                        message = ('Tracking latest frames. Green: measured object; cyan: mesh model; '
                                   'orange: FK hand.' if hand is not None else
                                   'Tracking latest frames. Green: measured object; cyan: mesh model.')
                        if not live and result.operation == 'register':
                            reader.resume()
                    result_times.append(result.completed_ns)
                log('result', generation=result.generation, source_id=result.frame.source_id,
                    timestamp_ns=result.frame.timestamp_ns, clock=result.frame.clock,
                    operation=result.operation, started_ns=result.started_ns, completed_ns=result.completed_ns,
                    inference_ms=(result.completed_ns - result.started_ns) / 1e6,
                    projection_ms=projection_ms, hand_ms=hand_ms, row_index=result.frame.row_index,
                    hand_timing_available=hand['timing_available'] if hand is not None else None,
                    hand_camera_skew_ms=hand['skew_ms'] if hand is not None else None,
                    source_age_ms=source_age_ms(result, time.monotonic_ns(), live=live),
                    T_camera_object=result.pose.tolist() if result.pose is not None else None,
                    diagnostics=cloud.diagnostics if cloud else {}, state=mode, error=result.error)
                saved_state, saved_valid = result_state(mode, result, time.monotonic_ns(),
                                                        live=live, max_age_ms=args.max_age_ms)
                if args.save_frames:
                    publish_latest(output / 'frames' / f'{processed - 1:06d}.npz',
                                   state=saved_state, valid=saved_valid, result=result, cloud=cloud,
                                   model_points=model_points, max_age_ms=args.max_age_ms, np=np,
                                   hand=hand, model_normals=model_normals)
                if args.save_video:
                    panel = cv2.cvtColor(np.hstack((result.rgb, overlay)), cv2.COLOR_RGB2BGR)
                    cv2.rectangle(panel, (0, 0), (panel.shape[1], 45), (0, 0, 0), -1)
                    cv2.putText(panel, f'{saved_state} | processed={processed} | source row={result.frame.row_index} | '
                                'green=depth cyan=model orange=hand', (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
                    if video is None:
                        video = cv2.VideoWriter(str(output / 'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                args.replay_fps, (panel.shape[1], panel.shape[0]))
                        if not video.isOpened():
                            raise RuntimeError('Could not open overlay.mp4 writer')
                    video.write(panel)
                if args.headless and mode == 'LOST':
                    raise RuntimeError(message)

            if segmenter is not None:
                segmentation_result = segmenter.poll()
                if segmentation_result is not None:
                    log('segmentation_result', generation=segmentation_result.generation,
                        candidates=len(segmentation_result.candidates), error=segmentation_result.error,
                        directory=str(segmentation_result.directory.relative_to(output)))
                    if (mode == 'SEGMENTING' and selection is not None
                            and segmentation_result.generation == selection_generation):
                        mode = 'SELECTING'
                        viewer.set_candidates(segmentation_result.candidates)
                        selection_details['directory'] = str(segmentation_result.directory.relative_to(output))
                        if segmentation_result.error:
                            message = 'Segmentation failed; see segmentation process.log. Press p for polygon or r to retry.'
                        elif not segmentation_result.candidates:
                            message = 'No object detected. Press p for polygon or r to recapture; no pose was registered.'
                        else:
                            message = (f'{len(segmentation_result.candidates)} candidates. Select 1-9 / [ ] / click, '
                                       'then Enter; p for polygon. Keep object still.')
                # reset() revokes old poses immediately but a CUDA call can still
                # finish. Never overlap that inference with segmentation CUDA.
                if (segmentation_pending and selection is not None and not worker.snapshot().busy
                        and not segmenter.busy):
                    frame, rectified = selection
                    if segmenter.submit(rectified[0], args.object_prompt, generation=selection_generation,
                                        source_id=frame.source_id, timestamp_ns=frame.timestamp_ns):
                        segmentation_pending = False
                        log('segmentation_submit', generation=selection_generation,
                            source_id=frame.source_id, timestamp_ns=frame.timestamp_ns, prompt=args.object_prompt)
                        message = 'Grounded-SAM is segmenting the frozen source frame. UI remains responsive; p: polygon.'

            # Stale registration may be expensive, so continue to the newest frame
            # while withholding validity; do not auto-recover a failed registration.
            if (mode == 'TRACKING' and latest_frame is not None
                    and (latest_frame.source_id, latest_frame.row_index) != last_submitted_id):
                submit(latest_frame, latest_rectified)
            now_ns = time.monotonic_ns()
            state, valid = result_state(mode, current, now_ns, live=live, max_age_ms=args.max_age_ms)
            replay_finished = (
                not live and source_snapshot is not None and source_snapshot.finished
                and not worker.snapshot().busy and mode == 'TRACKING' and current is not None
                and (current.frame.source_id, current.frame.row_index)
                == (latest_frame.source_id, latest_frame.row_index)
            )
            key = (state, current.completed_ns if current else None)
            if key != publish_key:
                publish_latest(output / 'latest.npz', state=state, valid=valid, result=current, cloud=cloud,
                               model_points=model_points, max_age_ms=args.max_age_ms, np=np,
                               hand=hand, model_normals=model_normals)
                log('state', state=state, valid=valid)
                publish_key = key
            age = source_age_ms(current, now_ns, live=live)
            age_text = f'{age:.0f} ms' if age is not None else 'N/A (offline)' if not live else 'N/A'
            hz = (len(result_times) - 1) * 1e9 / (result_times[-1] - result_times[0]) if len(result_times) > 1 else 0
            inference_ms = f'{(current.completed_ns - current.started_ns) / 1e6:.0f}' if current else 'N/A'
            lines = [f'{state} | valid={valid} | camera={capture_fps:.1f} Hz | processed={hz:.1f} Hz | '
                     f'infer={inference_ms} ms | source age={age_text}',
                     message,
                     f'latest source: {latest_frame.source_id if latest_frame else "waiting"}',
                     f'pose source: {current.frame.source_id if current else "none"}']
            if cloud is not None:
                diag = cloud.diagnostics
                residual = diag['median_abs_depth_error_m']
                residual_text = f'{residual * 1000:.1f} mm' if residual is not None else 'N/A'
                lines.append(f'measured={len(cloud.observed_points_camera)} | model={len(model_points)} | '
                             f'match={diag["matched_fraction"] or 0:.3f} | median residual={residual_text} (not confidence)')
            if current is not None and current.pose is not None:
                pose = current.pose
                lines.append('t [m]: ' + np.array2string(pose[:3, 3], precision=4))
                lines.append('R rows: ' + ' / '.join(','.join(f'{x:.3f}' for x in row) for row in pose[:3, :3]))
            if hand is not None:
                timing = (f'source skew={hand["skew_ms"]:.1f} ms' if hand['timing_available']
                          else 'row-aligned replay; source synchronization unverified')
                lines.append(f'hand={len(hand["points_camera"])} | {timing}')
            button = -1
            if viewer is not None:
                button = viewer.update(latest_rectified[0] if latest_rectified else None, overlay,
                                       observed_points=cloud.observed_points_camera if cloud else None,
                                       model_points=model_points, pose=current.pose if current else None,
                                       hand_points=hand['points_camera'] if hand else None,
                                       lines=lines, result_valid=valid)
            if replay_finished:
                completed_replay = True
                log('replay_end', final_source_id=current.frame.source_id, final_row_index=current.frame.row_index)
                break
            if button == ord('q') or button == 27 and selection is None:
                break
            if button == 27 and selection is not None:
                if segmenter:
                    segmenter.cancel()
                selection_generation += 1
                segmentation_pending = False
                viewer.cancel_selection()
                selection = None
                mode = 'PREVIEW'
                message = 'Selection cancelled. Press i/r to select again.'
            elif button in (ord('i'), ord('r'), ord('p')) and latest_frame is not None:
                worker.reset()
                if segmenter:
                    segmenter.cancel()
                selection_generation += 1
                manual = button == ord('p') or segmenter is None
                # p edits this exact frozen frame; i/r explicitly captures a new one.
                if button != ord('p') or selection is None:
                    selection = (latest_frame, latest_rectified)
                viewer.begin_selection(selection[1][0], manual=manual)
                segmentation_pending = not manual
                selection_details = dict(method='polygon' if manual else 'grounded_sam2',
                                         prompt=args.object_prompt if not manual else None)
                mode = 'SELECTING' if manual else 'SEGMENTING'
                message = ('Outline target only. Keep object still during selection and registration.' if manual else
                           'Waiting for prior GPU work to finish, then segmenting the frozen frame. p: polygon.')
                result_times.clear()
            elif selection is not None and viewer.choose_candidate(button):
                message = f'Candidate {viewer.selected_candidate_index + 1} selected. Enter confirms; p draws a polygon.'
            elif button in (10, 13) and selection is not None:
                mask = viewer.selected_mask()
                if mode == 'SEGMENTING' or segmenter is not None and segmenter.busy:
                    message = 'Waiting for segmentation to release GPU resources. Enter again after it finishes.'
                elif mask is None:
                    message = 'Select a detection or press p and draw at least 3 polygon corners / 16 pixels.'
                elif np.count_nonzero(mask & (selection[1][1] > 0)) < 4:
                    message = 'Selection has fewer than 4 valid depths; recapture with r.'
                elif submit(*selection, mask=mask, mask_metadata={
                        **selection_details, 'candidate_index': viewer.selected_candidate_index}):
                    viewer.cancel_selection()
                    selection = None
                    mode = 'REGISTERING'
                    message = 'Registering frozen source frame. Keep object still; UI remains responsive.'
                else:
                    message = 'Previous GPU call is finishing; press Enter again when ready.'
            elif button == ord(' ') and mode in ('TRACKING', 'PAUSED'):
                mode = 'PAUSED' if mode == 'TRACKING' else 'TRACKING'
                message = 'Paused: output invalid. Space resumes; if object moved far, press r.' if mode == 'PAUSED' else 'Tracking resumed.'
                result_times.clear()
            elif button == ord('s'):
                viewer.screenshot(output / f'snapshot_{now_ns}.png')
                _, still_fresh = result_state(mode, current, time.monotonic_ns(),
                                              live=live, max_age_ms=args.max_age_ms)
                publish_latest(output / f'snapshot_{now_ns}.npz', state=state,
                               valid=valid and still_fresh, result=current, cloud=cloud,
                               model_points=model_points, max_age_ms=args.max_age_ms, np=np,
                               hand=hand, model_normals=model_normals)
                message = f'Screenshot and pose/cloud NPZ saved in {output.name}'
            # OpenCV/Open3D poll every tick; no sleep waits for inference or mask.
            time.sleep(.001 if args.headless else .008)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        print(failure, file=sys.stderr)
    finally:
        def cleanup_error(message):
            nonlocal failure
            failure = f'{failure}; {message}' if failure else message
            print(message, file=sys.stderr)

        def invalidate(state):
            try:
                publish_latest(output / 'latest.npz', state=state, valid=False,
                               result=current, cloud=cloud, model_points=model_points,
                               max_age_ms=args.max_age_ms, np=np, hand=hand, model_normals=model_normals)
            except Exception as error:
                cleanup_error(f'Could not invalidate latest.npz: {error}')
                # Do not leave a readable valid snapshot after a failed write.
                try:
                    (output / 'latest.npz').unlink(missing_ok=True)
                except OSError as remove_error:
                    cleanup_error(f'Could not remove old snapshot: {remove_error}')

        # Revoke output before shutdown. Signal both owners first; a camera SDK
        # read or CUDA operation can only release its resources on its own thread.
        invalidate('ERROR' if failure else 'STOPPING')
        if viewer is not None:
            try:
                viewer.screenshot(output / 'last_frame.png')
            except Exception as error:
                cleanup_error(f'Final screenshot unavailable: {error}')
        pending = {name: resource for name, resource in (('camera', reader), ('pose', worker), ('segmentation', segmenter))
                   if resource is not None}
        deadline = time.monotonic() + 60
        while pending:
            for name, resource in list(pending.items()):
                try:
                    status = resource.close(timeout_s=0)
                    shutdown[name] = dict(stopped=status.stopped, error=status.error)
                    if status.stopped:
                        pending.pop(name)
                        if status.error:
                            cleanup_error(f'{name} cleanup: {status.error}')
                except Exception as error:
                    pending.pop(name)
                    shutdown[name] = dict(stopped=False, error=str(error))
                    cleanup_error(f'{name} cleanup: {error}')
            if not pending:
                break
            if time.monotonic() >= deadline:
                break
            if viewer is not None:
                try:
                    button = viewer.update(latest_rectified[0] if latest_rectified else None, overlay,
                                           lines=['STOPPING | output invalid',
                                                  'Waiting for camera / GPU owner to finish; q forces incomplete shutdown.'],
                                           result_valid=False)
                    if button in (ord('q'), 27):
                        break
                except Exception as error:
                    cleanup_error(f'Shutdown window: {error}')
                    # Continue waiting without a window if the GUI was closed.
                    try:
                        viewer.close()
                    except Exception as close_error:
                        cleanup_error(f'Viewer close: {close_error}')
                    viewer = None
            time.sleep(.02)
        if segmenter is not None and segmenter.busy:
            # A child process would otherwise outlive a force-closed GUI. It owns
            # no hardware and may be killed without racing the pose owner's CUDA.
            try:
                status = segmenter.close(timeout_s=2, force=True)
                shutdown['segmentation'] = dict(stopped=status.stopped, error=status.error)
                if status.stopped:
                    pending.pop('segmentation', None)
            except Exception as error:
                cleanup_error(f'Segmentation forced shutdown: {error}')
        if pending:
            cleanup_error(f'Incomplete shutdown: {", ".join(pending)} owner still running; '
                          'resource cleanup is unconfirmed')
        if viewer is not None:
            try:
                viewer.close()
            except Exception as error:
                cleanup_error(f'Viewer close: {error}')
        if video is not None:
            try:
                video.release()
            except Exception as error:
                cleanup_error(f'Video close: {error}')
        try:
            records.close()
        except Exception as error:
            cleanup_error(f'Records close: {error}')
        invalidate('ERROR' if failure else 'STOPPED')
        (output / 'result.json').write_text(json.dumps(dict(
            status='failed' if failure else 'completed_replay' if completed_replay else 'stopped', error=failure,
            processed_frames=processed, submitted_frames=input_index,
            source_frames=seen_version, live_camera_requested=live, shutdown=shutdown,
        ), indent=2, ensure_ascii=False))
    return 1 if failure else 0


def main(argv=None, *, workspace=None):
    return run(parse_args(argv, workspace=workspace))


if __name__ == '__main__':
    raise SystemExit(main())
