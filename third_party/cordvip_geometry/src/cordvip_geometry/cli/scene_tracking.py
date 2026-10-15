"""单物体与多物体 RGB-D 追踪；默认连接 RealSense，离线源只读。"""

import argparse
import json
import math
from pathlib import Path
# Bind Python's Expat before Mesa/native libraries load the system version.
import pyexpat
import re
import sys
import time


def parse_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument('--asset', type=Path, help='reconstruct_object.py 或 prepare 生成的米制物体资产目录')
    parser.add_argument('--scene', type=Path, help='多物体 JSON：objects 中逐个声明 id/role/asset/initial_selection/prompt')
    parser.add_argument('--all-frames', action='store_true', help='离线模式逐帧处理，不按回放速率丢帧或等待')
    parser.add_argument('--max-frames', type=int, help='最多处理的帧数；用于显式的部分验收')
    parser.add_argument('--foundationpose-root', type=Path, default=workspace / 'third_party/FoundationPose')
    parser.add_argument('--object-prompt', help='Grounded-SAM2 文本目标，例如 "red mug."；省略时手动圈选')
    from ..sam2_config import add_sam_arguments
    add_sam_arguments(parser, workspace)
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
    if not args.check and args.asset is None and args.scene is None:
        parser.error('追踪需要 --asset 或 --scene；只检查环境可用 --check')
    if args.scene and (args.asset or args.initial_selection or args.object_prompt):
        parser.error('--scene 内逐个指定资产、选择和 prompt，不能混用单物体参数')
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


def load_scene(path, workspace, *, offline, headless):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {'objects'} or not isinstance(data['objects'], list):
        raise ValueError('scene JSON must contain an objects list')
    if not data['objects']:
        raise ValueError('scene requires at least one fixed instance')
    objects = {}
    for item in data['objects']:
        if not isinstance(item, dict) or set(item) - {'id', 'role', 'asset', 'initial_selection', 'prompt'}:
            raise ValueError('Each scene object accepts id, role, asset, initial_selection, prompt')
        name = item.get('id', '')
        if not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_]*', name) or '__' in name or name in objects:
            raise ValueError('Scene object IDs must be unique lowercase identifiers without double underscores')
        item = item.copy()
        item['role'] = item.get('role', 'object')
        if item['role'] not in {'object', 'reference'}:
            raise ValueError('Scene role must be object or reference')
        for key in ('asset', 'initial_selection'):
            if item.get(key) is not None:
                candidate = Path(item[key]).expanduser()
                item[key] = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
                if not item[key].is_dir():
                    raise ValueError(f'{name}: missing {key} directory {item[key]}')
        if item.get('initial_selection') and not offline:
            raise ValueError('Historical initial_selection cannot initialize a new camera frame')
        if headless and not item.get('initial_selection'):
            raise ValueError(f'{name}: headless scene requires an exact-source initial_selection')
        if item.get('prompt') is not None and (not isinstance(item['prompt'], str) or not item['prompt'].strip()):
            raise ValueError('Scene prompt must be nonempty text')
        objects[name] = item
    return objects


def _choose_masks(objects, frame, args, viewer, output, worker):
    import numpy as np
    from ..io import _sha256, load_initial_selection
    from ..rgbd import rectify_frame, save_frame

    masks, details, replay_objects = {}, {}, []
    try:
        for name, item in objects.items():
            candidates = []
            detection_metrics = None
            if item.get('initial_selection'):
                masks[name], details[name] = load_initial_selection(
                    item['initial_selection'], frame)
                replay_objects.append({**item, 'initial_selection': str(item['initial_selection'])})
                continue
            # Select on the original RGB grid so the confirmed choice can use
            # the same source-verified selection format on the next replay.
            viewer.begin_selection(frame.rgb, manual=not item.get('prompt'))
            if item.get('prompt'):
                if not worker.submit('predict', frame, item['prompt'], box_threshold=args.box_threshold,
                                     text_threshold=args.text_threshold):
                    raise RuntimeError('Scene worker was not ready for selection')
                def selecting():
                    key = viewer.update(frame.rgb, frame.rgb, lines=[f'Grounded-SAM2: finding {name}'],
                                        status_text='Text segmentation', help_text='q: cancel')
                    if key in (27, ord('q')):
                        raise KeyboardInterrupt('Selection cancelled')
                result = _wait_worker(worker, selecting)
                if result.frame is not frame or result.operation != 'predict':
                    raise RuntimeError('Rejected mismatched detection result')
                candidates = result.value
                detection_metrics = dict(worker.segmenter.last_metrics)
                viewer.set_candidates(candidates)
                if not candidates:
                    viewer.use_polygon()
            while True:
                key = viewer.update(frame.rgb, frame.rgb,
                                    lines=[f'Select {name} ({item["role"]}), then Enter. q cancels the scene.'],
                                    status_text='Confirm every instance on the same frozen frame',
                                    help_text='1-9 / [ ] / click: candidate | p: polygon | Enter: confirm | q: quit')
                if key in (ord('q'), 27):
                    raise KeyboardInterrupt('Scene selection cancelled')
                if key == ord('p'):
                    viewer.use_polygon()
                viewer.choose_candidate(key)
                if key in (10, 13):
                    mask = viewer.selected_mask()
                    if mask is not None and np.count_nonzero(mask & (frame.depth_m > 0)) >= 4:
                        _, selected_depth, _, selected_mask = rectify_frame(frame, mask)
                        if np.count_nonzero(selected_mask & (selected_depth > 0)) < 4:
                            continue
                        masks[name] = selected_mask
                        details[name] = dict(method='gui_confirmed', prompt=item.get('prompt'),
                                             selected_instance=viewer.selected_candidate_index,
                                             detection_metrics=detection_metrics)
                        selected_index = viewer.selected_candidate_index
                        candidate = candidates[selected_index] if selected_index is not None else None
                        scores = dict(
                            detection_score=candidate['score'] if candidate is not None else None,
                            mask_score=candidate['mask_score'] if candidate is not None else None,
                            score_kind='grounding_dino' if candidate is not None else None,
                            mask_score_kind='sam2_predicted_iou' if candidate is not None else None,
                            box_threshold=args.box_threshold, text_threshold=args.text_threshold)
                        details[name].update(scores)
                        selected = output / 'initial_selections' / name
                        selected.mkdir(parents=True)
                        save_frame(selected / 'source_rgbd.npz', frame)
                        np.save(selected / 'mask.npy', mask, allow_pickle=False)
                        (selected / 'selection.json').write_text(json.dumps(dict(
                            selection='gui_confirmed', selected_instance=viewer.selected_candidate_index,
                            mask_grid='original_rgb_grid; not_automatically_rectified',
                            mask_file='mask.npy', mask_sha256=_sha256(selected / 'mask.npy'),
                            source_id=frame.source_id, timestamp_ns=frame.timestamp_ns, clock=frame.clock,
                            prompt=item.get('prompt'),
                            detection_metrics=detection_metrics,
                            **scores,
                            segmentation_model=worker.metadata['segmentation'] if item.get('prompt') else None,
                        ), indent=2, ensure_ascii=False))
                        replay_objects.append({**item, 'initial_selection': str(selected)})
                        break
            viewer.cancel_selection()
        save_frame(output / 'initial_source_rgbd.npz', frame)
        for name, mask in masks.items():
            path = output / f'initial_{name}_rectified.npy'
            np.save(path, mask, allow_pickle=False)
            details[name]['rectified_mask_sha256'] = _sha256(path)
        (output / 'initialization.json').write_text(json.dumps(
            dict(source_id=frame.source_id, timestamp_ns=frame.timestamp_ns, clock=frame.clock,
                 row_index=frame.row_index, objects=details), indent=2, ensure_ascii=False))
        (output / 'replay_scene.json').write_text(json.dumps(
            {'objects': [{k: str(v) if isinstance(v, Path) else v for k, v in item.items()}
                         for item in replay_objects]}, indent=2, ensure_ascii=False))
        return masks
    finally:
        if viewer is not None:
            viewer.cancel_selection()


def _wait_worker(worker, heartbeat, *, loading=False):
    while True:
        status = worker.snapshot()
        if status.error:
            raise RuntimeError(status.error)
        if loading and status.ready:
            return
        if not loading:
            result = worker.poll()
            if result is not None:
                if result.error:
                    raise RuntimeError(result.error)
                return result
        heartbeat()
        time.sleep(.005)


def scene_display(result, colors, hand=None):
    import cv2
    import numpy as np
    from ..geometry import project_points

    overlay = result['rgb'].copy()
    observed, models, observed_colors, model_colors = [], [], [], []
    for name, item in result['objects'].items():
        color = colors[name]
        mask = item['cloud'].semantic_mask
        overlay[mask] = (overlay[mask] * .62 + np.asarray(color) * .38).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)
        points = item['cloud'].observed_points_camera
        observed.append(points)
        observed_colors.append(np.tile(np.asarray(color) / 255., (len(points), 1)))
        points = item['model_points_camera']
        models.append(points)
        model_colors.append(np.tile(np.asarray(color) / 255., (len(points), 1)))
        if item['pose_valid']:
            p = item['pose']
            axes = np.array([[0, 0, 0], [.035, 0, 0], [0, .035, 0], [0, 0, .035]])
            uv, good = project_points(axes @ p[:3, :3].T + p[:3, 3], result['K'])
            if good.all() and np.abs(uv).max() < 1e6:
                uv = np.rint(uv).astype(int)
                for i, axis_color in enumerate(((255, 70, 70), (70, 255, 70), (70, 70, 255)), 1):
                    cv2.line(overlay, tuple(uv[0]), tuple(uv[i]), axis_color, 2)
                cv2.putText(overlay, name, tuple(uv[0]), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    if hand is not None:
        uv, good = project_points(hand['points_camera'], result['K'])
        h, w = overlay.shape[:2]
        good &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        for u, v in np.rint(uv[good]).astype(int):
            cv2.circle(overlay, (int(u), int(v)), 1, (255, 140, 35), -1)
    return overlay, dict(observed_points=np.concatenate(observed), model_points=np.concatenate(models),
                         observed_colors=np.concatenate(observed_colors), model_colors=np.concatenate(model_colors))


def run_scene(args):
    from ..sam2_config import check_sam_environment
    from .check_tracking import check_tracking_environment, print_tracking_report

    live = not (args.episode or args.rgbd_dir)
    objects = (load_scene(args.scene, args.workspace, offline=not live, headless=args.headless)
               if args.scene else {'object': dict(id='object', role='object', asset=args.asset,
                                                  initial_selection=args.initial_selection, prompt=args.object_prompt)})
    report = check_tracking_environment(args.foundationpose_root, need_camera=live,
                                        need_gui=not args.check and not args.headless,
                                        need_pose=args.scene is None or any(item.get('asset') for item in objects.values()))
    report['checks'].extend(check_sam_environment(args, require_grounding=any(
        item.get('prompt') and not item.get('initial_selection') for item in objects.values())))
    report['ok'] = all(item['ok'] for item in report['checks'])
    print_tracking_report(report)
    if args.check or not report['ok']:
        return 0 if report['ok'] else 2

    import cv2
    import numpy as np
    from ..io import load_asset, _new_output, _sha256, write_observation
    from ..integrations.dexmani import iter_raw_episode, iter_live_realsense, RawHandCloudProvider
    from ..live_workers import LatestFrameReader, SceneWorker
    from ..rgbd import iter_rgbd_directory, rectify_frame, save_frame
    from ..scene_tracking import SceneTracker, scene_payload
    from ..segmentation import GroundedSAM2Segmenter

    assets, asset_metadata = {}, {}
    for name, item in objects.items():
        if item.get('asset'):
            mesh, points, normals, metadata = load_asset(item['asset'])
            assets[name] = (mesh, points, normals)
            asset_metadata[name] = metadata
    protected = [args.scene, args.episode, args.rgbd_dir]
    for item in objects.values():
        protected.extend((item.get('asset'), item.get('initial_selection')))
    output = _new_output(args.output, workspace=args.workspace, protected=protected)
    if args.save_frames:
        (output / 'frames').mkdir()
    if args.save_inputs:
        (output / 'inputs').mkdir()
    roles = {name: item['role'] for name, item in objects.items()}
    palette = [(50, 220, 230), (220, 90, 230), (80, 220, 100), (240, 220, 70)]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(objects)}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    hand_provider = (RawHandCloudProvider(args.episode, config_path=args.experiment_config,
                                         num_points=args.hand_points) if args.with_hand else None)
    (output / 'run.json').write_text(json.dumps(dict(
        config=config, scene={n: {k: str(v) if isinstance(v, Path) else v for k, v in o.items()}
                              for n, o in objects.items()}, assets=asset_metadata, environment_check=report,
        scene_sha256=_sha256(args.scene) if args.scene else None,
        coordinate_frame='color_camera', units='m', colors_rgb=colors,
        observed_semantics='current semantic mask + measured depth, independent of mesh; segmentation can be wrong',
        model_semantics='fixed generated surface + per-instance T_camera_object; geometry agreement is not confidence',
        processing='all frames, unpaced' if args.all_frames else 'latest frame, bounded mailbox, no backlog',
        source_clock='preserved; offline timestamps do not prove sensor synchronization',
        hand=hand_provider.metadata if hand_provider else None,
        video='processed frames only, replay-fps playback, no reconstruction of original timing',
    ), indent=2, ensure_ascii=False))
    if args.episode:
        source_factory = lambda: iter_raw_episode(args.episode)
    elif args.rgbd_dir:
        source_factory = lambda: iter_rgbd_directory(args.rgbd_dir)
    else:
        source_factory = lambda: iter_live_realsense(serial=args.serial, width=args.width, height=args.height, fps=args.fps)
    viewer = worker = segmenter = reader = source = video = None
    current = hand = overlay = display = None
    records = (output / 'records.jsonl').open('x')
    processed = source_count = 0
    last_version = 0
    failure = None
    completed = False
    first_row = last_row = None
    published_state = 'LOADING'
    def heartbeat():
        nonlocal published_state
        fresh = current is not None
        if current is not None and live:
            age = (time.monotonic_ns() - current['frame'].timestamp_ns) / 1e6
            fresh = 0 <= age <= args.max_age_ms
            if not fresh and published_state != 'STALE':
                write_observation(output / 'latest.npz', scene_payload(
                    current, roles, valid=False, state='STALE', max_age_ms=args.max_age_ms))
                published_state = 'STALE'
        if viewer is not None:
            key = viewer.update(current['rgb'] if current else None, overlay, **(display or {}),
                                result_valid=fresh, hand_points=hand['points_camera'] if hand else None,
                                lines=['Loading / processing / waiting for the next frame'],
                                status_text=published_state, help_text='q: quit')
            if key in (27, ord('q')):
                raise KeyboardInterrupt('Viewer closed')

    print(f'Scene output: {output}', flush=True)
    try:
        write_observation(output / 'latest.npz', scene_payload(None, roles, valid=False, state='LOADING', max_age_ms=args.max_age_ms))
        if not args.headless:
            from ..live_viewer import LiveViewer
            viewer = LiveViewer(show_3d=not args.no_3d)
        segmenter = GroundedSAM2Segmenter(args.sam_python, args.sam_root, args.sam_checkpoint,
                                         sam_config=args.sam_config, dino_weights=args.dino_weights,
                                         log_path=output / 'sam2_worker.log')
        worker = SceneWorker(segmenter, lambda segmenter: SceneTracker(
            segmenter, assets, foundationpose_root=args.foundationpose_root,
            debug_dir=output / 'foundationpose', pixel_stride=args.pixel_stride,
            depth_tolerance_m=args.depth_tolerance_m, min_matched_pixels=args.min_matched_pixels,
            min_matched_fraction=args.min_matched_fraction)).start()
        _wait_worker(worker, heartbeat, loading=True)
        (output / 'models.json').write_text(json.dumps(worker.metadata, indent=2))
        if args.all_frames:
            source = iter(source_factory())
        else:
            reader = LatestFrameReader(source_factory, replay_interval_s=0 if live else 1 / args.replay_fps,
                                       wait_on_first=not live).start()
        while True:
            if source is not None:
                frame = next(source, None)
                if frame is None:
                    completed = True
                    break
                source_count += 1
            else:
                snapshot = reader.snapshot()
                source_count = snapshot.version
                if snapshot.error:
                    raise RuntimeError(snapshot.error)
                if snapshot.version == last_version:
                    if snapshot.finished:
                        if live:
                            raise RuntimeError('Live RGB-D source ended')
                        completed = True
                        break
                    heartbeat()
                    time.sleep(.002)
                    continue
                frame, last_version = snapshot.frame, snapshot.version
            rectified = rectify_frame(frame)[:3]
            initial = (_choose_masks(objects, frame, args, viewer, output, worker) if processed == 0 else None)
            if processed == 0:
                # First-frame text prompts may have lazily loaded the detector.
                (output / 'models.json').write_text(json.dumps(worker.metadata, indent=2))
            if not worker.submit('step', frame, *rectified, initial_masks=initial):
                raise RuntimeError('Scene worker was not ready for a frame')
            finished = _wait_worker(worker, heartbeat)
            if finished.frame is not frame or finished.operation != 'step':
                raise RuntimeError('Rejected mismatched scene worker result')
            current = finished.value
            hand = hand_provider.for_frame(frame) if hand_provider else None
            processed += 1
            if processed == 1:
                first_row = frame.row_index
            last_row = frame.row_index
            if reader is not None and processed == 1:
                reader.resume()
            age_ms = (time.monotonic_ns() - frame.timestamp_ns) / 1e6 if live else None
            valid = not live or (0 <= age_ms <= args.max_age_ms)
            state = 'TRACKING' if live and valid else 'REPLAY' if not live else 'STALE'
            payload = scene_payload(current, roles, valid=valid, state=state, max_age_ms=args.max_age_ms, hand=hand)
            write_observation(output / 'latest.npz', payload)
            published_state = state
            if args.save_frames:
                write_observation(output / 'frames' / f'{processed-1:06d}.npz', payload)
            if args.save_inputs:
                save_frame(output / 'inputs' / f'{processed-1:06d}.npz', frame)
            record = dict(source_id=frame.source_id, row_index=frame.row_index, timestamp_ns=frame.timestamp_ns,
                          clock=frame.clock, operation=current['operation'], source_age_ms=age_ms,
                          segmentation_ms=current['segmentation_ms'], pose_ms=current['pose_ms'],
                          segmentation_metrics=current['segmentation_metrics'],
                          processing_ms=(current['completed_ns']-current['started_ns'])/1e6,
                          completed_ns=current['completed_ns'], valid=valid,
                          objects={name: dict(state=item['state'], error=item['error'],
                                             pose_valid=item['pose_valid'], observed_valid=item['observed_valid'],
                                             T_camera_object=item['pose'].tolist() if item['pose_valid'] else None,
                                             diagnostic_T_camera_object=(item['diagnostic_pose'].tolist()
                                                                         if item['diagnostic_pose'] is not None else None),
                                             diagnostics=item['cloud'].diagnostics)
                                   for name, item in current['objects'].items()})
            records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            records.flush()
            if viewer is not None or args.save_video:
                overlay, display = scene_display(current, colors, hand)
                lines = [f'{state} | source row={frame.row_index} | processed={processed} | '
                         f'pipeline={record["processing_ms"]:.1f} ms | SAM={record["segmentation_ms"]:.1f} ms']
                lines.extend(f'{name}: {item["state"]} | observed={len(item["cloud"].observed_points_camera)}'
                             for name, item in current['objects'].items())
                lines.append('Model agreement is not pose ground truth; orange=FK hand')
                if args.save_video:
                    panel = cv2.cvtColor(np.hstack((current['rgb'], overlay)), cv2.COLOR_RGB2BGR)
                    cv2.rectangle(panel, (0, 0), (panel.shape[1], 20+20*len(lines)), (0, 0, 0), -1)
                    for i, line in enumerate(lines):
                        cv2.putText(panel, line, (8, 18+20*i), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
                    if video is None:
                        video = cv2.VideoWriter(str(output / 'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                args.replay_fps, (panel.shape[1], panel.shape[0]))
                        if not video.isOpened():
                            raise RuntimeError('Could not create scene overlay video')
                    video.write(panel)
                if viewer is not None:
                    key = viewer.update(current['rgb'], overlay, **display, result_valid=valid,
                                        hand_points=hand['points_camera'] if hand else None, lines=lines,
                                        status_text='Per-instance pose and observed-depth states shown below',
                                        help_text='v: measured/model layers | s: screenshot | q: quit')
                    if key == ord('s'):
                        viewer.screenshot(output / f'scene_{processed:06d}.png')
                    if key in (27, ord('q')):
                        break
            if args.max_frames and processed >= args.max_frames:
                break
        if processed == 0:
            raise ValueError('RGB-D source contained no frames')
    except KeyboardInterrupt:
        print('Scene stopped by user', flush=True)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        print(failure, flush=True)
    finally:
        # Invalidate before resource shutdown, even after partial inference.
        try:
            write_observation(output / 'latest.npz', scene_payload(current, roles, valid=False,
                              state='ERROR' if failure else 'STOPPED', max_age_ms=args.max_age_ms))
        except Exception as error:
            failure = f'{failure or ""}; invalidation: {error}'
            (output / 'latest.npz').unlink(missing_ok=True)
        if viewer is not None and processed:
            try:
                viewer.screenshot(output / 'last_frame.png')
            except Exception as error:
                failure = f'{failure or ""}; screenshot: {error}'
        for resource in (reader, source, worker, segmenter if worker is None else None, viewer, video):
            if resource is None:
                continue
            try:
                if resource is reader or resource is worker:
                    status = resource.close(timeout_s=2)
                    if not status.stopped or status.error:
                        raise RuntimeError(f'{type(resource).__name__} shutdown unconfirmed: {status}')
                elif resource is video:
                    video.release()
                else:
                    resource.close()
            except Exception as error:
                failure = f'{failure or ""}; cleanup: {error}'
        records.close()
        cuda_allocated = cuda_reserved = None
        if assets:
            import torch
            if torch.cuda.is_available():
                cuda_allocated = torch.cuda.max_memory_allocated()
                cuda_reserved = torch.cuda.max_memory_reserved()
        (output / 'result.json').write_text(json.dumps(dict(
            status='failed' if failure else 'completed_replay' if completed and not live else 'stopped',
            error=failure, processed_frames=processed, source_frames=source_count,
            first_processed_row=first_row, last_processed_row=last_row, live_camera_requested=live,
            cuda_peak_allocated_bytes=cuda_allocated,
            cuda_peak_reserved_bytes=cuda_reserved,
        ), indent=2, ensure_ascii=False))
    return 1 if failure else 0


def main(argv=None, *, workspace=None):
    try:
        return run_scene(parse_args(argv, workspace=workspace))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, ImportError, RuntimeError, KeyError) as error:
        print(f'追踪失败：{error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
