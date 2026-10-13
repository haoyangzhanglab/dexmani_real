"""Multi-instance mode for the existing track_object example."""

import json
from pathlib import Path
import re
import time


def load_scene(path, workspace, *, offline, headless):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {'objects'} or not isinstance(data['objects'], list):
        raise ValueError('scene JSON must contain an objects list')
    if not data['objects']:
        raise ValueError('scene requires at least one object')
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


def _choose_masks(objects, frame, rectified, args, viewer, output):
    import cv2
    import numpy as np
    from .. import geometry
    from ..io import _save_input, _sha256
    from ..segmentation import GroundedSegmenter
    from .live_object_tracking import load_initial_selection

    masks, details, replay_objects = {}, {}, []
    detector = None
    try:
        for name, item in objects.items():
            if item.get('initial_selection'):
                _, masks[name], details[name] = load_initial_selection(
                    item['initial_selection'], frame, geometry, np, cv2)
                replay_objects.append({**item, 'initial_selection': str(item['initial_selection'])})
                continue
            # Select on the original RGB grid so the confirmed choice can use
            # the same source-verified selection format on the next replay.
            viewer.begin_selection(frame.rgb, manual=not item.get('prompt'))
            if item.get('prompt'):
                if detector is None:
                    detector = GroundedSegmenter(args.dino_weights, args.sam_root, args.sam_checkpoint, args.sam_config)
                candidates = detector.predict(frame.rgb, item['prompt'], args.box_threshold, args.text_threshold)
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
                        _, selected_depth, _, selected_mask = geometry.rectify_rgbd(
                            frame.rgb, frame.depth_m, frame.K, frame.distortion_coeffs, mask=mask)
                        if np.count_nonzero(selected_mask & (selected_depth > 0)) < 4:
                            continue
                        masks[name] = selected_mask
                        details[name] = dict(method='gui_confirmed', prompt=item.get('prompt'),
                                             selected_instance=viewer.selected_candidate_index)
                        selected = output / 'initial_selections' / name
                        selected.mkdir(parents=True)
                        _save_input(selected / 'source_rgbd.npz', frame, np)
                        np.save(selected / 'mask.npy', mask, allow_pickle=False)
                        (selected / 'selection.json').write_text(json.dumps(dict(
                            selection='gui_confirmed', selected_instance=viewer.selected_candidate_index,
                            mask_grid='original_rgb_grid; not_automatically_rectified',
                            mask_file='mask.npy', mask_sha256=_sha256(selected / 'mask.npy'),
                            source_id=frame.source_id, timestamp_ns=frame.timestamp_ns, clock=frame.clock,
                            prompt=item.get('prompt'),
                            segmentation_model=detector.provenance if item.get('prompt') else None,
                        ), indent=2, ensure_ascii=False))
                        replay_objects.append({**item, 'initial_selection': str(selected)})
                        break
            viewer.cancel_selection()
        _save_input(output / 'initial_source_rgbd.npz', frame, np)
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
        if detector is not None:
            detector.close()


def scene_display(result, colors, hand=None):
    import cv2
    import numpy as np
    from ..viewer import project_points

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
    import cv2
    import numpy as np
    from .. import geometry
    from ..io import _asset, _new_output, _save_input, load_asset_normals, _sha256
    from ..integrations.dexmani import iter_raw_episode, iter_live_realsense, RawHandCloudProvider
    from ..live_workers import LatestFrameReader
    from ..observation import write_observation
    from ..rgbd import iter_rgbd_directory
    from ..scene_tracking import SceneTracker, scene_payload
    from ..stream_segmentation import StreamSegmenter
    from .live_object_tracking import rectify_frame, check_segmentation_environment
    from .check_tracking import check_tracking_environment, print_tracking_report

    live = not (args.episode or args.rgbd_dir)
    objects = load_scene(args.scene, args.workspace, offline=not live, headless=args.headless)
    report = check_tracking_environment(args.foundationpose_root, need_camera=live,
                                        need_gui=not args.check and not args.headless,
                                        need_pose=any(item.get('asset') for item in objects.values()))
    report['checks'].extend(check_segmentation_environment(
        args, need_grounding=any(item.get('prompt') and not item.get('initial_selection')
                                 for item in objects.values())))
    report['ok'] = all(item['ok'] for item in report['checks'])
    print_tracking_report(report)
    if args.check or not report['ok']:
        return 0 if report['ok'] else 2
    assets, asset_metadata = {}, {}
    for name, item in objects.items():
        if item.get('asset'):
            mesh, points, metadata = _asset(item['asset'], geometry)
            assets[name] = (mesh, points, load_asset_normals(item['asset'], points))
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
        scene_sha256=_sha256(args.scene), sam_checkpoint_sha256=_sha256(args.sam_checkpoint),
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
    viewer = tracker = segmenter = reader = source = video = None
    current = hand = overlay = display = None
    records = (output / 'records.jsonl').open('x')
    processed = source_count = 0
    last_version = 0
    failure = None
    completed = False
    first_row = last_row = None
    published_state = 'LOADING'
    print(f'Scene output: {output}', flush=True)
    try:
        write_observation(output / 'latest.npz', scene_payload(None, roles, valid=False, state='LOADING', max_age_ms=args.max_age_ms))
        segmenter = StreamSegmenter(args.sam_root, args.sam_checkpoint, args.sam_config)
        (output / 'segmentation.json').write_text(json.dumps(segmenter.provenance, indent=2))
        tracker = SceneTracker(segmenter, assets, foundationpose_root=args.foundationpose_root,
                               debug_dir=output / 'foundationpose', pixel_stride=args.pixel_stride,
                               depth_tolerance_m=args.depth_tolerance_m, min_matched_pixels=args.min_matched_pixels,
                               min_matched_fraction=args.min_matched_fraction)
        if not args.headless:
            from ..live_viewer import LiveViewer
            viewer = LiveViewer(show_3d=not args.no_3d)
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
                    fresh = current is not None
                    if current is not None and live:
                        age = (time.monotonic_ns() - current['frame'].timestamp_ns) / 1e6
                        fresh = 0 <= age <= args.max_age_ms
                        if not fresh and published_state != 'STALE':
                            write_observation(output / 'latest.npz', scene_payload(
                                current, roles, valid=False, state='STALE', max_age_ms=args.max_age_ms))
                            published_state = 'STALE'
                    if viewer is not None:
                        key = viewer.update(current['rgb'] if current else None, overlay,
                                            **(display or {}), result_valid=fresh,
                                            hand_points=hand['points_camera'] if hand else None,
                                            lines=['Waiting for next source frame'],
                                            status_text=published_state, help_text='q: quit')
                        if key in (27, ord('q')):
                            break
                    time.sleep(.002)
                    continue
                frame, last_version = snapshot.frame, snapshot.version
            rectified = rectify_frame(frame, geometry, np)
            initial = (_choose_masks(objects, frame, rectified, args, viewer, output) if processed == 0 else None)
            current = tracker.step(frame, *rectified, initial_masks=initial)
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
                _save_input(output / 'inputs' / f'{processed-1:06d}.npz', frame, np)
            record = dict(source_id=frame.source_id, row_index=frame.row_index, timestamp_ns=frame.timestamp_ns,
                          clock=frame.clock, operation=current['operation'], source_age_ms=age_ms,
                          segmentation_ms=current['segmentation_ms'], pose_ms=current['pose_ms'],
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
    except (Exception, KeyboardInterrupt) as error:
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
        for resource in (reader, source, tracker, segmenter if tracker is None else None, viewer, video):
            if resource is None:
                continue
            try:
                if resource is reader:
                    status = reader.close(timeout_s=2)
                    if not status.stopped or status.error:
                        raise RuntimeError(f'RGB-D owner shutdown unconfirmed: {status}')
                elif resource is video:
                    video.release()
                else:
                    resource.close()
            except Exception as error:
                failure = f'{failure or ""}; cleanup: {error}'
        records.close()
        import torch
        (output / 'result.json').write_text(json.dumps(dict(
            status='failed' if failure else 'completed_replay' if completed and not live else 'stopped',
            error=failure, processed_frames=processed, source_frames=source_count,
            first_processed_row=first_row, last_processed_row=last_row, live_camera_requested=live,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        ), indent=2, ensure_ascii=False))
    return 1 if failure else 0
