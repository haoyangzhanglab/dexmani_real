#!/usr/bin/env python3
"""读取离线追踪的逐行结果；不打开 Raw、相机或感知模型。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv=None, *, workspace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='track 输出目录，含 scene.json、run.json 和 frames/')
    parser.add_argument('--no-3d', action='store_true', help='只看 RGB 叠加，不加载 Open3D')
    parser.add_argument('--headless', action='store_true', help='只读取 RGB 和来源行/时钟，不加载 OpenCV/Open3D 或打开窗口')
    args = parser.parse_args(argv)
    return view_results(args.result, show_3d=not args.no_3d, headless=args.headless)


def view_results(directory, *, show_3d=True, headless=False):
    import numpy as np

    directory = Path(directory).expanduser().resolve()
    paths = sorted((directory / 'frames').glob('*.npz'))
    if not paths:
        raise ValueError('scene result contains no saved frames')
    viewer = None
    show_model = False
    count = 0
    try:
        if not headless:
            from ..viewer import DemoViewer, scene_clouds, scene_display

            scene = json.loads((directory / 'scene.json').read_text())
            run = json.loads((directory / 'run.json').read_text())
            names = [item['id'] for item in scene['objects']]
            colors = run['colors_rgb']
            viewer = DemoViewer(show_3d=show_3d)
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                # 每个显示字段只解压一次；分析用大数组留在 NPZ 中。
                frame = {key: saved[key] for key in ('rgb', 'row_index', 'timestamp_ns', 'clock')}
                if not headless:
                    frame['K'] = saved['K']
                    for name in names:
                        prefix = name + '__'
                        for field in ('semantic_mask', 'pose_valid', 'state', 'error'):
                            frame[prefix + field] = saved[prefix + field]
                        if bool(frame[prefix + 'pose_valid']):
                            frame[prefix + 'T_camera_object'] = saved[prefix + 'T_camera_object']
                        if show_3d:
                            for field in ('observed_points_camera', 'model_points_camera'):
                                frame[prefix + field] = saved[prefix + field]
                if show_3d and not headless and 'hand_points_camera' in saved.files:
                    frame['hand_points_camera'] = saved['hand_points_camera']
            rgb = frame['rgb']
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
                raise ValueError('scene result RGB must be nonempty uint8 [H,W,3]')
            row = int(frame['row_index'])
            if path.stem != f'{row:06d}':
                raise ValueError(f'{path}: filename differs from source row_index')
            print(f"row={row} timestamp_ns={int(frame['timestamp_ns'])} clock={frame['clock']!s}")
            if not headless:
                viewer.update(scene_display(frame, names, colors))
                if show_3d:
                    points, point_colors = scene_clouds(frame, names, colors, show_model=show_model)
                    viewer.update_cloud(points, point_colors)
            while not headless:
                key = viewer.poll()
                if key in (ord('q'), ord('Q'), 27):
                    return 0
                if key == ord('v'):
                    show_model = not show_model
                    if show_3d:
                        points, point_colors = scene_clouds(frame, names, colors, show_model=show_model)
                        viewer.update_cloud(points, point_colors)
                elif key in (ord('n'), ord(' '), 13):
                    break
            count += 1
    finally:
        if viewer is not None:
            failed = sys.exc_info()[0] is not None
            try:
                viewer.close()
            except BaseException:
                if not failed:
                    raise
                logger.exception("scene viewer close also failed")
    print(f'已读取 {count} 行场景结果：{directory}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
