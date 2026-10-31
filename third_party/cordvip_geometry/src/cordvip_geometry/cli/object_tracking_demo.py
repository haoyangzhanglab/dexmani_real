#!/usr/bin/env python3
"""读取离线追踪的逐行结果；不打开 Raw、相机或感知模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None, *, workspace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='track 输出目录，含 scene.json、run.json 和 frames/')
    parser.add_argument('--no-3d', action='store_true', help='只看 RGB 叠加，不加载 Open3D')
    parser.add_argument('--headless', action='store_true', help='逐行读取并检查 RGB，不打开窗口')
    args = parser.parse_args(argv)
    return view_results(args.result, show_3d=not args.no_3d, headless=args.headless)


def view_results(directory, *, show_3d=True, headless=False):
    import numpy as np

    from ..viewer import DemoViewer, scene_display

    directory = Path(directory).expanduser().resolve()
    scene = json.loads((directory / 'scene.json').read_text())
    run = json.loads((directory / 'run.json').read_text())
    names = [item['id'] for item in scene['objects']]
    colors = run['colors_rgb']
    paths = sorted((directory / 'frames').glob('*.npz'))
    if not paths:
        raise ValueError('scene result contains no saved frames')
    viewer = None
    show_model = False
    count = 0
    try:
        if not headless:
            viewer = DemoViewer(show_3d=show_3d)
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                # 每次只持有当前结果，关闭 NPZ 后不留下文件句柄。
                frame = {key: saved[key] for key in saved.files}
            row = int(frame['row_index'])
            if path.stem != f'{row:06d}':
                raise ValueError(f'{path}: filename differs from source row_index')
            print(f"row={row} timestamp_ns={int(frame['timestamp_ns'])} clock={str(frame['clock'])}")
            while True:
                panel, points, point_colors = scene_display(
                    frame, names, colors, show_model=show_model,
                    include_clouds=not headless and show_3d,
                )
                if headless:
                    break
                key = viewer.update(panel, points, point_colors)
                if key in (ord('q'), 27):
                    return 0
                if key == ord('v'):
                    show_model = not show_model
                elif key in (ord('n'), ord(' '), 13):
                    break
            count += 1
    finally:
        if viewer is not None:
            viewer.close()
    print(f'已读取 {count} 行场景结果：{directory}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
