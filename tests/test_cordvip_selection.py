"""使用真实归档文件验证 mask 复用；不加载分割或位姿模型。"""

import json
from dataclasses import replace

import numpy as np
import pytest

from cordvip_geometry.io import load_initial_selection, write_confirmed_selection
from cordvip_geometry.rgbd import RGBDFrame, save_frame


@pytest.fixture
def selection(tmp_path):
    frame = RGBDFrame(
        rgb=np.zeros((3, 4, 3), dtype=np.uint8),
        depth_m=np.ones((3, 4), dtype=np.float32),
        K=np.array([[100., 0., 2.], [0., 100., 1.], [0., 0., 1.]]),
        distortion_model='brown_conrady', distortion_coeffs=(0.,) * 5,
        source_id='original', timestamp_ns=100, clock='host_monotonic', row_index=7,
    )
    mask = np.array([[False, True, True, False]] * 3)
    save_frame(tmp_path / 'source_rgbd.npz', frame)
    write_confirmed_selection(tmp_path, mask, {
        'selection': 'gui_confirmed', 'selected_instance': 0,
        'mask_grid': 'original_rgb_grid; not_automatically_rectified',
    }, mask_file='mask.npy')
    return tmp_path, frame, mask


@pytest.mark.parametrize('field,before,after', [
    ('timestamp_ns', 100, 101),
    ('clock', 'host_monotonic', 'device_clock'),
    ('row_index', 7, 8), ('row_index', None, 7), ('row_index', 7, None),
    ('source_id', 'original', 'relocated'),
])
def test_selection_reuses_identical_content_across_sources(selection, field, before, after):
    directory, frame, mask = selection
    saved = replace(frame, **{field: before})
    save_frame(directory / 'source_rgbd.npz', saved)
    files = [directory / name for name in ('source_rgbd.npz', 'selection.json', 'mask.npy')]
    originals = [(path.read_bytes(), path.stat().st_mtime_ns) for path in files]

    actual = load_initial_selection(directory, replace(saved, **{field: after}))

    np.testing.assert_array_equal(actual, mask)
    assert actual.dtype == np.bool_
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in files] == originals


@pytest.mark.parametrize('field', ['rgb', 'depth_m', 'K', 'distortion_model', 'distortion_coeffs'])
def test_selection_rejects_changed_content_or_geometry(selection, field):
    directory, frame, _ = selection
    if field in {'rgb', 'depth_m', 'K'}:
        value = getattr(frame, field).copy()
        index = (0, 0, 0) if field == 'rgb' else (0, 0)
        value[index] += 1
    elif field == 'distortion_model':
        value = 'none'
    else:
        value = (.01, 0., 0., 0., 0.)
    # 先构造有效帧，确保拒绝来自复用比较，尤其是畸变系数差异。
    changed = replace(frame, **{field: value})
    with pytest.raises(ValueError, match=f'initial-selection {field} differs'):
        load_initial_selection(directory, changed)


@pytest.mark.parametrize('damage,message', [
    ('shape', '同尺寸'), ('dtype', 'bool/uint8'), ('path', 'inside its directory'),
    ('confirmation', 'explicitly confirmed'), ('grid', 'original source RGB grid'),
])
def test_selection_keeps_mask_usage_checks(selection, damage, message):
    directory, frame, mask = selection
    path = directory / 'selection.json'
    metadata = json.loads(path.read_text())
    if damage == 'shape':
        np.save(directory / 'mask.npy', mask[:1])
    elif damage == 'dtype':
        np.save(directory / 'mask.npy', mask.astype(np.float32))
    elif damage == 'path':
        metadata['mask_file'] = '../outside.npy'
        np.save(directory.parent / 'outside.npy', mask)
    elif damage == 'confirmation':
        metadata['selection'] = 'unconfirmed'
    else:
        metadata['mask_grid'] = 'rectified'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        load_initial_selection(directory, frame)
