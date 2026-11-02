"""调用真实 viewer 收尾路径，替换相机和桌面窗口 I/O。"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.config.real import RealConfig
from scripts.tools import view_pointcloud, view_realsense
from scripts.visualize_episode import EpisodeVisualizer


@pytest.mark.parametrize("has_primary", [False, True])
def test_realsense_capture_closes_both_windows(monkeypatch, caplog, has_primary):
    from dexmani_real.utils.visualization import image, pointcloud

    image_window, cloud_window = Mock(), Mock()
    image_window.poll_events.side_effect = [[], ["q"]]
    cloud_window.poll_events.return_value = []
    primary = ValueError("read failed")
    first, second = OSError("cloud close failed"), OSError("image close failed")
    image_window.close.side_effect = second
    cloud_window.close.side_effect = first
    monkeypatch.setattr(image, "ImageViewer", Mock(return_value=image_window))
    monkeypatch.setattr(pointcloud, "PointCloudViewer", Mock(return_value=cloud_window))
    viewer = view_realsense.RealSenseViewer(SimpleNamespace(cloud="raw"))
    viewer.camera = Mock(depth_scale_m=.001)
    viewer.camera.read = Mock(side_effect=primary) if has_primary else Mock(return_value=None)
    viewer.pointcloud_config = PointCloudConfig()
    monkeypatch.setattr(viewer, "ensure_cloud_transform", lambda: np.eye(4))
    with pytest.raises(type(primary if has_primary else first)) as caught:
        viewer.capture()
    assert caught.value is (primary if has_primary else first)
    image_window.close.assert_called_once()
    cloud_window.close.assert_called_once()
    assert "image close failed" in caplog.text
    if has_primary:
        assert "cloud close failed" in caplog.text


def test_realsense_run_preserves_capture_error_and_disconnects(monkeypatch, caplog):
    from dexmani_real.config import real
    from dexmani_real.sensor import realsense

    camera = Mock(active_serial="fake")
    camera.disconnect.side_effect = OSError("disconnect failed")
    monkeypatch.setattr(real, "load_config", lambda **kwargs: RealConfig())
    monkeypatch.setattr(realsense, "RealSenseCamera", Mock(return_value=camera))
    monkeypatch.setattr(view_realsense, "configure_logging", lambda: None)
    viewer = view_realsense.RealSenseViewer(SimpleNamespace(config=None))
    primary = ValueError("capture failed")
    monkeypatch.setattr(viewer, "capture", Mock(side_effect=primary))
    with pytest.raises(ValueError) as caught:
        viewer.run()
    assert caught.value is primary
    camera.disconnect.assert_called_once()
    assert "disconnect failed" in caplog.text


@pytest.mark.parametrize("has_primary", [False, True])
def test_pointcloud_diagnostic_preview_and_camera_cleanup(monkeypatch, caplog, has_primary):
    from dexmani_real.config import real
    from dexmani_real.sensor import realsense
    from dexmani_real.utils import log
    from dexmani_real.utils.visualization import image

    frame = SimpleNamespace(rgb=np.zeros((2, 2, 3), np.uint8), depth=np.ones((2, 2), np.uint16))
    camera = Mock(active_serial="fake", depth_scale_m=.001)
    camera.read.return_value = frame
    camera.disconnect.side_effect = OSError("disconnect failed")
    preview = Mock()
    preview.poll_events.return_value = ["q"]
    primary = ValueError("preview failed")
    if has_primary:
        preview.show.side_effect = primary
    first = OSError("preview close failed")
    preview.close.side_effect = first
    monkeypatch.setattr("builtins.input", lambda *args: "n")
    monkeypatch.setattr(real, "load_config", lambda **kwargs: replace(
        RealConfig(), pointcloud=PointCloudConfig(remove_table=False),
    ))
    monkeypatch.setattr(realsense, "RealSenseCamera", Mock(return_value=camera))
    monkeypatch.setattr(image, "ImageViewer", Mock(return_value=preview))
    monkeypatch.setattr(log, "configure_logging", lambda: None)
    with pytest.raises(type(primary if has_primary else first)) as caught:
        view_pointcloud.run_pointcloud_diagnostic(SimpleNamespace(config=None, camera_calibration=None))
    assert caught.value is (primary if has_primary else first)
    preview.close.assert_called_once()
    camera.disconnect.assert_called_once()
    assert "disconnect failed" in caplog.text
    if has_primary:
        assert "preview close failed" in caplog.text


@pytest.mark.parametrize("has_primary", [False, True])
def test_episode_visualizer_closes_reader_and_rerun(caplog, has_primary):
    visualizer = EpisodeVisualizer.__new__(EpisodeVisualizer)
    first, second = OSError("reader close failed"), OSError("rerun close failed")
    visualizer.reader = Mock(close=Mock(side_effect=first))
    visualizer.rr = Mock(disconnect=Mock(side_effect=second))
    if has_primary:
        visualizer.close(ValueError("main failed"))
        assert "reader close failed" in caplog.text
    else:
        with pytest.raises(OSError) as caught:
            visualizer.close()
        assert caught.value is first
    visualizer.reader.close.assert_called_once()
    visualizer.rr.disconnect.assert_called_once()
    assert "rerun close failed" in caplog.text
