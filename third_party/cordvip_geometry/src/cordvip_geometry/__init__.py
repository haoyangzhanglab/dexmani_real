"""物体资产与 RGB-D 几何追踪；导入不加载模型或设备。"""

from .assets import load_asset, prepare_asset
from .rgbd import RGBDFrame
from .tracking import SceneTracker, TrackingFailure

__all__ = ["RGBDFrame", "SceneTracker", "TrackingFailure", "load_asset", "prepare_asset"]
