"""OpenCV 图像绘制与窗口；输入 RGB，显示画布为 BGR。"""

import cv2
import numpy as np


def colorize_depth(
    depth_m: np.ndarray, depth_range_m: tuple[float, float],
) -> np.ndarray:
    lower, upper = depth_range_m
    if not lower < upper:
        raise ValueError("depth display range must be increasing")
    valid = np.isfinite(depth_m) & (depth_m > 0)
    safe = np.where(valid, depth_m, lower)
    gray = (255 * (1 - np.clip((safe - lower) / (upper - lower), 0, 1))).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    image[~valid] = 0
    return image


def rgbd_panel(
    rgb: np.ndarray, depth_m: np.ndarray, *, depth_range_m: tuple[float, float],
) -> np.ndarray:
    return np.hstack((
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        colorize_depth(depth_m, depth_range_m),
    ))


class ImageViewer:
    def __init__(self, title: str):
        self.title = title
        self.closed = False
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def show(self, image: np.ndarray) -> None:
        if not self.closed:
            cv2.imshow(self.title, image)

    def poll_events(self) -> list[str]:
        if self.closed:
            return ["closed"]
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            self.close()
            return ["closed"]
        if key in (ord("q"), ord("Q")):
            return ["q"]
        if key in (10, 13):
            return ["enter"]
        return ["esc"] if key == 27 else []

    def close(self) -> None:
        if not self.closed:
            cv2.destroyWindow(self.title)
            self.closed = True
