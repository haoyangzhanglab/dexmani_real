"""OpenCV 图像绘制与窗口；输入 RGB，显示画布为 BGR。"""

from collections.abc import Sequence

import cv2
import numpy as np


def colorize_depth(
    depth_m: np.ndarray, depth_range_m: tuple[float, float], *, colormap: str = "jet",
) -> np.ndarray:
    lower, upper = depth_range_m
    if not lower < upper:
        raise ValueError("depth display range must be increasing")
    valid = np.isfinite(depth_m) & (depth_m > 0)
    safe = np.where(valid, depth_m, lower)
    gray = (255 * (1 - np.clip((safe - lower) / (upper - lower), 0, 1))).astype(np.uint8)
    if colormap == "jet":
        image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    elif colormap == "gray":
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"unknown depth colormap: {colormap}")
    image[~valid] = 0
    return image


def draw_text(
    image: np.ndarray, lines: Sequence[str], *, origin=(10, 24), color=(255, 255, 255),
) -> None:
    """在调用方拥有的 BGR 画布上绘制文字。"""
    x, y = origin
    wrapped = []
    for line in lines:
        current = ""
        for word in line.split():
            candidate = f"{current} {word}" if current else word
            width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            if current and width > image.shape[1] - x - 10:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        wrapped.append(current)
    for index, line in enumerate(wrapped):
        at = (x, y + index * 23)
        cv2.putText(image, line, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def rgbd_panel(
    rgb: np.ndarray, depth_m: np.ndarray, *, depth_range_m: tuple[float, float],
    colormap: str = "jet", lines: Sequence[str] = (),
) -> np.ndarray:
    panel = np.hstack((
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        colorize_depth(depth_m, depth_range_m, colormap=colormap),
    ))
    draw_text(panel, lines)
    return panel


def draw_aruco(
    image: np.ndarray, corners: Sequence[np.ndarray], ids: np.ndarray | None,
    pose: tuple[np.ndarray, np.ndarray] | None, *, intrinsics: np.ndarray,
    distortion: np.ndarray, axis_length: float,
) -> None:
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    if pose is not None:
        rvec, tvec = pose
        cv2.drawFrameAxes(image, intrinsics, distortion, rvec, tvec, axis_length)


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
        key = cv2.waitKey(1)
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            self.close()
            return ["closed"]
        if key < 0:
            return []
        key &= 0xFF
        return [{27: "esc", 13: "enter", 10: "enter", 32: "space"}.get(key, chr(key).lower())]

    def close(self) -> None:
        if not self.closed:
            cv2.destroyWindow(self.title)
            self.closed = True
