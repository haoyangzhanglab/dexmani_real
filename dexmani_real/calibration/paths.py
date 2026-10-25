"""当前标定数据的默认位置；导入时不读取标定文件。"""

from dexmani_real import PACKAGE_DIR

CAMERAS_PATH = PACKAGE_DIR / "calibration" / "parameters" / "cameras.json"
TABLE_PLANE_PATH = PACKAGE_DIR / "calibration" / "parameters" / "table_plane.json"
VR_TRANSFORM_PATH = PACKAGE_DIR / "calibration" / "parameters" / "vr_transform.json"

TABLE_PLANE_RELATIVE_PATH = TABLE_PLANE_PATH.relative_to(PACKAGE_DIR.parent).as_posix()
