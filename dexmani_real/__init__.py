from pathlib import Path
PACKAGE_DIR = Path(__file__).parent.resolve()
ASSET_DIR = (PACKAGE_DIR / "assets") if (PACKAGE_DIR / "assets").exists() else (PACKAGE_DIR.parent / "assets")