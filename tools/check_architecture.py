"""Check import boundaries and ratcheted architecture metrics.

The checker parses Python source with :mod:`ast`; it never imports production
modules, vendor SDKs, or hardware entry points.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "dexmani_real"
BASELINE_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "contracts" / "architecture_baseline.json"
)

HEAVY_IMPORTS = frozenset(
    {
        "av",
        "cv2",
        "h5py",
        "mplib",
        "open3d",
        "pinocchio",
        "pyrealsense2",
        "scipy",
        "xarm",
        "xhand_controller",
        "zarr",
    }
)
FORBIDDEN_EDGES = frozenset(
    {
        ("deployment", "integrations"),
        ("ipc", "data"),
        ("ipc", "recording"),
        ("ipc", "robot"),
        ("ipc", "runtime"),
        ("ipc", "sensor"),
    }
)


@dataclass(frozen=True)
class ImportRecord:
    source_file: Path
    source_package: str
    target_module: str
    imported_names: tuple[str, ...]

    @property
    def target_package(self) -> str | None:
        prefix = "dexmani_real."
        if not self.target_module.startswith(prefix):
            return None
        suffix = self.target_module[len(prefix) :]
        return suffix.split(".", 1)[0] if suffix else None


def _package_for(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else "__root__"


def _imports(path: Path) -> list[ImportRecord]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_package = _package_for(path)
    records: list[ImportRecord] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            target_module = node.module
            if node.level:
                relative_parent = (
                    path.relative_to(REPOSITORY_ROOT).with_suffix("").parts[:-1]
                )
                if path.name == "__init__.py":
                    relative_parent = path.relative_to(REPOSITORY_ROOT).parts[:-1]
                keep = len(relative_parent) - (node.level - 1)
                target_module = ".".join((*relative_parent[:keep], node.module))
            records.append(
                ImportRecord(
                    source_file=path,
                    source_package=source_package,
                    target_module=target_module,
                    imported_names=tuple(alias.name for alias in node.names),
                )
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                records.append(
                    ImportRecord(
                        source_file=path,
                        source_package=source_package,
                        target_module=alias.name,
                        imported_names=(),
                    )
                )
    return records


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph.get(node, set()):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == node:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return sorted(components)


def analyze_repository(paths: Iterable[Path] | None = None) -> dict[str, object]:
    python_files = list(paths or sorted(PACKAGE_ROOT.rglob("*.py")))
    records = [record for path in python_files for record in _imports(path)]
    graph: dict[str, set[str]] = {}
    forbidden: list[str] = []
    defaults: list[str] = []
    private_imports: list[str] = []
    heavy_init: list[str] = []
    missing_internal: list[str] = []

    for record in records:
        target_package = record.target_package
        if record.target_module == "dexmani_real" or record.target_module.startswith(
            "dexmani_real."
        ):
            module_parts = record.target_module.split(".")
            module_path = REPOSITORY_ROOT.joinpath(*module_parts)
            if (
                not module_path.with_suffix(".py").is_file()
                and not (module_path / "__init__.py").is_file()
            ):
                missing_internal.append(
                    f"{record.source_file.relative_to(REPOSITORY_ROOT)}: "
                    f"{record.target_module}"
                )
        if target_package and target_package != record.source_package:
            graph.setdefault(record.source_package, set()).add(target_package)
            if (record.source_package, target_package) in FORBIDDEN_EDGES:
                forbidden.append(
                    f"{record.source_file.relative_to(REPOSITORY_ROOT)}: "
                    f"{record.source_package} -> {target_package}"
                )
        if record.target_module == "dexmani_real.config.defaults":
            defaults.append(str(record.source_file.relative_to(REPOSITORY_ROOT)))
        if record.target_module.startswith("dexmani_real.") and any(
            name.startswith("_") and not name.startswith("__")
            for name in record.imported_names
        ):
            private_imports.append(str(record.source_file.relative_to(REPOSITORY_ROOT)))
        if record.source_file.name == "__init__.py":
            external_root = record.target_module.split(".", 1)[0]
            if external_root in HEAVY_IMPORTS:
                heavy_init.append(
                    f"{record.source_file.relative_to(REPOSITORY_ROOT)}: {external_root}"
                )

    cycles = _strongly_connected_components(graph)
    return {
        "package_cycles": cycles,
        "forbidden_edges": sorted(set(forbidden)),
        "direct_defaults_imports": len(defaults),
        "direct_defaults_files": sorted(set(defaults)),
        "private_imports": len(private_imports),
        "private_import_files": sorted(set(private_imports)),
        "heavy_init_imports": sorted(set(heavy_init)),
        "missing_internal_imports": sorted(set(missing_internal)),
    }


def check_against_baseline(
    report: dict[str, object], baseline: dict[str, int]
) -> list[str]:
    errors: list[str] = []
    for metric in ("direct_defaults_imports", "private_imports"):
        actual = cast(int, report[metric])
        maximum = int(baseline[metric])
        if actual > maximum:
            errors.append(f"{metric} increased: {actual} > baseline {maximum}")
    for metric in (
        "package_cycles",
        "forbidden_edges",
        "heavy_init_imports",
        "missing_internal_imports",
    ):
        actual = len(cast(Sized, report[metric]))
        maximum = int(baseline[metric])
        if actual > maximum:
            errors.append(f"{metric} increased: {actual} > baseline {maximum}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args()
    report = analyze_repository()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    errors = check_against_baseline(report, baseline)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        cycles = cast(Sized, report["package_cycles"])
        forbidden = cast(Sized, report["forbidden_edges"])
        heavy_init = cast(Sized, report["heavy_init_imports"])
        missing = cast(Sized, report["missing_internal_imports"])
        print(
            "architecture: "
            f"cycles={len(cycles)} "
            f"forbidden={len(forbidden)} "
            f"defaults={report['direct_defaults_imports']} "
            f"private={report['private_imports']} "
            f"heavy_init={len(heavy_init)}"
            f" missing={len(missing)}"
        )
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
