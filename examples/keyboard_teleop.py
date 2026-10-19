#!/usr/bin/env python3
"""通过键盘控制 xArm7/XHand，支持回零与点动。

python examples/keyboard_teleop.py
python examples/keyboard_teleop.py --config local.yaml
python examples/keyboard_teleop.py --no-hand

参数：--config 覆盖配置；--no-hand 仅连接机械臂，要求 XHand 未安装或已固定在 HOME。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dexmani_real.config.experiment import load_experiment_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for xArm7")
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not connect XHand; it must be absent or secured at configured home",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional experiment YAML overrides"
    )
    parser.add_argument("--control-hz", type=float, default=None, help="Keyboard target frequency")
    args = parser.parse_args(argv)

    from dexmani_real.teleop.keyboard_session import KeyboardTeleopSession

    try:
        runtime = load_experiment_config(
            yaml_path=args.config,
            cli_overrides={
                "control.hand_enabled": False if args.no_hand else None,
                "keyboard_teleop.control_hz": args.control_hz,
            },
        )
    except (
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        parser.error(f"invalid experiment config: {exc}")
    return KeyboardTeleopSession(runtime, no_hand=args.no_hand).run()


if __name__ == "__main__":
    raise SystemExit(main())
