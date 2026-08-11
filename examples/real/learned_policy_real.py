#!/usr/bin/env python3
"""Legacy filename alias for ``deploy_policy.py``.

The old backend-class ``--manifest`` contract cannot be converted safely to a
function-adapter PolicySpec. Create a PolicySpec YAML and use ``--policy``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.policy.deployment import main

if __name__ == "__main__":
    if any(argument == "--manifest" or argument.startswith("--manifest=") for argument in sys.argv[1:]):
        print(
            "--manifest is obsolete. Create a PolicySpec YAML with adapter_module, "
            "observations, action, resources, and hardware_deployable, then pass --policy.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(main())
