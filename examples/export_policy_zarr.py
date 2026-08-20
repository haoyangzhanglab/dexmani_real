"""Usage: ``python examples/export_policy_zarr.py --input-root DIR --output PATH``.

Export processed task episodes to dexmani_policy Zarr.
"""

from dexmani_real.data_processing.zarr_cli import main

if __name__ == "__main__":
    main()
