"""Export one task's processed HDF5 episodes to dexmani_policy Zarr.

Usage:
    python examples/export_policy_zarr.py \
        --input-root episodes_processed/<task_name> \
        --output dataset/<task_name>.zarr \
        --task-name <task_name>

The exported store contains only data arrays and meta/episode_ends. Source
provenance and task-success labels remain intentionally absent.
"""

from dexmani_real.data_processing.zarr_cli import main

if __name__ == "__main__":
    main()
