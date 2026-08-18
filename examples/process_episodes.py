"""Compact one task's schema-v17 episodes into one HDF5 per source.

Usage:
    python examples/process_episodes.py \
        --input-root episodes/<task_name> --profile rgb_pc

Without ``--output-root``, the batch is published to
``episodes_processed/<task_name>/``.

All modalities share one compacted row mask. The command never splits one demo
into multiple outputs, and the default bridge policy rejects unsafe adjacency.
"""

from dexmani_real.data_processing.cli import main

if __name__ == "__main__":
    main()
