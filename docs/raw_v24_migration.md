# Frozen raw v24 migration

`tools/convert_raw_v24_to_v25.py` is an offline, one-shot historical converter.
It does not import the runtime reader, schema, processing, or recording code.
Keep original v24 episodes until representative converted episodes have passed
processing and Policy Zarr comparison.

```bash
python tools/convert_raw_v24_to_v25.py episodes/<task>/episode_<id> /new/path/episode_<id>
python tools/convert_raw_v24_to_v25.py episodes/<task> /new/path/<task>
```

Existing destination episodes are refused. Media is hard-linked where possible;
cross-filesystem links fall back to copying. Treat both source and converted
media as immutable. Each episode is verified and atomically published separately;
batch failures are reported without stopping conversion of the other episodes.

The conversion preserves row count, SOURCE/HOLD/PLACEHOLDER identity, retained
numeric values, and RGB/depth media. Integer fields specified as unsigned in v25
are range-checked and converted losslessly from the v24 signed representation;
other datasets are copied directly. Missing fields, including the authoritative
`action_arm_joint_sent`, are errors. No action is reconstructed or invented.

## Regression baseline

Before removing v24 runtime support, use the v24 implementation at `958fcf2`
to produce processed and Zarr outputs in a new local directory:

```python
from pathlib import Path
from dexmani_real.dataset.contracts import ProcessingConfig, OutputProfile
from dexmani_real.dataset.processing import process_episode_root
from dexmani_real.dataset.export import export_processed_hdf5_to_zarr

source = Path("episodes/pick_place_toy")
baseline = Path("/tmp/dexmani_v25_golden_before")  # must not already contain outputs
for episode in ("episode_20260827_165510", "episode_20260827_194525"):
    for profile in (OutputProfile.JOINT, OutputProfile.RGB, OutputProfile.POINTCLOUD):
        output = baseline / profile.value / episode
        process_episode_root(source / episode, output,
                             ProcessingConfig(profile=profile), verify_output=True)
        report = export_processed_hdf5_to_zarr(output, output.with_suffix(".zarr"))
        print(episode, profile.value, report)
```

After v25 runtime support lands, repeat with converted input and a different
output root. Compare processed dataset keys, shapes, discrete arrays with
`numpy.array_equal`, and floating arrays with `numpy.allclose(equal_nan=True)`.
Compare Zarr `data/*`, `meta/episode_ends`, task labels, episode counts, and export
rejections. Inspect any difference before deleting original data. Removed raw
audit fields and quality/debug JSON are not equality targets.

Phase 0 baseline, generated offline with the `real_robot` conda environment:

| Episode | Profile | Processed rows | Zarr admission |
|---|---|---:|---|
| 20260827_165510 | joint / rgb / pointcloud | 272 each | accepted each |
| 20260827_194525 | joint | 286 | accepted |
| 20260827_194525 | rgb / pointcloud | 285 each | rejected each |

The latter visual rejection already exists before this refactor: source row
`[0,1)` has `tactile_invalid` and `nonfinite_real_modality`. Preserve and compare
that result alongside successful exports. The local baseline lives under
`/tmp/dexmani_v25_golden_before`; it is not tracked source data. All 60 available
episodes have SOURCE rows only, so converter tests cover historical synthetic
HOLD/PLACEHOLDER rows separately. No hardware was exercised.
