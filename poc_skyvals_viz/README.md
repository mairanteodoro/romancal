# Skyval export + pixmatch POC figures

Generated from overlapping F062 L2 segmentation skyvals under
`/grp/roman/mteodoro/DR_DATA/20261002`, exported to Dario HDF5 format and
matched with `wfiskymatch.tools.pixmatch`.

## Regenerate

```bash
uv run python -m romancal.scripts.visualize_skyvals_dario_poc \
  --h5-dir /tmp/skyvals_dario_test_20261002/overlap_h5 \
  --outdir ./poc_skyvals_viz
```

## Figures

| File | Meaning |
| --- | --- |
| `01_skyval_footprints.png` | RA/Dec HEALPix skyval samples per exposure |
| `02_skyval_histograms.png` | Sky value distributions |
| `03_overlap_matrix.png` | Pairwise HEALPix17 overlap counts |
| `04_overlap_diffs_before_after.png` | Shared-pixel Δsky before/after offsets |
| `05_pixmatch_offsets.png` | Solved per-exposure offsets |
| `06_sky_levels_before_after.png` | Median sky before vs median−offset |
