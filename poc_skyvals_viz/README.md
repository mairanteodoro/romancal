# Skyval export + pixmatch POC figures

Generated from overlapping F062 L2 segmentation skyvals under
`/grp/roman/mteodoro/DR_DATA/20261002`, exported to Dario HDF5 format and
matched with `wfiskymatch.tools.pixmatch`.

## Regenerate

Core panels from the packaged script:

```bash
uv run python -m romancal.scripts.visualize_skyvals_dario_poc \
  --h5-dir /tmp/skyvals_dario_test_20261002/overlap_h5 \
  --outdir ./poc_skyvals_viz
```

Additional combined / footprint figures in this directory were produced with
one-off plotting commands during the POC (same HDF5 inputs + optional L2
`*_cal.asdf` WCS outlines from the DR data path).

## Figures

| File | Meaning |
| --- | --- |
| `01_combined_skyvals_skymatch.png` | Dario-style **combined** HEALPix skyvals on one sky grid: median map and local std **before/after** `value + offset` |
| `01_combined_skyvals_with_healpix_cells.png` | Combined skyvals with **HEALPix11** coverage outlines plus zoom of true **HEALPix17** cell polygons |
| `01_skyval_footprints.png` | Per-visit RA/Dec skyval footprints (both detectors) with shared axes; optional L2 WCS outlines |
| `01b_combined_skyvals_scatter_before_after.png` | Combined-grid **local std** only, before vs after pixmatch |
| `02_skyval_histograms.png` | Skyval value distributions (robust display range; outlier tails hidden) |
| `03_overlap_matrix.png` | Pairwise HEALPix17 overlap counts between exposures |
| `04_overlap_diffs_before_after.png` | Shared-healpix Δsky histograms before/after applying offsets |
| `05_pixmatch_offsets.png` | Bar chart of solved per-exposure pixmatch offsets |
| `06_sky_levels_before_after.png` | Per-exposure median sky vs median + offset |
| `07_visit_footprints_healpix.png` | Per-visit L2 WCS detector footprints + HEALPix17 samples (shared RA/Dec axes) |
| `07b_all_visits_footprints_healpix.png` | All visits overlaid: footprints (line style = visit, color = detector) + HEALPix cells |

## Notes

- `POC_REPORT.md` is **not** tracked (gitignored permanently).
- Combined maps show **background skyval samples**, not science images.
- `pixmatch` applies a **constant per exposure**; residual spatial structure and hot skyvals inside a detector are expected after matching.
