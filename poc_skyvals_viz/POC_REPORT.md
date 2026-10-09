# POC report: romancal skyvals → Dario format → pixmatch

**Branch:** `feature/export-skyvals-dario`  
**Data:** `/grp/roman/mteodoro/DR_DATA/20261002` (F062 L2 `*_segm.asdf`)  
**Figures:** `poc_skyvals_viz/`

## Goal (Eddie’s next step)

Do **not** build pipeline steps yet. Validate:

1. skyvals already on L2 segmentation products  
2. stand-alone export to Dario’s HDF5 layout  
3. Dario `pixmatch` runs on those files and returns usable offsets  

## What we built

| Piece | Location |
| --- | --- |
| Export library | `romancal/source_catalog/export_skyvals_dario.py` |
| CLI | `roman_export_skyvals_to_dario` |
| Unit tests | `romancal/source_catalog/tests/test_export_skyvals_dario.py` (5 passed) |
| POC plots | `romancal/scripts/visualize_skyvals_dario_poc.py` |

**Field mapping**

| romancal segm | Dario `.h5` |
| --- | --- |
| `skyvals["healpix17"]` | `hpimage["pixel"]` |
| `skyvals["data"]` | `hpimage["value"]` |
| `skyvals["err"]` | `hpimage["unc"]` |
| `healpix11_cov` | `hpcoverage` |

## Test setup

- **8** overlapping F062 exposures (WFI01 + WFI10 across four visit/pointing combos)  
- ~7.5e4 HEALPix17 sky samples each; median sky **~0.204–0.214**  
- Exported to `/tmp/skyvals_dario_test_20261002/overlap_h5/`  
- Ran unmodified `wfiskymatch.tools.pixmatch` → `offsets.h5`

## Findings

### 1. Export works on real DR products

- Round-trip checks: `pixel` / `value` / `unc` / `hpcoverage` match source segm arrays exactly  
- Gzip `hpimage` layout matches Dario’s `asdf2healpix` writer  

### 2. Overlap geometry matters

- **Same-visit adjacent SCAs** share coarse `healpix11` tiles but **zero** healpix17 skyval pixels → `pixmatch` reports “overlaps” then hits a **singular matrix** (all-NaN offsets)  
- **Dithered / multi-visit** footprints with shared healpix17 cells are required  
- This set: **22** pairs with nonzero healpix17 overlap (up to ~5e4 shared pixels)

### 3. Skyval distributions (histogram fix)

- Bulk of samples sit in a tight band near **~0.20**  
- A few residual-source / bad healpixels reach **O(1)–O(100)** (10 samples `> 1` across 8 files) and previously crushed `02_skyval_histograms.png`  
- Histogram now uses a **0.5–99.5 percentile** display window and notes hidden outliers  

### 4. pixmatch succeeds when overlaps are real

Finite offsets (spread **~0.016**):

| stem (short) | offset |
| --- | --- |
| `...001_wfi01` | +7.93e-3 |
| `...001_wfi10` | -6.78e-3 |
| `...002_wfi01` | +6.92e-3 |
| `...002_wfi10` | -8.50e-3 |
| `...001_wfi01` (v2) | +4.96e-3 |
| `...001_wfi10` (v2) | -4.17e-3 |
| `...002_wfi01` (v2) | +4.27e-3 |
| `...002_wfi10` (v2) | -4.79e-3 |

### 5. Offset application sign

On shared-pixel differences, **adding** the pixmatch epsilon to skyvals improves pairwise median residuals; subtracting makes them worse on this set.

| metric (22 pairs) | before | after (`value + offset`) |
| --- | --- | --- |
| mean \|median Δ\| | 5.9e-3 | **2.0e-3** (~3× better) |
| WFI10–WFI10 mean \|median Δ\| | 3.0e-3 | **1.2e-3** |
| global RMS | ~0.38 | ~0.38 (dominated by rare outliers) |

**Action for later SkyMatch wiring:** confirm whether pipeline should treat epsilon as add or subtract relative to science arrays (header/SKYUSER convention in Dario’s mosaic path).

### 6. Algorithm note (not a blocker)

romancal skyvals = median of masked pixels per healpix; Dario’s full path uses block-mean + interpolate. Format compatibility is demonstrated; science parity is still open.

## Figures

| File | Content |
| --- | --- |
| `01_skyval_footprints.png` | RA/Dec HEALPix samples |
| `02_skyval_histograms.png` | Robust-range sky distributions (**fixed**) |
| `03_overlap_matrix.png` | Pairwise healpix17 overlap counts |
| `04_overlap_diffs_before_after.png` | Shared-pixel Δsky before/after offsets |
| `05_pixmatch_offsets.png` | Solved offsets |
| `06_sky_levels_before_after.png` | Median sky vs median+offset |

## Verdict vs Eddie’s ask

| Step | Status |
| --- | --- |
| 1. skyvals on L2 | Done (existing `compute_skyvals`) |
| Stand-alone export to Dario format | **Done** |
| Run Dario step 2 | **Done** (finite offsets on overlapping set) |
| 3. romancal `SkyMatch` consumes offset files | **Not started** (next) |

## Recommended next steps

1. Agree offset **sign** and units with Eddie/Dario for L2/L3 application  
2. Prototype reading `offsets.h5` in `SkyMatchStep` (or sibling) without replacing current modes  
3. Optional: filter extreme skyval outliers before match; expand POC to more filters / full ASN  
4. Keep export as a script until the offset handoff is proven in mosaic products  
