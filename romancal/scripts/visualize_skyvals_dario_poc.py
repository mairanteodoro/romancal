#!/usr/bin/env python
"""
POC visualizations for the Dario skyval export + pixmatch workflow.

Reads Dario-format skyval HDF5 files (from ``roman_export_skyvals_to_dario``)
and an optional ``offsets.h5`` from ``wfiskymatch.tools.pixmatch``, then writes
summary PNGs.

Examples
--------
$ uv run python -m romancal.scripts.visualize_skyvals_dario_poc \\
    --h5-dir /tmp/skyvals_dario_test_20261002/overlap_h5 \\
    --outdir ./poc_skyvals_viz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy_healpix import HEALPix

log = logging.getLogger(__name__)

HEALPIX17_NSIDE = 2**17


def _load_skyval_dir(h5_dir: Path) -> dict[str, dict]:
    """Load all skyval ``.h5`` files (excluding offsets) from a directory."""
    data = {}
    for path in sorted(h5_dir.glob("*.h5")):
        if path.name.startswith("offsets") or path.stem == "offsets":
            continue
        with h5py.File(path, "r") as hdf:
            hpimage = hdf["hpimage"][:]
            coverage = hdf["hpcoverage"][:]
        data[path.stem] = {
            "path": path,
            "pixel": np.asarray(hpimage["pixel"], dtype=np.int64),
            "value": np.asarray(hpimage["value"], dtype=np.float64),
            "unc": np.asarray(hpimage["unc"], dtype=np.float64),
            "coverage": np.asarray(coverage, dtype=np.int64),
        }
    if not data:
        raise FileNotFoundError(f"No skyval HDF5 files found in {h5_dir}")
    return data


def _load_offsets(h5_dir: Path) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Load pixmatch offsets if present."""
    candidates = [h5_dir / "offsets.h5", h5_dir / "offsets"]
    for path in candidates:
        if not path.exists():
            continue
        with h5py.File(path, "r") as hdf:
            offsets = np.asarray(hdf["offsets"][:], dtype=np.float64)
            names = [
                n.decode() if isinstance(n, bytes | bytearray) else str(n)
                for n in hdf["files"][:]
            ]
        return offsets, names
    return None, None


def _short_label(name: str) -> str:
    """Compress long Roman product stems for plot labels."""
    parts = name.split("_")
    # keep visit-ish + detector + filter tails when possible
    if len(parts) >= 4:
        return "_".join(parts[-4:])
    return name


def _pixels_to_lonlat(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    healpix = HEALPix(nside=HEALPIX17_NSIDE, order="nested", frame="icrs")
    lon, lat = healpix.healpix_to_lonlat(pixels)
    return lon.to_value(u.deg), lat.to_value(u.deg)


def _shared_radec_limits(
    lon_list: list[np.ndarray], lat_list: list[np.ndarray], pad_frac: float = 0.05
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return shared (xlim, ylim, dec_mid) covering all lon/lat samples."""
    ra_cat = np.concatenate(lon_list)
    dec_cat = np.concatenate(lat_list)
    pad_ra = pad_frac * (ra_cat.max() - ra_cat.min() + 1e-12)
    pad_dec = pad_frac * (dec_cat.max() - dec_cat.min() + 1e-12)
    ra0, ra1 = float(ra_cat.min() - pad_ra), float(ra_cat.max() + pad_ra)
    dec0, dec1 = float(dec_cat.min() - pad_dec), float(dec_cat.max() + pad_dec)
    dec_mid = 0.5 * (dec0 + dec1)
    # Expand the shorter sky axis so the window is roughly square on-sky.
    ra_span_sky = (ra1 - ra0) * np.cos(np.deg2rad(dec_mid))
    dec_span = dec1 - dec0
    if ra_span_sky >= dec_span:
        extra = 0.5 * (ra_span_sky - dec_span)
        dec0 -= extra
        dec1 += extra
    else:
        extra = 0.5 * (dec_span - ra_span_sky) / max(np.cos(np.deg2rad(dec_mid)), 1e-6)
        ra0 -= extra
        ra1 += extra
    return (ra0, ra1), (dec0, dec1), dec_mid


def _visit_key(name: str) -> str:
    """Group exposure stems like ``r..._0001_wfi01_f062`` by visit (drop detector/filter)."""
    parts = name.split("_")
    # Expect ..._<expid>_wfiXX_<filter>; fall back to full name if too short.
    if len(parts) >= 4 and parts[-2].lower().startswith("wfi"):
        return "_".join(parts[:-2])
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return name


def _detector_key(name: str) -> str:
    parts = name.split("_")
    for p in parts:
        if p.lower().startswith("wfi"):
            return p.lower()
    return "unknown"


def plot_sky_footprints(
    data: dict[str, dict],
    outdir: Path,
    max_points: int = 12000,
    cal_dir: Path | None = None,
) -> Path:
    """Scatter RA/Dec skyval footprints, one panel per visit (both detectors)."""
    from matplotlib.lines import Line2D

    # Detector colors so both SCAs are distinguishable in a visit panel.
    det_color = {
        "wfi01": "C0",
        "wfi10": "C3",
    }

    # Group exposures by visit.
    visits: dict[str, list[str]] = {}
    for name in data:
        visits.setdefault(_visit_key(name), []).append(name)
    visit_names = sorted(visits)
    n = len(visit_names)
    ncols = min(2, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # Precompute lon/lat samples (cap points per exposure).
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, d in data.items():
        idx = np.arange(d["pixel"].size)
        if idx.size > max_points:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(idx, size=max_points, replace=False))
        lon, lat = _pixels_to_lonlat(d["pixel"][idx])
        samples[name] = (lon, lat)

    xlim, ylim, dec_mid = _shared_radec_limits(
        [s[0] for s in samples.values()], [s[1] for s in samples.values()]
    )
    aspect = 1.0 / np.cos(np.deg2rad(dec_mid))

    # Optional L2 WCS outlines (true detector footprints).
    outlines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if cal_dir is not None:
        try:
            from roman_datamodels import datamodels as rdm
        except ImportError:
            rdm = None
        if rdm is not None:
            for name in data:
                cal_path = Path(cal_dir) / f"{name}_cal.asdf"
                if not cal_path.exists():
                    continue
                with rdm.open(cal_path) as model:
                    wcs = model.meta.wcs
                    ny, nx = model.data.shape
                    xy = np.array(
                        [[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1], [0, 0]],
                        dtype=float,
                    )
                    ra, dec = wcs.pixel_to_world_values(xy[:, 0], xy[:, 1])
                outlines[name] = (np.asarray(ra), np.asarray(dec))

    for ax, visit in zip(axes.ravel(), visit_names, strict=False):
        members = sorted(visits[visit])
        handles = []
        for name in members:
            lon, lat = samples[name]
            det = _detector_key(name)
            color = det_color.get(det, "0.3")
            ax.scatter(
                lon,
                lat,
                s=2,
                c=color,
                linewidths=0,
                rasterized=True,
                alpha=0.35,
                zorder=1,
            )
            if name in outlines:
                ra, dec = outlines[name]
                ax.plot(ra, dec, "-", color=color, lw=2.0, zorder=3)
                ax.plot(ra[:-1], dec[:-1], "o", color=color, ms=3, zorder=4)
            else:
                # Fallback: axis-aligned envelope of healpix samples.
                ra0, ra1 = float(np.min(lon)), float(np.max(lon))
                dec0, dec1 = float(np.min(lat)), float(np.max(lat))
                ax.plot(
                    [ra0, ra1, ra1, ra0, ra0],
                    [dec0, dec0, dec1, dec1, dec0],
                    "-",
                    color=color,
                    lw=1.6,
                    zorder=3,
                )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=2.0,
                    marker="o",
                    ms=4,
                    label=det.upper(),
                )
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(aspect, adjustable="box", anchor="C")
        ax.set_title(f"visit {visit}", fontsize=10)
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend(handles=handles, fontsize=8, loc="upper right", frameon=True)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(
        "HEALPix skyvals by visit (both detectors; shared RA/Dec axes)",
        fontsize=13,
    )
    fig.tight_layout()
    out = outdir / "01_skyval_footprints.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_value_histograms(data: dict[str, dict], outdir: Path) -> Path:
    """Per-exposure sky value histograms on a robust (outlier-clipped) range."""
    all_vals = np.concatenate([d["value"] for d in data.values()])
    # Rare residual-source / edge healpixels can reach O(100) and crush the axis.
    lo, hi = np.nanpercentile(all_vals, [0.5, 99.5])
    # Keep a little padding so the bulk of the distribution is visible.
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    xmin, xmax = lo - pad, hi + pad
    n_out = int(np.sum((all_vals < xmin) | (all_vals > xmax)))

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(xmin, xmax, 81)
    for name, d in data.items():
        vals = d["value"]
        # Clip only for display; density is over the displayed window.
        in_win = vals[(vals >= xmin) & (vals <= xmax)]
        ax.hist(
            in_win,
            bins=bins,
            histtype="step",
            density=True,
            label=_short_label(name),
            linewidth=1.4,
        )
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("sky value")
    ax.set_ylabel("density")
    ax.set_title(
        "Skyval distributions by exposure "
        f"(display {xmin:.4g}-{xmax:.4g}; {n_out} outlier sample(s) hidden)"
    )
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    out = outdir / "02_skyval_histograms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_overlap_matrix(data: dict[str, dict], outdir: Path) -> Path:
    """Heatmap of pairwise healpix17 pixel overlap counts."""
    names = list(data)
    n = len(names)
    sets = [set(data[name]["pixel"].tolist()) for name in names]
    mat = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        mat[i, i] = len(sets[i])
        for j in range(i + 1, n):
            n_ov = len(sets[i] & sets[j])
            mat[i, j] = mat[j, i] = n_ov

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.log10(mat + 1.0), cmap="magma")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [_short_label(name) for name in names]
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{mat[i, j]}", ha="center", va="center", fontsize=7, color="w"
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log10(N_overlap + 1)")
    ax.set_title("Pairwise HEALPix17 skyval overlaps")
    fig.tight_layout()
    out = outdir / "03_overlap_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _pairwise_diffs(
    data: dict[str, dict],
    offsets: dict[str, float] | None = None,
    max_pairs: int = 12,
    max_samples: int = 20000,
) -> list[tuple[str, str, np.ndarray]]:
    """Collect sky differences on shared healpix pixels, optionally offset-corrected."""
    names = list(data)
    # Sort once so intersect1d + searchsorted gathers are O(N log N).
    sorted_pix = {}
    sorted_val = {}
    for name in names:
        order = np.argsort(data[name]["pixel"], kind="mergesort")
        sorted_pix[name] = data[name]["pixel"][order]
        sorted_val[name] = data[name]["value"][order]

    pairs: list[tuple[str, str, np.ndarray]] = []
    for i in range(len(names)):
        a = names[i]
        pix_a = sorted_pix[a]
        val_a = sorted_val[a]
        off_a = 0.0 if offsets is None else offsets.get(a, 0.0)
        for j in range(i + 1, len(names)):
            b = names[j]
            pix_b = sorted_pix[b]
            val_b = sorted_val[b]
            off_b = 0.0 if offsets is None else offsets.get(b, 0.0)
            inter = np.intersect1d(pix_a, pix_b, assume_unique=True)
            if inter.size < 50:
                continue
            if inter.size > max_samples:
                rng = np.random.default_rng(i * 1009 + j)
                inter = np.sort(rng.choice(inter, size=max_samples, replace=False))
            ia = np.searchsorted(pix_a, inter)
            ib = np.searchsorted(pix_b, inter)
            # pixmatch epsilon improves pairwise medians when ADDED to skyvals
            # (i.e. corrected = value + offset). Confirm sign when wiring SkyMatch.
            diffs = (val_a[ia] + off_a) - (val_b[ib] + off_b)
            pairs.append((a, b, diffs))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def plot_before_after_diffs(
    data: dict[str, dict],
    offsets: np.ndarray | None,
    offset_names: list[str] | None,
    outdir: Path,
) -> Path | None:
    """Histograms of pairwise sky differences before and after applying offsets."""
    if offsets is None or offset_names is None:
        log.warning("No offsets.h5 found; skipping before/after difference plot")
        return None

    # Map offsets onto skyval stems (pixmatch stores basenames without .h5)
    off_map = {}
    for name, off in zip(offset_names, offsets, strict=False):
        stem = Path(name).stem
        off_map[stem] = float(off)
        # also allow exact keys present in data
        if name in data:
            off_map[name] = float(off)

    before = _pairwise_diffs(data, offsets=None)
    after = _pairwise_diffs(data, offsets=off_map)
    if not before:
        log.warning("No overlapping pairs with enough shared pixels")
        return None

    n = len(before)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.4 * n), sharex=False, squeeze=False)
    for row, ((a, b, d0), (_, _, d1)) in enumerate(zip(before, after, strict=True)):
        label = f"{_short_label(a)}\nvs {_short_label(b)}"
        for col, (diffs, title, color) in enumerate(
            [
                (d0, "before match", "C0"),
                (d1, "after match", "C1"),
            ]
        ):
            ax = axes[row, col]
            ax.hist(diffs, bins=60, color=color, alpha=0.85)
            med = float(np.nanmedian(diffs))
            mad = float(np.nanmedian(np.abs(diffs - med)))
            ax.axvline(0.0, color="k", lw=0.8, ls="--")
            ax.axvline(med, color="r", lw=1.0, ls="-")
            ax.set_title(f"{title}: med={med:.4g}, MAD={mad:.4g}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("delta sky (A - B)")
    fig.suptitle("Overlap sky differences before/after pixmatch offsets", fontsize=13)
    fig.tight_layout()
    out = outdir / "04_overlap_diffs_before_after.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_offsets_bar(
    offsets: np.ndarray | None,
    offset_names: list[str] | None,
    outdir: Path,
) -> Path | None:
    """Bar chart of solved sky offsets."""
    if offsets is None or offset_names is None:
        return None
    labels = [_short_label(Path(n).stem) for n in offset_names]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(offsets))
    colors = ["C0" if v >= 0 else "C3" for v in offsets]
    ax.bar(x, offsets, color=colors, edgecolor="k", linewidth=0.4)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("offset")
    ax.set_title("pixmatch solved sky offsets")
    fig.tight_layout()
    out = outdir / "05_pixmatch_offsets.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_matched_sky_levels(
    data: dict[str, dict],
    offsets: np.ndarray | None,
    offset_names: list[str] | None,
    outdir: Path,
) -> Path | None:
    """Median sky per exposure before and after adding pixmatch offsets."""
    if offsets is None or offset_names is None:
        return None
    off_map = {
        Path(n).stem: float(o) for n, o in zip(offset_names, offsets, strict=False)
    }
    names = list(data)
    before = [float(np.nanmedian(data[n]["value"])) for n in names]
    after = [float(np.nanmedian(data[n]["value"]) + off_map.get(n, 0.0)) for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, before, width, label="median skyvals", color="C0")
    ax.bar(x + width / 2, after, width, label="median + offset", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_label(n) for n in names], rotation=60, ha="right", fontsize=8
    )
    ax.set_ylabel("sky level")
    ax.set_title("Exposure sky levels before/after applying offsets")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = outdir / "06_sky_levels_before_after.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run(h5_dir: Path, outdir: Path) -> list[Path]:
    """Generate all POC figures; return written paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    data = _load_skyval_dir(h5_dir)
    offsets, offset_names = _load_offsets(h5_dir)

    written = [
        plot_sky_footprints(data, outdir),
        plot_value_histograms(data, outdir),
        plot_overlap_matrix(data, outdir),
    ]
    for fn in (
        plot_before_after_diffs,
        plot_offsets_bar,
        plot_matched_sky_levels,
    ):
        path = fn(data, offsets, offset_names, outdir)
        if path is not None:
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5-dir",
        type=Path,
        required=True,
        help="Directory with Dario skyval .h5 files and optional offsets.h5",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory for output PNG figures",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    written = run(args.h5_dir, args.outdir)
    print(f"Wrote {len(written)} figure(s) to {args.outdir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
