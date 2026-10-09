"""
Export romancal segmentation skyvals into Dario wfiskymatch HDF5 files.

Dario's ``asdf2healpix`` / ``pixmatch`` expect one ``.h5`` per exposure with:

* ``hpimage`` — structured array fields ``pixel``, ``value``, ``unc``
  (NEST healpix index at nside=2**17, sky value, uncertainty)
* ``hpcoverage`` — 1-D int64 NEST indices at nside=2**11

romancal stores the same information on L2 segmentation products as:

* ``skyvals`` — structured ``(healpix17, data, err, covfrac)``
* ``healpix11_cov`` — coarse footprint indices
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from roman_datamodels import datamodels

log = logging.getLogger(__name__)

HPIMAGE_DTYPE = np.dtype(
    [
        ("pixel", np.int64),
        ("value", np.float32),
        ("unc", np.float32),
    ]
)


def _model_has(model, name: str) -> bool:
    """Return True if ``name`` is present on a datamodel or plain object."""
    contains = getattr(model, "__contains__", None)
    if callable(contains):
        try:
            return name in model
        except TypeError:
            pass
    return hasattr(model, name)


__all__ = [
    "HPIMAGE_DTYPE",
    "export_segm_file",
    "export_segm_files",
    "export_segm_model",
    "resolve_input_paths",
    "skyvals_to_hpimage",
    "write_dario_skyval_h5",
]


def skyvals_to_hpimage(skyvals: np.ndarray) -> np.ndarray:
    """
    Map a romancal ``skyvals`` structured array to Dario ``hpimage``.

    Parameters
    ----------
    skyvals
        Structured array with at least ``healpix17``, ``data``, and ``err``.

    Returns
    -------
    hpimage
        Structured array with fields ``pixel``, ``value``, ``unc``.
    """
    required = ("healpix17", "data", "err")
    missing = [name for name in required if name not in skyvals.dtype.names]
    if missing:
        raise ValueError(f"skyvals is missing required field(s): {', '.join(missing)}")

    hpimage = np.empty(skyvals.shape[0], dtype=HPIMAGE_DTYPE)
    hpimage["pixel"] = np.asarray(skyvals["healpix17"], dtype=np.int64)
    hpimage["value"] = np.asarray(skyvals["data"], dtype=np.float32)
    hpimage["unc"] = np.asarray(skyvals["err"], dtype=np.float32)
    return hpimage


def write_dario_skyval_h5(
    outfile: str | Path,
    hpimage: np.ndarray,
    hpcoverage: np.ndarray,
) -> Path:
    """
    Write a Dario-format skyval HDF5 file.

    Parameters
    ----------
    outfile
        Destination path (``.h5``).
    hpimage
        Structured array with ``pixel``, ``value``, ``unc``.
    hpcoverage
        1-D integer coverage indices (nside=2**11).

    Returns
    -------
    path
        Resolved output path.
    """
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if hpimage.dtype.names is None or set(HPIMAGE_DTYPE.names) - set(
        hpimage.dtype.names
    ):
        raise ValueError(
            "hpimage must be a structured array with fields "
            f"{HPIMAGE_DTYPE.names}; got {hpimage.dtype}"
        )

    coverage = np.asarray(hpcoverage, dtype=np.int64).reshape(-1)
    # Match Dario's asdf2healpix layout: gzip-compressed hpimage plus a plain
    # int64 hpcoverage vector.
    with h5py.File(outfile, "w") as hdf5_file:
        hdf5_file.create_dataset("hpimage", data=hpimage, compression="gzip")
        coverage_ds = hdf5_file.create_dataset(
            "hpcoverage", (coverage.size,), dtype="int64"
        )
        coverage_ds[:] = coverage

    return outfile


def export_segm_model(
    segm_model,
    outdir: str | Path,
    *,
    stem: str | None = None,
    allow_empty: bool = False,
) -> Path:
    """
    Export skyvals from an open segmentation model to one Dario ``.h5`` file.

    Parameters
    ----------
    segm_model
        Segmentation map model carrying ``skyvals`` and ``healpix11_cov``.
    outdir
        Directory for the output ``.h5`` file.
    stem
        Basename without extension. Defaults to the model's ``meta.filename``
        stem when available, otherwise ``"skyvals"``.
    allow_empty
        If False (default), raise when ``skyvals`` is empty.

    Returns
    -------
    path
        Path of the written HDF5 file.
    """
    if not (
        _model_has(segm_model, "skyvals") and _model_has(segm_model, "healpix11_cov")
    ):
        raise ValueError(
            "Segmentation model is missing skyvals and/or healpix11_cov. "
            "Re-run SourceCatalogStep with compute_skyvals=True."
        )

    skyvals = np.asarray(segm_model.skyvals)
    healpix11_cov = np.asarray(segm_model.healpix11_cov)

    if skyvals.size == 0 and not allow_empty:
        raise ValueError("skyvals array is empty; refusing to write an empty hpimage")

    if stem is None:
        filename = getattr(getattr(segm_model, "meta", None), "filename", None)
        if filename:
            stem = Path(str(filename)).stem
            # Prefer the exposure stem without a trailing _segm suffix so
            # downstream offset tables map cleanly back to L2 names.
            if stem.endswith("_segm"):
                stem = stem[: -len("_segm")]
        else:
            stem = "skyvals"

    outpath = Path(outdir) / f"{stem}.h5"
    hpimage = skyvals_to_hpimage(skyvals)
    write_dario_skyval_h5(outpath, hpimage, healpix11_cov)
    log.info(
        "Wrote %s (n_skyvals=%d, n_coverage=%d)",
        outpath,
        hpimage.size,
        np.asarray(healpix11_cov).size,
    )
    return outpath


def export_segm_file(
    segm_path: str | Path,
    outdir: str | Path,
    *,
    stem: str | None = None,
    allow_empty: bool = False,
) -> Path:
    """
    Open a segmentation ASDF file and export its skyvals to Dario HDF5.

    Parameters
    ----------
    segm_path
        Path to a segmentation map product.
    outdir
        Output directory for the ``.h5`` file.
    stem
        Optional output basename override (no extension).
    allow_empty
        Passed through to :func:`export_segm_model`.

    Returns
    -------
    path
        Path of the written HDF5 file.
    """
    segm_path = Path(segm_path)
    if stem is None:
        stem = segm_path.stem
        if stem.endswith("_segm"):
            stem = stem[: -len("_segm")]

    with datamodels.open(segm_path) as segm_model:
        return export_segm_model(
            segm_model,
            outdir,
            stem=stem,
            allow_empty=allow_empty,
        )


def _expand_one(path_like: str | Path) -> list[Path]:
    path = Path(path_like)
    if path.is_dir():
        # Prefer segmentation products; callers can still pass explicit files.
        segm = sorted(path.glob("*_segm.asdf"))
        return segm if segm else sorted(path.glob("*.asdf"))
    if any(ch in str(path_like) for ch in "*?["):
        # glob relative to CWD; Path.glob only works for relative patterns on self
        parent = path.parent if path.parent != Path("") else Path(".")
        return sorted(parent.glob(path.name))
    return [path]


def resolve_input_paths(
    inputs: Sequence[str | Path],
    *,
    l2_to_segm: bool = False,
) -> list[Path]:
    """
    Expand CLI inputs into concrete segmentation file paths.

    Parameters
    ----------
    inputs
        Files, directories, or glob patterns.
    l2_to_segm
        If True, map ``*_cal.asdf`` (or plain L2 stems) to sibling
        ``*_segm.asdf`` paths when the L2 path itself is not a segm product.

    Returns
    -------
    paths
        Deduplicated existing paths, sorted.
    """
    found: list[Path] = []
    for item in inputs:
        found.extend(_expand_one(item))

    resolved: list[Path] = []
    for path in found:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        name = path.name
        if name.endswith("_segm.asdf") or "segm" in path.suffixes:
            resolved.append(path)
            continue

        if l2_to_segm:
            # Common Roman naming: replace _cal with _segm, or append _segm.
            candidates = []
            if "_cal" in name:
                candidates.append(path.with_name(name.replace("_cal", "_segm")))
            stem = path.stem
            candidates.append(path.with_name(f"{stem}_segm.asdf"))
            candidates.append(path.with_name(f"{stem.replace('_cal', '')}_segm.asdf"))
            match = next((c for c in candidates if c.exists()), None)
            if match is None:
                raise FileNotFoundError(
                    f"No segmentation product found for L2-like input {path}"
                )
            resolved.append(match)
            continue

        # Allow explicit non-_segm names if the file itself carries skyvals.
        resolved.append(path)

    # Prefer unique paths in stable order.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def export_segm_files(
    inputs: Sequence[str | Path],
    outdir: str | Path,
    *,
    l2_to_segm: bool = False,
    allow_empty: bool = False,
) -> list[Path]:
    """
    Export many segmentation products to Dario-format skyval HDF5 files.

    Parameters
    ----------
    inputs
        Segm paths, directories, and/or globs (see :func:`resolve_input_paths`).
    outdir
        Directory that will hold one ``.h5`` per input.
    l2_to_segm
        Attempt to pair L2 cal files to sibling segm products.
    allow_empty
        Allow empty skyvals arrays.

    Returns
    -------
    paths
        List of written HDF5 paths.
    """
    segm_paths = resolve_input_paths(inputs, l2_to_segm=l2_to_segm)
    if not segm_paths:
        raise FileNotFoundError("No input segmentation files resolved")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for segm_path in segm_paths:
        written.append(
            export_segm_file(
                segm_path,
                outdir,
                allow_empty=allow_empty,
            )
        )
    return written
