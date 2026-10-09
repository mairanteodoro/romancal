#!/usr/bin/env python
"""
Export romancal segmentation skyvals to Dario wfiskymatch HDF5 skyval files.

Example
-------
$ roman_export_skyvals_to_dario --outdir ./h5_skyvals *_segm.asdf
$ roman_export_skyvals_to_dario --outdir ./h5_skyvals --l2-to-segm *_cal.asdf

After export, Dario step 2 can be run (with wfiskymatch installed) as::

    from pathlib import Path
    from wfiskymatch.tools import pixmatch
    files = sorted(str(p) for p in Path('./h5_skyvals').glob('*.h5'))
    pixmatch(files, outfile='offsets')
"""

from __future__ import annotations

import argparse
import logging

from romancal.source_catalog.export_skyvals_dario import export_segm_files


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export skyvals/healpix11_cov from L2 segmentation products into "
            "Dario wfiskymatch-format HDF5 files (hpimage + hpcoverage)."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Segmentation ASDF files, directories, or globs. "
            "With --l2-to-segm, L2 cal paths are mapped to sibling *_segm.asdf."
        ),
    )
    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        help="Directory for output .h5 skyval files.",
    )
    parser.add_argument(
        "--l2-to-segm",
        action="store_true",
        help="Resolve L2-like inputs to sibling segmentation products.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write HDF5 files even when skyvals arrays are empty.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )
    args = parser.parse_args(argv)

    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = log_levels[min(len(log_levels) - 1, args.verbose)]
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    try:
        written = export_segm_files(
            args.inputs,
            args.outdir,
            l2_to_segm=args.l2_to_segm,
            allow_empty=args.allow_empty,
        )
    except (OSError, ValueError) as err:
        logging.error("%s", err)
        return 1

    print(f"Wrote {len(written)} skyval file(s) to {args.outdir}")
    for path in written:
        print(path)
    return 0


def main(argv: list[str] | None = None) -> None:
    """Console-script entry that exits with the export status code."""
    raise SystemExit(_main(argv))


if __name__ == "__main__":
    main()
