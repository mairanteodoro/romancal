"""
Tests for exporting romancal segmentation skyvals to Dario wfiskymatch HDF5.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from astropy.time import Time
from roman_datamodels.datamodels import ImageModel, SegmentationMapModel

from romancal.source_catalog._skyvals import SKYVALS_DTYPE
from romancal.source_catalog.export_skyvals_dario import (
    HPIMAGE_DTYPE,
    export_segm_file,
    export_segm_files,
    export_segm_model,
    resolve_input_paths,
    skyvals_to_hpimage,
    write_dario_skyval_h5,
)
from romancal.source_catalog.source_catalog_step import SourceCatalogStep
from romancal.source_catalog.tests.test_source_catalog import make_test_image


@pytest.fixture
def image_model():
    """Minimal L2 image model with sources for SourceCatalogStep skyvals."""
    import astropy.units as u

    model = ImageModel.create_fake_data(shape=(101, 101))
    model.meta.exposure.start_time = Time(
        "2024-01-03T00:00:00.0", format="isot", scale="utc"
    )
    model.meta.filename = "none"
    model.meta.cal_step = {}
    for step_name in model.schema_info("required")["roman"]["meta"]["cal_step"][
        "required"
    ].info:
        model.meta.cal_step[step_name] = "INCOMPLETE"
    model.meta.cal_logs = []
    data, err = make_test_image()
    model.data = data
    model.err = err
    model.meta.photometry.conversion_megajanskys = (0.3324 * u.MJy / u.sr).value
    return model


def _make_skyvals(n=4):
    skyvals = np.zeros(n, dtype=SKYVALS_DTYPE)
    skyvals["healpix17"] = np.arange(n, dtype=np.int64) + 10
    skyvals["data"] = np.linspace(1.0, 4.0, n, dtype=np.float32)
    skyvals["err"] = np.full(n, 0.25, dtype=np.float32)
    skyvals["covfrac"] = np.full(n, 0.8, dtype=np.float32)
    return skyvals


def test_skyvals_to_hpimage_field_mapping():
    """Purpose: romancal skyvals fields must map 1:1 onto Dario hpimage fields."""
    skyvals = _make_skyvals()
    hpimage = skyvals_to_hpimage(skyvals)

    assert hpimage.dtype == HPIMAGE_DTYPE
    np.testing.assert_array_equal(hpimage["pixel"], skyvals["healpix17"])
    np.testing.assert_array_equal(hpimage["value"], skyvals["data"])
    np.testing.assert_array_equal(hpimage["unc"], skyvals["err"])


def test_write_and_read_dario_h5_roundtrip(tmp_path):
    """Purpose: written HDF5 must match Dario layout (hpimage gzip + hpcoverage)."""
    skyvals = _make_skyvals()
    hpimage = skyvals_to_hpimage(skyvals)
    coverage = np.array([1, 2, 3], dtype=np.int64)
    out = tmp_path / "demo.h5"

    write_dario_skyval_h5(out, hpimage, coverage)

    with h5py.File(out, "r") as hdf:
        assert "hpimage" in hdf
        assert "hpcoverage" in hdf
        got_image = hdf["hpimage"][:]
        got_cov = hdf["hpcoverage"][:]
        assert hdf["hpimage"].compression == "gzip"

    np.testing.assert_array_equal(got_image["pixel"], hpimage["pixel"])
    np.testing.assert_array_equal(got_image["value"], hpimage["value"])
    np.testing.assert_array_equal(got_image["unc"], hpimage["unc"])
    np.testing.assert_array_equal(got_cov, coverage)


def test_export_segm_model_from_source_catalog(image_model, tmp_path):
    """
    Purpose: skyvals produced by SourceCatalogStep export cleanly to Dario HDF5
    without requiring a pipeline Step of our own.
    """
    assert isinstance(image_model, ImageModel)
    _, segm = SourceCatalogStep.call(
        image_model,
        bkg_boxsize=50,
        kernel_fwhm=2.0,
        snr_threshold=5,
        npixels=10,
        save_results=False,
        fit_psf=False,
    )
    assert isinstance(segm, SegmentationMapModel)
    assert "skyvals" in segm

    out = export_segm_model(segm, tmp_path, stem="exposure_a")
    assert out == tmp_path / "exposure_a.h5"
    assert out.is_file()

    with h5py.File(out, "r") as hdf:
        hpimage = hdf["hpimage"][:]
        coverage = hdf["hpcoverage"][:]

    np.testing.assert_array_equal(hpimage["pixel"], segm.skyvals["healpix17"])
    np.testing.assert_array_equal(hpimage["value"], segm.skyvals["data"])
    np.testing.assert_array_equal(hpimage["unc"], segm.skyvals["err"])
    np.testing.assert_array_equal(coverage, np.asarray(segm.healpix11_cov))


def test_export_segm_file_and_l2_pairing(image_model, tmp_path, function_jail):
    """
    Purpose: file-based export works, strips _segm from stems, and can resolve
    L2 cal paths to sibling segm products.
    """
    _, segm = SourceCatalogStep.call(
        image_model,
        bkg_boxsize=50,
        kernel_fwhm=2.0,
        snr_threshold=5,
        npixels=10,
        save_results=False,
        fit_psf=False,
    )

    segm_path = tmp_path / "demo_cal_segm.asdf"
    cal_path = tmp_path / "demo_cal.asdf"
    # Write only the segm product; cal path is a pairing hint.
    segm.meta.filename = segm_path.name
    segm.save(str(segm_path))
    cal_path.write_text("placeholder", encoding="utf-8")

    outdir = tmp_path / "h5"
    written = export_segm_file(segm_path, outdir)
    assert written.name == "demo_cal.h5"

    paired = resolve_input_paths([cal_path], l2_to_segm=True)
    assert paired == [segm_path.resolve()] or paired[0].resolve() == segm_path.resolve()

    written_many = export_segm_files([cal_path], outdir, l2_to_segm=True)
    assert len(written_many) == 1
    assert written_many[0].name == "demo_cal.h5"


def test_export_rejects_missing_skyvals(tmp_path):
    """Purpose: fail loudly when compute_skyvals was disabled / fields absent."""

    class Dummy:
        pass

    dummy = Dummy()
    with pytest.raises(ValueError, match="missing skyvals"):
        export_segm_model(dummy, tmp_path)

    # Empty skyvals should also fail by default.
    class EmptySegm:
        skyvals = np.empty((0,), dtype=SKYVALS_DTYPE)
        healpix11_cov = np.empty((0,), dtype=np.int64)
        meta = type("M", (), {"filename": "empty_segm.asdf"})()

    with pytest.raises(ValueError, match="empty"):
        export_segm_model(EmptySegm(), tmp_path)
