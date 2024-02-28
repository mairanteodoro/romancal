import os
from pathlib import Path

import numpy as np
import pytest
from astropy.units import Quantity

from romancal.datamodels import ModelContainer
from romancal.outlier_detection import OutlierDetectionStep, outlier_detection


@pytest.fixture()
def clean_up_after_test():
    """
    Clean up working directory after test.
    """

    def _clean_up_working_directory(pattern="*.asdf"):
        """
        Clean up the working directory by removing files matching the specified pattern.
        Parameters
        ----------
        pattern : str, optional
            The pattern of the file pattern to match (default is "*.asdf")

        Returns
        -------
        None

        """
        for x in (Path(os.getcwd())).glob(pattern):
            if x.exists():
                x.unlink()

    return _clean_up_working_directory


@pytest.mark.parametrize(
    "input_models",
    [
        list(),
        [""],
        "",
        "testing",
    ],
)
def test_outlier_raises_error_on_invalid_input_models(input_models):
    """Test that OutlierDetection raises an error if input cannot be parsed into a ModelContainer."""

    step = OutlierDetectionStep()

    with pytest.raises(Exception) as exec_info:
        step(input_models)

    assert step.skip is True
    assert exec_info.type == AttributeError


def test_outlier_skips_step_on_invalid_number_of_elements_in_input(base_image):
    """Test that OutlierDetection skips processing when provided with an invalid number of elements in the input,
    and sets the appropriate metadata for the skipped step."""
    img = base_image()

    step = OutlierDetectionStep()
    step(ModelContainer([img]))

    assert step.skip is True
    assert step.input_models[0].meta.cal_step.outlier_detection == "SKIPPED"


def test_outlier_skips_step_on_exposure_type_different_from_wfi_image(base_image):
    """
    Test if the outlier detection step is skipped when the exposure type is different from WFI image.
    """
    img_1 = base_image()
    img_1.meta.exposure.type = "WFI_PRISM"
    img_2 = base_image()
    img_2.meta.exposure.type = "WFI_PRISM"

    step = OutlierDetectionStep()
    step(ModelContainer([img_1, img_2]))

    assert step.input_models[0].meta.cal_step.outlier_detection == "SKIPPED"
    assert step.skip


def test_outlier_valid_input_asn(tmp_path, base_image, create_mock_asn_file):
    """
    Test that OutlierDetection runs with valid ASN file as input.
    """
    img_1 = base_image()
    img_1.meta.filename = "img_1.asdf"
    img_1.save(tmp_path / "img_1.asdf")
    img_2 = base_image()
    img_1.meta.filename = "img_2.asdf"
    img_2.save(tmp_path / "img_2.asdf")

    asn_filepath = create_mock_asn_file(tmp_path)

    step = OutlierDetectionStep()
    # set output dir for all files created by the step
    step.output_dir = tmp_path.as_posix()
    # make sure resample does not save file to disk
    step.in_memory = True
    step(asn_filepath)

    assert step.skip is False
    assert all(
        x.meta.cal_step.outlier_detection == "COMPLETE" for x in step.input_models
    )


def test_outlier_valid_input_modelcontainer(tmp_path, base_image):
    """
    Test that OutlierDetection runs with valid ModelContainer as input.
    """
    img_1 = base_image()
    img_1.meta.filename = "img_1.asdf"
    img_2 = base_image()
    img_1.meta.filename = "img_2.asdf"

    mc = ModelContainer([img_1, img_2])

    step = OutlierDetectionStep()
    # make sure resample does not save file to disk
    step.in_memory = True
    step(mc)

    assert step.skip is False
    assert all(
        x.meta.cal_step.outlier_detection == "COMPLETE" for x in step.input_models
    )


@pytest.mark.parametrize(
    "pars",
    [
        {
            "weight_type": "exptime",
            "pixfrac": 1.0,
            "kernel": "square",
            "fillval": "INDEF",
            "nlow": 0,
            "nhigh": 0,
            "maskpt": 0.7,
            "grow": 1,
            "snr": "4.0 3.0",
            "scale": "0.5 0.4",
            "backg": 0.0,
            "kernel_size": "7 7",
            "save_intermediate_results": False,
            "resample_data": True,
            "good_bits": 0,
            "allowed_memory": None,
            "in_memory": True,
            "make_output_path": None,
            "resample_suffix": "i2d",
        },
        {
            "weight_type": "exptime",
            "save_intermediate_results": True,
            "make_output_path": None,
            "resample_suffix": "some_other_suffix",
        },
    ],
)
def test_outlier_init_default_parameters(pars, base_image):
    """
    Test parameter setting on initialization for OutlierDetection.
    """
    img_1 = base_image()
    img_1.meta.filename = "img_1.asdf"
    input_models = ModelContainer([img_1])

    step = outlier_detection.OutlierDetection(input_models, **pars)

    assert step.input_models == input_models
    assert step.outlierpars == pars
    assert step.make_output_path == pars["make_output_path"]
    assert step.resample_suffix == f"_outlier_{pars['resample_suffix']}.asdf"


def test_outlier_do_detection_write_files_to_custom_location(
    tmp_path, base_image, clean_up_after_test
):
    """
    Test that OutlierDetection can create files on disk in a custom location.
    """
    img_1 = base_image()
    img_1.meta.filename = "img_1.asdf"
    img_2 = base_image()
    img_2.meta.filename = "img_2.asdf"
    input_models = ModelContainer([img_1, img_2])

    outlier_step = OutlierDetectionStep()
    # set output dir for all files created by the step
    outlier_step.output_dir = tmp_path.as_posix()
    # make sure files are written out to disk
    outlier_step.in_memory = False

    pars = {
        "weight_type": "exptime",
        "pixfrac": 1.0,
        "kernel": "square",
        "fillval": "INDEF",
        "nlow": 0,
        "nhigh": 0,
        "maskpt": 0.7,
        "grow": 1,
        "snr": "4.0 3.0",
        "scale": "0.5 0.4",
        "backg": 0.0,
        "kernel_size": "7 7",
        "save_intermediate_results": False,
        "resample_data": True,
        "good_bits": 0,
        "allowed_memory": None,
        "in_memory": outlier_step.in_memory,
        "make_output_path": outlier_step.make_output_path,
        "resample_suffix": "i2d",
    }

    blot_path_1 = tmp_path / img_1.meta.filename.replace(".asdf", "_blot.asdf")
    blot_path_2 = tmp_path / img_2.meta.filename.replace(".asdf", "_blot.asdf")
    median_path = tmp_path / "img_median.asdf"

    outlier_files_path = [
        blot_path_1,
        blot_path_2,
        median_path,
    ]

    detection_step = outlier_detection.OutlierDetection
    step = detection_step(input_models, **pars)

    step.do_detection()

    assert all(x.exists() for x in outlier_files_path)

    clean_up_after_test("*.asdf")


def test_outlier_do_detection_find_outliers(tmp_path, base_image, clean_up_after_test):
    """
    Test that OutlierDetection can find outliers.
    """
    img_1 = base_image()
    img_1.meta.filename = "img_1.asdf"
    img_2 = base_image()
    img_2.meta.filename = "img_2.asdf"

    # add outliers
    img_1_input_coords = np.array(
        [(5, 45), (25, 25), (45, 85), (65, 65), (85, 5)], dtype=[("x", int), ("y", int)]
    )
    img_2_input_coords = np.array(
        [(15, 25), (35, 5), (75, 65), (95, 45), (99, 5)], dtype=[("x", int), ("y", int)]
    )
    img_1.data[img_1_input_coords["x"], img_1_input_coords["y"]] = Quantity(
        100000, "electron / s"
    )
    img_2.data[img_2_input_coords["x"], img_2_input_coords["y"]] = Quantity(
        100000, "electron / s"
    )

    input_models = ModelContainer([img_1, img_2])

    outlier_step = OutlierDetectionStep()
    # set output dir for all files created by the step
    outlier_step.output_dir = tmp_path.as_posix()
    # make sure files are written out to disk
    outlier_step.in_memory = False

    pars = {
        "weight_type": "exptime",
        "pixfrac": 1.0,
        "kernel": "square",
        "fillval": "INDEF",
        "nlow": 0,
        "nhigh": 0,
        "maskpt": 0.7,
        "grow": 1,
        "snr": "4.0 3.0",
        "scale": "0.5 0.4",
        "backg": 0.0,
        "kernel_size": "7 7",
        "save_intermediate_results": False,
        "resample_data": True,
        "good_bits": 0,
        "allowed_memory": None,
        "in_memory": outlier_step.in_memory,
        "make_output_path": outlier_step.make_output_path,
        "resample_suffix": "i2d",
    }

    detection_step = outlier_detection.OutlierDetection
    step = detection_step(input_models, **pars)

    step.do_detection()

    # get flagged outliers coordinates from DQ array
    img_1_outlier_output_coords = np.where(step.input_models[0].dq > 0)

    # reformat output and input coordinates and sort by x coordinate
    outliers_output_coords = np.array(
        [(x, y) for x, y in zip(*img_1_outlier_output_coords)],
        dtype=[("x", int), ("y", int)],
    )
    outliers_input_coords = np.concatenate((img_1_input_coords, img_2_input_coords))

    outliers_output_coords.sort(axis=0)
    outliers_input_coords.sort(axis=0)

    assert all(outliers_input_coords == outliers_output_coords)

    clean_up_after_test("*.asdf")
