import pytest
import numpy as np
from astropy import units as u
from gwcs import WCS
from astropy.modeling import models
from astropy.time import Time
from astropy import coordinates as coord
from gwcs import coordinate_frames as cf
from roman_datamodels import datamodels, maker_utils
from romancal.resample import ResampleStep
from romancal.resample import gwcs_drizzle, resample_utils
from romancal.datamodels import ModelContainer
from asdf import AsdfFile


class WfiSca:
    def __init__(self, fiducial_world, pscale, shape, filename):
        self.fiducial_world = fiducial_world
        self.pscale = pscale
        self.shape = shape
        self.filename = filename

    def create_image(self):
        """
        Create a dummy L2 datamodel given the coordinates of the fiducial point,
        a pixel scale, and the image shape and filename.

        Returns
        -------
        datamodels.ImageModel
            An L2 ImageModel datamodel.
        """
        l2 = maker_utils.mk_level2_image(
            shape=self.shape,
            **{
                "meta": {
                    "wcsinfo": {"ra_ref": 10, "dec_ref": 0, "vparity": -1},
                    "exposure": {"exposure_time": 152.04000000000002},
                    "observation": {
                        "program": "00005",
                        "execution_plan": 1,
                        "pass": 1,
                        "observation": 1,
                        "segment": 1,
                        "visit": 1,
                        "visit_file_group": 1,
                        "visit_file_sequence": 1,
                        "visit_file_activity": "01",
                        "exposure": 1,
                    },
                },
                "data": u.Quantity(
                    np.random.poisson(2.5, size=self.shape).astype(np.float32),
                    u.electron / u.s,
                    dtype=np.float32,
                ),
                "var_rnoise": u.Quantity(
                    np.random.normal(1, 0.05, size=self.shape).astype(
                        np.float32
                    ),
                    u.electron**2 / u.s**2,
                    dtype=np.float32,
                ),
                "var_poisson": u.Quantity(
                    np.random.poisson(1, size=self.shape).astype(np.float32),
                    u.electron**2 / u.s**2,
                    dtype=np.float32,
                ),
                "var_flat": u.Quantity(
                    np.random.uniform(0, 1, size=self.shape).astype(np.float32),
                    u.electron**2 / u.s**2,
                    dtype=np.float32,
                ),
            },
        )
        # data from WFISim simulation of SCA #01
        l2.meta.filename = self.filename
        l2.meta["wcs"] = create_wcs_object_without_distortion(
            fiducial_world=self.fiducial_world,
            pscale=self.pscale,
            shape=self.shape,
        )
        return datamodels.ImageModel(l2)


class Mosaic:
    def __init__(self, fiducial_world, pscale, shape, filename, n_images):
        self.fiducial_world = fiducial_world
        self.pscale = pscale
        self.shape = shape
        self.filename = filename
        self.n_images = n_images

    def create_mosaic(self):
        """
        Create a dummy L3 datamodel given the coordinates of the fiducial point,
        a pixel scale, and the image shape and filename.

        Returns
        -------
        datamodels.MosaicModel
            An L3 MosaicModel datamodel.
        """
        l3 = maker_utils.mk_level3_mosaic(
            shape=self.shape,
            n_images=self.n_images,
        )
        # data from WFISim simulation of SCA #01
        l3.meta.filename = self.filename
        l3.meta["wcs"] = create_wcs_object_without_distortion(
            fiducial_world=self.fiducial_world,
            pscale=self.pscale,
            shape=self.shape,
        )
        l3.meta.wcs.forward_transform
        return datamodels.MosaicModel(l3)


def create_wcs_object_without_distortion(
    fiducial_world, pscale, shape, **kwargs
):
    """
    Create a simple WCS object without either distortion or rotation.

    Parameters
    ----------
    fiducial_world : tuple
        A pair of values corresponding to the fiducial's world coordinate.
    pscale : tuple
        A pair of values corresponding to the pixel scale in each axis.
    shape : tuple
        A pair of values specifying the dimensions of the WCS object.

    Returns
    -------
    gwcs.WCS
        A gwcs.WCS object.
    """
    # components of the model
    shift = models.Shift() & models.Shift()
    affine = models.AffineTransformation2D(
        matrix=[[1, 0], [0, 1]], translation=[0, 0], name="pc_rotation_matrix"
    )
    scale = models.Scale(pscale[0]) & models.Scale(pscale[1])
    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(
        fiducial_world[0],
        fiducial_world[1],
        180,
    )

    # transforms between frames
    # detector -> sky
    det2sky = shift | affine | scale | tan | celestial_rotation
    det2sky.name = "linear_transform"

    # frames
    detector_frame = cf.Frame2D(
        name="detector",
        axes_order=(0, 1),
        axes_names=("x", "y"),
        unit=(u.pix, u.pix),
    )
    sky_frame = cf.CelestialFrame(
        reference_frame=coord.FK5(), name="fk5", unit=(u.deg, u.deg)
    )

    pipeline = [
        (detector_frame, det2sky),
        (sky_frame, None),
    ]

    wcs_obj = WCS(pipeline)

    wcs_obj.bounding_box = kwargs.get(
        "bounding_box",
        (
            (-0.5, shape[-1] - 0.5),
            (-0.5, shape[-2] - 0.5),
        ),
    )

    wcs_obj.pixel_shape = kwargs.get("pixel_shape", shape[::-1])
    wcs_obj.array_shape = kwargs.get("shape", shape)

    return wcs_obj


@pytest.fixture
def wfi_sca1():
    sca = WfiSca(
        fiducial_world=(10, 0),
        pscale=(0.000031, 0.000031),
        shape=(100, 100),
        filename="r0000501001001001001_01101_0001_WFI01_cal.asdf",
    )

    return sca.create_image()


@pytest.fixture
def wfi_sca2():
    sca = WfiSca(
        fiducial_world=(10.00139, 0),
        pscale=(0.000031, 0.000031),
        shape=(100, 100),
        filename="r0000501001001001001_01101_0001_WFI02_cal.asdf",
    )

    return sca.create_image()


@pytest.fixture
def wfi_sca3():
    sca = WfiSca(
        fiducial_world=(10.00278, 0),
        pscale=(0.000031, 0.000031),
        shape=(100, 100),
        filename="r0000501001001001001_01101_0001_WFI03_cal.asdf",
    )

    return sca.create_image()


@pytest.fixture
def wfi_sca4():
    sca = WfiSca(
        fiducial_world=(10, 0),
        pscale=(0.000031, 0.000031),
        shape=(100, 100),
        filename="r0000501001001001001_01101_0002_WFI01_cal.asdf",
    )

    return sca.create_image()


@pytest.fixture
def wfi_sca5():
    sca = WfiSca(
        fiducial_world=(10.00139, 0),
        pscale=(0.000031, 0.000031),
        shape=(100, 100),
        filename="r0000501001001001001_01101_0002_WFI02_cal.asdf",
    )

    return sca.create_image()


@pytest.fixture
def wfi_sca6():
    sca = WfiSca(
        fiducial_world=(10.00278, 0),
        pscale=(0.000031, 0.000031),
        shape=(100, 100),
        filename="r0000501001001001001_01101_0002_WFI03_cal.asdf",
    )

    return sca.create_image()


@pytest.fixture
def exposure_1(wfi_sca1, wfi_sca2, wfi_sca3):
    """Returns a list with models corresponding to a dummy exposure 1."""
    # set the same exposure time for all SCAs
    for sca in [wfi_sca1, wfi_sca2, wfi_sca3]:
        sca.meta.exposure["start_time"] = Time(
            "2020-02-01T00:00:00", format="isot", scale="utc"
        )
        sca.meta.exposure["end_time"] = Time(
            "2020-02-01T00:02:30", format="isot", scale="utc"
        )
        sca.meta.observation["exposure"] = 1
    return [wfi_sca1, wfi_sca2, wfi_sca3]


@pytest.fixture
def exposure_2(wfi_sca4, wfi_sca5, wfi_sca6):
    """Returns a list with models corresponding to a dummy exposure 2."""
    # set the same exposure time for all SCAs
    for sca in [wfi_sca4, wfi_sca5, wfi_sca6]:
        sca.meta.exposure["start_time"] = Time(
            "2020-05-01T00:00:00", format="isot", scale="utc"
        )
        sca.meta.exposure["end_time"] = Time(
            "2020-05-01T00:02:30", format="isot", scale="utc"
        )
        sca.meta.observation["exposure"] = 2
    return [wfi_sca4, wfi_sca5, wfi_sca6]


@pytest.fixture
def multiple_exposures(exposure_1, exposure_2):
    """Returns a list with all the datamodels from exposure 1 and 2."""
    exposure_1.extend(exposure_2)
    return exposure_1


@pytest.mark.parametrize(
    "wcsinfo",
    [
        {
            "cd1_1": 1,
            "cd1_2": 2,
            "cd2_1": 3,
            "cd2_2": 4,
            "ap_1_1": 5,
            "ap_1_2": 6,
            "ap_2_1": 7,
            "ap_2_2": 8,
            "ap_order": 9,
        },
        {
            "ra_ref": 10,
            "dec_ref": 11,
            "v2_ref": 12,
            "v3_ref": 13,
            "roll_ref": 14,
            "v3yangle": 15,
            "vparity": 16,
        },
    ],
)
def test_update_wcs_removes_unnecessary_keywords(wcsinfo):
    step = ResampleStep()

    mosaic_model = Mosaic(
        fiducial_world=(10, 0),
        pscale=(0.000031, 0.000031),
        shape=(100, 100),
        n_images=2,
        filename="output_mosaic",
    )
    mosaic = mosaic_model.create_mosaic()
    assert True


@pytest.mark.parametrize(
    "vals, name, min_vals, expected",
    [
        ([1, 2], "list1", None, [1, 2]),
        ([None, None], "list2", None, None),
        ([1, 2], "list4", [0, 0], [1, 2]),
    ],
)
def test_check_list_pars_valid(vals, name, min_vals, expected):
    step = ResampleStep()

    result = step._check_list_pars(vals, name, min_vals)
    assert result == expected


@pytest.mark.parametrize(
    "vals, name, min_vals",
    [
        ([1, None], "list3", None),  # One value is None
        ([1, 2], "list5", [3, 3]),  # Values do not meet minimum requirements
        ([1, 2, 3], "list6", None),  # Invalid number of elements
    ],
)
def test_check_list_pars_exception(vals, name, min_vals):
    step = ResampleStep()
    with pytest.raises(ValueError):
        step._check_list_pars(vals, name, min_vals)


@pytest.fixture
def asdf_wcs_file(tmp_path):
    def _create_asdf_wcs_file(tmp_path, pixel_shape, bounding_box):
        file_path = tmp_path / "wcs.asdf"
        wcs_data = create_wcs_object_without_distortion(
            (10, 0),
            (0.000031, 0.000031),
            (100, 100),
            **{"pixel_shape": pixel_shape, "bounding_box": bounding_box},
        )
        wcs = {"wcs": wcs_data}
        with AsdfFile(wcs) as af:
            af.write_to(file_path)
        return str(file_path)

    return _create_asdf_wcs_file


def test_load_custom_wcs_no_file():
    step = ResampleStep()
    result = step._load_custom_wcs(None, (512, 512))
    assert result is None


@pytest.mark.parametrize(
    "output_shape, pixel_shape, bounding_box, expected",
    [
        (None, (100, 100), None, {"array_shape": (100, 100)}),
        (None, (50, 50), None, {"array_shape": (50, 50)}),
        (None, None, ((-0.5, 99.5), (-0.5, 99.5)), {"array_shape": (100, 100)}),
    ],
)
def test_load_custom_wcs_valid_pixel_shape(
    output_shape, expected, pixel_shape, bounding_box, request, tmp_path
):
    step = ResampleStep()
    asdf_wcs_file = request.getfixturevalue("asdf_wcs_file")(
        tmp_path, pixel_shape, bounding_box
    )
    result = step._load_custom_wcs(asdf_wcs_file, output_shape)
    assert result.pixel_shape == expected["pixel_shape"]


def test_load_custom_wcs_missing_output_shape(asdf_wcs_file):
    with pytest.raises(ValueError):
        resample_step._load_custom_wcs(asdf_wcs_file, None)


def test_load_custom_wcs_invalid_file(tmp_path):
    invalid_file = tmp_path / "invalid.asdf"
    with open(invalid_file, "w") as f:
        f.write("invalid asdf file")

    with pytest.raises(asdf.exceptions.AsdfFileNotFoundError):
        resample_step._load_custom_wcs(str(invalid_file), (512, 512))
