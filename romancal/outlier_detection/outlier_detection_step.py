"""Public common step definition for OutlierDetection processing."""

from functools import partial
from pathlib import Path

from romancal.datamodels import ModelLibrary
from romancal.outlier_detection import outlier_detection

from ..stpipe import RomanStep

__all__ = ["OutlierDetectionStep"]


class OutlierDetectionStep(RomanStep):
    """Flag outlier bad pixels and cosmic rays in DQ array of each input image.

    Input images can be listed in an input association file or already wrapped
    with a ModelLibrary. DQ arrays are modified in place.

    Parameters
    -----------
    input_data : `~romancal.datamodels.container.ModelLibrary`
        A `~romancal.datamodels.container.ModelLibrary` object.

    """

    class_alias = "outlier_detection"

    # The members of spec needs to be a super-set of all parameters needed
    # by the various versions of the outlier_detection algorithms, and each
    # version will pick and choose what they need while ignoring the rest.
    spec = """
        weight_type = option('ivm','exptime',default='ivm') # Weighting type to use to create the median image
        pixfrac = float(default=1.0) # Fraction by which input pixels are shrunk before being drizzled onto the output image grid
        kernel = string(default='square') # Shape of the kernel used for flux distribution onto output images
        fillval = string(default='INDEF') # Value assigned to output pixels that have zero weight or no flux during drizzling
        maskpt = float(default=0.7) # Percentage of weight image values below which they are flagged as bad pixels
        snr = string(default='5.0 4.0') # The signal-to-noise values to use for bad pixel identification
        scale = string(default='1.2 0.7') # The scaling factor applied to derivative used to identify bad pixels
        backg = float(default=0.0) # User-specified background value to subtract during final identification step
        kernel_size = string(default='7 7') # Size of kernel to be used during resampling of the data
        save_intermediate_results = boolean(default=False) # Specifies whether or not to write out intermediate products to disk
        resample_data = boolean(default=True) # Specifies whether or not to resample the input images when performing outlier detection
        good_bits = string(default="~DO_NOT_USE+NON_SCIENCE")  # DQ bit value to be considered 'good'
        in_memory = boolean(default=False) # Specifies whether or not to keep all intermediate products and datamodels in memory
    """  # noqa: E501

    def process(self, input_models):
        """Perform outlier detection processing on input data."""

        if isinstance(input_models, ModelLibrary):
            library = input_models
        else:
            library = ModelLibrary(input_models)

        # check number of input models
        if len(library) < 2:
            # if input can be parsed into a ModelLibrary
            # but is not valid then log a warning message and
            # skip outlier detection step
            self.log.warning(
                "Skipping outlier_detection - at least two imaging observations are needed."
            )

            def set_skip(model, index):
                model.meta.cal_step["outlier_detection"] = "SKIPPED"

            list(library.map_function(set_skip))

            return library

        # check that all inputs are WFI_IMAGE
        def get_exptype(model, index):
            return model.meta.exposure.type

        exptypes = list(library.map_function(get_exptype, modify=False))
        if any(exptype != "WFI_IMAGE" for exptype in exptypes):
            raise ValueError(
                f"outlier_detection only supports WFI_IMAGE exposure types: {set(exptypes)}"
            )

        # Setup output path naming if associations are involved.
        asn_id = library.asn.get("asn_id", None)
        if asn_id is not None:
            _make_output_path = self.search_attr("_make_output_path", parent_first=True)
            self._make_output_path = partial(_make_output_path, asn_id=asn_id)

        detection_step = outlier_detection.OutlierDetection
        pars = {
            "weight_type": self.weight_type,
            "pixfrac": self.pixfrac,
            "kernel": self.kernel,
            "fillval": self.fillval,
            "maskpt": self.maskpt,
            "snr": self.snr,
            "scale": self.scale,
            "backg": self.backg,
            "kernel_size": self.kernel_size,
            "save_intermediate_results": self.save_intermediate_results,
            "resample_data": self.resample_data,
            "good_bits": self.good_bits,
            "in_memory": self.in_memory,
            "make_output_path": self.make_output_path,
            "resample_suffix": "i2d",
        }

        self.log.debug(f"Using {detection_step.__name__} class for outlier_detection")

        # Set up outlier detection, then do detection
        step = detection_step(library, **pars)
        step.do_detection()

        state = "COMPLETE"

        if not self.save_intermediate_results:
            self.log.debug(
                "The following files will be deleted since \
                save_intermediate_results=False:"
            )
        with library:
            for i, model in enumerate(library):
                model.meta.cal_step["outlier_detection"] = state
                if not self.save_intermediate_results:
                    #  remove intermediate files found in
                    #  make_output_path() and the local dir
                    intermediate_files_paths = [
                        Path(self.make_output_path()).parent,
                        Path().cwd(),
                    ]
                    intermediate_files_suffixes = (
                        "*blot.asdf",
                        "*median.asdf",
                        f'*outlier_{pars.get("resample_suffix")}.asdf',
                    )
                    for current_path in intermediate_files_paths:
                        for suffix in intermediate_files_suffixes:
                            for filename in current_path.glob(suffix):
                                filename.unlink()
                                self.log.debug(f"    {filename}")
                library.shelve(model, i)
        return library
