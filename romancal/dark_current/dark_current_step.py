#! /usr/bin/env python

import numpy as np
from astropy import units as u
from roman_datamodels import datamodels as rdd
from roman_datamodels import maker_utils
from stcal.dark_current import dark_sub

from romancal.stpipe import RomanStep

__all__ = ["DarkCurrentStep"]


class DarkCurrentStep(RomanStep):
    """
    DarkCurrentStep: Performs dark current correction by subtracting
    dark current reference data from the input science data model.
    """

    spec = """
        dark_output = output_file(default = None) # Dark corrected model
    """

    reference_file_types = ["dark"]

    def process(self, input):
        # Open the input data model
        with rdd.open(input) as input_model:
            # Get the name of the dark reference file to use
            self.dark_name = self.get_reference_file(input_model, "dark")
            self.log.info("Using DARK reference file %s", self.dark_name)

            # Open dark model
            dark_model = rdd.open(self.dark_name)

            # Temporary patch to utilize stcal dark step until MA table support
            # is fully implemented
            if "ngroups" not in dark_model.meta.exposure:
                dark_model.meta.exposure["ngroups"] = dark_model.data.shape[0]
            if "nframes" not in dark_model.meta.exposure:
                dark_model.meta.exposure["nframes"] = input_model.meta.exposure.nframes
            if "groupgap" not in dark_model.meta.exposure:
                dark_model.meta.exposure[
                    "groupgap"
                ] = input_model.meta.exposure.groupgap

            # Do the dark correction

            # out_data, dark_data = dark_sub.do_correction(
            #     input_model, dark_model, self.dark_output
            # )
            # this is somewhat committed to doing a deep copy.
            out_data = input_model
            out_data.data -= dark_model.data
            out_data.pixeldq |= dark_model.dq
            out_data.meta.cal_step.dark = 'COMPLETE'

            # Save dark data to file
            if self.dark_output is not None:
                dark_model.save(self.dark_output)  # not clear to me that this makes any sense
            dark_model.close()

        if self.save_results:
            try:
                self.suffix = "darkcurrent"
            except AttributeError:
                self["suffix"] = "darkcurrent"
            dark_model.close()

        return out_data
