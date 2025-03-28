"""Regression tests for the Reference Pixel Correction step"""

import pytest

from romancal.stpipe import RomanStep

from .regtestdata import compare_asdf


@pytest.mark.bigdata
def test_refpix_step(rtdata, ignore_asdf_paths, resource_tracker, request):
    # I have no idea what this is supposed to be
    input_datafile = "r0000101001001001001_0001_wfi01_saturation.asdf"
    rtdata.get_data(f"WFI/image/{input_datafile}")
    rtdata.input = input_datafile

    args = ["romancal.step.RefPixStep", rtdata.input]
    with resource_tracker.track(log=request):
        RomanStep.from_cmdline(args)

    # Again I have no idea here
    output = "r0000101001001001001_0001_wfi01_refpix.asdf"
    rtdata.output = output
    rtdata.get_truth(f"truth/WFI/image/{output}")

    diff = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
    assert diff.identical, diff.report()
