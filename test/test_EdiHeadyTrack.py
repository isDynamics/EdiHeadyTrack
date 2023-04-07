import unittest

import EdiHeadyTrack.calibrating    # for calibrating camera, extracting intrinsic parameters
import EdiHeadyTrack.logging        # for opening csv files to write data to
import EdiHeadyTrack.tracking       # for tracking specified video files
import EdiHeadyTrack.filtering      # for filtering measured signals with Butterworth filter
import EdiHeadyTrack.processing     # for calculating kinematics from measured displacements
import EdiHeadyTrack.comparing      # for comparing values with sensor data 
import EdiHeadyTrack.archiving      # for archiving outputs to ordered folder system


def test_calibrating():
    assert(EdiHeadyTrack.calibrating.calibrate('videos/calibration_xiaomi.mp4'))

def test_logging():
    ...

def test_tracking():
    ...

def test_filtering():
    ...

def test_processing():
    ...

def test_comparing():
    ...

def test_archiving():
    ...