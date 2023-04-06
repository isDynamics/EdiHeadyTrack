import unittest

import EdiHeadyTrack.calibrating    # for calibrating camera, extracting intrinsic parameters
import EdiHeadyTrack.logging        # for opening csv files to write data to
import EdiHeadyTrack.tracking       # for tracking specified video files
import EdiHeadyTrack.filtering      # for filtering measured signals with Butterworth filter
import EdiHeadyTrack.processing     # for calculating kinematics from measured displacements
import EdiHeadyTrack.comparing      # for comparing values with sensor data 
import EdiHeadyTrack.archiving      # for archiving outputs to ordered folder system
