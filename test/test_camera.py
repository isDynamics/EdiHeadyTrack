from EdiHeadyTrack.camera import Camera, Checkerboard, Calibrator

TEST_FILE = 'resources/calibration_example.mp4'
from EdiHeadyTrack.video import Video
TEST_VIDEO = Video(TEST_FILE)
SHOW = True
CHECKERBOARD = (9,6)

def test_Camera_calibrate():
    camera = Camera().calibrate(CHECKERBOARD, TEST_VIDEO, SHOW)
    assert type(camera.video) == Video
    assert type(camera.calibrator) == Calibrator
    assert camera.calibrated

    import os
    camera.calibrator.save_outputs()
    assert os.path.exists('camera_matrix.csv')
    assert os.path.exists('camera_distortion.csv')

# def test_initialise():
#     '''
#     Assert correct values are returned for dummy checkerboard dimensions
#     '''
#     checkerboard = (2,2)
#     criteria, threedpoints, twodpoints, objectp3d = EdiHeadyTrack.calibrating.initialise(checkerboard)
#     assert criteria == (3, 30, 0.001)
#     assert threedpoints == []
#     assert twodpoints == []
#     import numpy as np
#     assert np.array_equal(objectp3d, [[[0., 0., 0.],[1., 0., 0.],[0., 1., 0.],[1., 1., 0.]]])


# def test_vid():
#     '''
#     Assert frames of video file can be read, and that video fps is greater than 0
#     '''
#     assert EdiHeadyTrack.calibrating.open_vid(TEST_FILE)[0].read()[0] == True
#     assert EdiHeadyTrack.calibrating.open_vid(TEST_FILE)[1] > 0


# def test_calibrate():
#     '''
#     Assert that calibrate function runs without error
#     '''
#     assert EdiHeadyTrack.calibrating.calibrate(TEST_FILE) == None

# def test_webcam():
#     '''
#     Assert frames from webcam can be read, and that video fps is greater than 0
#     '''
#     assert EdiHeadyTrack.calibrating.open_vid(0)[0].read()[0] == True
#     assert EdiHeadyTrack.calibrating.open_vid(0)[1] > 0