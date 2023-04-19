import EdiHeadyTrack.calibrating    # for calibrating camera, extracting intrinsic parameters

TEST_FILE = 'videos/calibration_xiaomi.mp4'

def test_initialise():
    '''
    Assert correct values are returned for dummy checkerboard dimensions
    '''
    checkerboard = (2,2)
    criteria, threedpoints, twodpoints, objectp3d = EdiHeadyTrack.calibrating.initialise(checkerboard)
    assert criteria == (3, 30, 0.001)
    assert threedpoints == []
    assert twodpoints == []
    import numpy as np
    assert np.array_equal(objectp3d, [[[0., 0., 0.],[1., 0., 0.],[0., 1., 0.],[1., 1., 0.]]])


def test_vid():
    '''
    Assert frames of video file can be read, and that video fps is greater than 0
    '''
    assert EdiHeadyTrack.calibrating.open_vid(TEST_FILE)[0].read()[0] == True
    assert EdiHeadyTrack.calibrating.open_vid(TEST_FILE)[1] > 0


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