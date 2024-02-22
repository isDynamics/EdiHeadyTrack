from EdiHeadyTrack.camera import Camera, Checkerboard, Calibrator

TEST_FILE = 'resources/calibration_example.mp4'
from EdiHeadyTrack.video import Video
TEST_VIDEO = Video(TEST_FILE)
SHOW = False
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
