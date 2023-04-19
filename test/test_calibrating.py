import EdiHeadyTrack.calibrating    # for calibrating camera, extracting intrinsic parameters

def test_vid():
    '''
    Assert frames of video file can be read, and that video fps is greater than 0
    '''
    assert EdiHeadyTrack.calibrating.open_vid('videos/calibration_xiaomi.mp4')[0].read()[0] == True
    assert EdiHeadyTrack.calibrating.open_vid('videos/calibration_xiaomi.mp4')[1] > 0

def test_webcam():
    '''
    Assert frames from webcam can be read, and that video fps is greater than 0
    '''
    assert EdiHeadyTrack.calibrating.open_vid(0)[0].read()[0] == True
    assert EdiHeadyTrack.calibrating.open_vid(0)[1] > 0