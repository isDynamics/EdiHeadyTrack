from EdiHeadyTrack.posedetector import PoseDetector, MediaPipe

TEST_FILE = 'test/resources/testvidshort.mp4'
from EdiHeadyTrack import Video
TEST_VIDEO = Video(TEST_FILE)
from EdiHeadyTrack import Camera
TEST_CAMERA = Camera()
SHOW = False

def test_PoseDetector():
    posedetector = PoseDetector(TEST_VIDEO, TEST_CAMERA, SHOW)
    assert type(posedetector.camera) == Camera
    assert type(posedetector.video) == Video

def test_MediaPipe():
    mediapipe = MediaPipe(TEST_VIDEO, TEST_CAMERA, SHOW)
    assert mediapipe.face2d['key landmark positions'][0][0] == [726, 252]
    assert round(mediapipe.pose['yaw'][0], 2) == -8.52

    
    