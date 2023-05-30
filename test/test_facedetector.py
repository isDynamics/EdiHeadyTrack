from EdiHeadyTrack.facedetector import FaceDetector, MediaPipe

TEST_FILE = 'test/resources/testvidshort.mp4'
from EdiHeadyTrack import Video
TEST_VIDEO = Video(TEST_FILE)
from EdiHeadyTrack import Camera
TEST_CAMERA = Camera()
SHOW = False

def test_FaceDetector():
    facedetector = FaceDetector(TEST_VIDEO, TEST_CAMERA, SHOW)
    assert type(facedetector.camera) == Camera
    assert type(facedetector.video) == Video
    assert facedetector.face2d
    assert facedetector.face3d == []

def test_MediaPipe():
    mediapipe = MediaPipe(TEST_VIDEO, TEST_CAMERA, SHOW)
    assert mediapipe.face2d['key landmark positions'][0][0] == [726, 252]