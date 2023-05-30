from EdiHeadyTrack.facedetector import FaceDetector, MediaPipe

TEST_FILE = 'resources/testvid.mp4'
from EdiHeadyTrack import Video
TEST_VIDEO = Video(TEST_FILE)
from EdiHeadyTrack import Camera
TEST_CAMERA = Camera()
SHOW = True

def test_FaceDetector():
    facedetector = FaceDetector(TEST_VIDEO, TEST_CAMERA, SHOW)
    assert type(facedetector.camera) == Camera
    assert type(facedetector.video) == Video
    assert facedetector.face2d
    assert facedetector.face3d == []

def test_MediaPipe():
    mediapipe = MediaPipe(TEST_VIDEO, TEST_CAMERA, SHOW)
    mediapipe.run()
    assert type(mediapipe) == MediaPipe
    