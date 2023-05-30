from EdiHeadyTrack.sensordata import SensorData, Head, IMU

def test_SensorData():
    sensor = SensorData()
    assert sensor.velocity
    assert sensor.acceleration

TEST_FILE = 'test/resources/testvidshort.mp4'
from EdiHeadyTrack import Video
TEST_VIDEO = Video(TEST_FILE)
from EdiHeadyTrack import Camera
TEST_CAMERA = Camera()
SHOW = False
from EdiHeadyTrack.facedetector import MediaPipe
MEDIAPIPE = MediaPipe(TEST_VIDEO, TEST_CAMERA, SHOW)


def test_Head_counter():
    head = Head(MEDIAPIPE)
    assert head.id == 1
    head = Head(MEDIAPIPE)
    assert head.id == 2
    head = Head(MEDIAPIPE, id='head')
    assert head.id == 'head'


def test_Head_calculate_pose():
    sensor = SensorData()
    head = Head(MEDIAPIPE)

    

def test_Head_calculate_kinematics():
    ...

def test_Head_apply_filter():
    ...

def test_IMU():
    ...