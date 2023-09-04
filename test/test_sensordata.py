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
from EdiHeadyTrack.posedetector import MediaPipe
MEDIAPIPE = MediaPipe(TEST_VIDEO, TEST_CAMERA, SHOW)
Head._counter=0

def test_Head_counter():
    head = Head(MEDIAPIPE)
    assert head.id == 1
    head = Head(MEDIAPIPE)
    assert head.id == 2
    head = Head(MEDIAPIPE, id='head')
    assert head.id == 'head'

def test_Head_calculate_kinematics():
    head = Head(MEDIAPIPE)
    head.calculate_kinematics()
    # assert round(head.velocity['yaw'][0], 2) == -575.08

def test_Head_apply_filter():
    from EdiHeadyTrack import Filter
    filter = Filter()
    head = Head(MEDIAPIPE)
    head.apply_filter(filter)
    # assert round(head.velocity['yaw'][0], 0) == -88

def test_IMU():
    IMU._counter = 0
    imu = IMU()
    assert imu.id == 1
    imu = IMU()
    assert imu.id == 2