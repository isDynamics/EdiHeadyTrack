from EdiHeadyTrack.plot import Plot

TEST_FILE = 'test/resources/testvidshort.mp4'
from EdiHeadyTrack import Video
TEST_VIDEO = Video(TEST_FILE)
from EdiHeadyTrack import Camera
TEST_CAMERA = Camera()
SHOW = False
from EdiHeadyTrack.facedetector import MediaPipe
MEDIAPIPE = MediaPipe(TEST_VIDEO, TEST_CAMERA, SHOW)
from EdiHeadyTrack import Filter
FILTER = Filter().low_pass_butterworth(fs=4000, lowcut=160, order=4)
from EdiHeadyTrack import Head
HEAD = Head(MEDIAPIPE).apply_filter(FILTER)

def test_plot_property():
    plot = Plot(HEAD).plot_property(show=False)
    # x_plot, y_plot = line.get_xydata().T