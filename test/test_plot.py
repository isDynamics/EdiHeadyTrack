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
from EdiHeadyTrack import Wax9

def test_plot_property():
    key_times = [0.01, 0.02]
    line = Plot(HEAD).plot_property(show=False)
    line = Plot(HEAD).plot_property(property='velocity', 
                                               xlim=(0, 0.001), 
                                               ylim=(-220, 220), 
                                               key_times=key_times,
                                               show=False) 
    wax9 = Wax9('resources/example_imu.csv', time_offset=-59.335, id='WAX-9')
    line = Plot(HEAD, wax9).plot_property(show=False)
    line = Plot(HEAD).plot_property(property='pose',show=False)
    line = Plot(HEAD).plot_property(property='acceleration',show=False)
    line = Plot(wax9).plot_property(show=False)
    # x_plot, y_plot = line.line.get_xydata().T

    # import numpy as np
    # np.testing.assert_array_equal(x_plot, wax9.velocity['time'])
    # np.testing.assert_array_equal(y_plot, wax9.velocity['roll'])

def test_plot_summarise():
    wax9 = Wax9('resources/example_imu.csv', time_offset=-59.335, id='WAX-9')
    plot = Plot(HEAD, wax9).plot_property(show=False)

    plot.summarise()