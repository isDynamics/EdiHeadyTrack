from EdiHeadyTrack.plot import Plot

TEST_FILE = 'test/resources/testvidshort.mp4'
from EdiHeadyTrack import Video
TEST_VIDEO = Video(TEST_FILE)
from EdiHeadyTrack import Camera
TEST_CAMERA = Camera()
SHOW = False
from EdiHeadyTrack.posedetector import MediaPipe
MEDIAPIPE = MediaPipe(TEST_VIDEO, TEST_CAMERA, SHOW)
from EdiHeadyTrack import Filter
FILTER = Filter().low_pass_butterworth(fs=4000, lowcut=160, order=4)
from EdiHeadyTrack import Head
HEAD = Head(MEDIAPIPE).apply_filter(FILTER)
from EdiHeadyTrack import Wax9

import matplotlib.pyplot as plt
plt.clf()

def test_plot_property():
    key_times = [0.01, 0.02]
    # print(HEAD.pose)
    plt.clf()
    line = Plot(HEAD).plot_property(xlim=(0, 0.1), 
                                    ylim=(-220, 220),
                                    show=False)
    plt.clf()
    line = Plot(HEAD).plot_property(property='velocity', 
                                    xlim=(0, 0.1), 
                                    ylim=(-220, 220), 
                                    key_times=key_times,
                                    show=False) 
    plt.clf()
    wax9 = Wax9('resources/example_imu.csv', time_offset=-59.335, id='WAX-9')
    line = Plot(HEAD, wax9).plot_property(xlim=(0, 0.1), 
                                          ylim=(-220, 220), 
                                          show=False)
    plt.clf()
    line = Plot(HEAD).plot_property(property='pose',
                                    xlim=(0, 0.1), 
                                    ylim=(-220, 220),
                                    show=False)
    plt.clf()
    line = Plot(HEAD).plot_property(property='acceleration',
                                    xlim=(0, 0.1), 
                                    ylim=(-220, 220),
                                    show=False)
    plt.clf()
    line = Plot(wax9).plot_property(xlim=(0, 0.1), 
                                    ylim=(-220, 220),
                                    show=False)
    plt.clf()

def test_plot_summarise():
    plt.clf()
    wax9 = Wax9('resources/example_imu.csv', time_offset=-59.335, id='WAX-9')
    plot = Plot(HEAD, wax9).plot_property(xlim=(0, 0.1),  
                                          show=False)

    plot.summarise()
