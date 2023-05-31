import EdiHeadyTrack as eht

# calibration_video = eht.Video(filename='test/resources/testvid.mp4')
# width, height = calibration_video.get_dim()
# print('Height: ', height)
# print('Width: ', width)
# camera = eht.Camera().calibrate(checkerboard=(9,6), video=calibration_video)
camera = eht.Camera()
tracking_video = eht.Video(filename='resources/header1.mp4')
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False)
filter = eht.Filter().low_pass_butterworth(fs=4000, lowcut=160, order=2)
# uhead = eht.Head(facedetector=mediapipe)
head = eht.Head(facedetector=mediapipe, id='MP').apply_filter(filter)
wax9 = eht.Wax9(filename='resources/example_imu.csv', time_offset=-59.335, id='WAX-9')
key_times = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
# key_times = [0.01, 0.1]
comparison_plot = eht.Plot(head,wax9).plot_property(property='velocity', 
                                               xlim=(0.85, 1.35), 
                                               ylim=(-220, 220), 
                                               key_times=key_times,
                                               show=True) 
comparison_plot.summarise()