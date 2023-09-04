import EdiHeadyTrack as eht

filter_wax9 = eht.Filter().low_pass_butterworth(fs=2000, lowcut=200, order=4)

# d0_2
camera = eht.Camera()
tracking_video = eht.Video(filename='phantom/1. Phantom videos/d0_2.mp4')
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False, refineLandmarks=False)
tddfa = eht.TDDFA_V2(video=tracking_video, camera=camera, show=True)
filter = eht.Filter().low_pass_butterworth(fs=24000, lowcut=200, order=4)
head_MP = eht.Head(posedetector=mediapipe, id='MP').apply_filter(filter)
head_TDDFA = eht.Head(posedetector=tddfa, id='TDDFA').apply_filter(filter)
wax9 = eht.Wax9(filename='phantom/3. Sensor data/d0_2.csv', time_offset=-7.82, id='WAX-9').apply_filter(filter_wax9)
key_times = [0.1,0.2,0.3,0.4,0.5,0.6]
# comparison_plot_velocity = eht.Plot(wax9).plot_property(property='velocity')
comparison_plot_velocity = eht.Plot(wax9, head_MP, head_TDDFA).plot_property(property='velocity', xlim=(0.1,0.6), key_times=key_times, ylim=(-250,250))
comparison_plot_velocity.summarise()

# d0_4
camera = eht.Camera()
tracking_video = eht.Video(filename='phantom/1. Phantom videos/d0_4.mp4')
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False, refineLandmarks=False)
tddfa = eht.TDDFA_V2(video=tracking_video, camera=camera, show=False)
filter = eht.Filter().low_pass_butterworth(fs=24000, lowcut=200, order=4)
head_MP = eht.Head(posedetector=mediapipe, id='MP').apply_filter(filter)
head_TDDFA = eht.Head(posedetector=tddfa, id='TDDFA').apply_filter(filter)
wax9 = eht.Wax9(filename='phantom/3. Sensor data/d0_4.csv', time_offset=-26.8, id='WAX-9').apply_filter(filter_wax9)
key_times = [0.3,0.4,0.5,0.6,0.7,0.8]
# comparison_plot_velocity = eht.Plot(wax9).plot_property(property='velocity')
comparison_plot_velocity = eht.Plot(wax9, head_MP, head_TDDFA).plot_property(property='velocity', key_times=key_times,xlim=(0.3,0.8), ylim=(-350,350))
comparison_plot_velocity.summarise()

# d0_5
camera = eht.Camera()
tracking_video = eht.Video(filename='phantom/1. Phantom videos/d0_5.mp4')
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False, refineLandmarks=False)
tddfa = eht.TDDFA_V2(video=tracking_video, camera=camera, show=False)
filter = eht.Filter().low_pass_butterworth(fs=24000, lowcut=200, order=4)
head_MP = eht.Head(posedetector=mediapipe, id='MP').apply_filter(filter)
head_TDDFA = eht.Head(posedetector=tddfa, id='TDDFA').apply_filter(filter)
# comparison_plot_pose = eht.Plot(head_MP, head_TDDFA).plot_property(property='pose')
wax9 = eht.Wax9(filename='phantom/3. Sensor data/d0_5.csv', time_offset=-15.44, id='WAX-9').apply_filter(filter_wax9)
key_times = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
# comparison_plot_velocity = eht.Plot(wax9).plot_property(property='velocity')
comparison_plot_velocity = eht.Plot(wax9, head_MP, head_TDDFA).plot_property(property='velocity', key_times=key_times,xlim=(0.1,0.7), ylim=(-250,250))
comparison_plot_velocity.summarise()

# d45_1
camera = eht.Camera()
tracking_video = eht.Video(filename='phantom/1. Phantom videos/d45_1.mp4')
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False, refineLandmarks=False)
tddfa = eht.TDDFA_V2(video=tracking_video, camera=camera, show=False)
filter = eht.Filter().low_pass_butterworth(fs=24000, lowcut=200, order=4)
head_MP = eht.Head(posedetector=mediapipe, id='MP').apply_filter(filter)
head_TDDFA = eht.Head(posedetector=tddfa, id='TDDFA').apply_filter(filter)
# comparison_plot_pose = eht.Plot(head_MP, head_TDDFA).plot_property(property='pose')
wax9 = eht.Wax9(filename='phantom/3. Sensor data/d45_1.csv', time_offset=-18.23, id='WAX-9').apply_filter(filter_wax9)
key_times = [0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
comparison_plot_velocity = eht.Plot(wax9, head_MP, head_TDDFA).plot_property(property='velocity', xlim=(0,1), ylim=(-200,200), key_times=key_times)
comparison_plot_velocity.summarise()

# d45_3
camera = eht.Camera()
tracking_video = eht.Video(filename='phantom/1. Phantom videos/d45_3.mp4')
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False, refineLandmarks=False)
tddfa = eht.TDDFA_V2(video=tracking_video, camera=camera, show=False)
filter = eht.Filter().low_pass_butterworth(fs=24000, lowcut=200, order=4)
head_MP = eht.Head(posedetector=mediapipe, id='MP').apply_filter(filter)
head_TDDFA = eht.Head(posedetector=tddfa, id='TDDFA').apply_filter(filter)
# comparison_plot_pose = eht.Plot(head_MP, head_TDDFA).plot_property(property='pose')
wax9 = eht.Wax9(filename='phantom/3. Sensor data/d45_3.csv', time_offset=-16.67, id='WAX-9').apply_filter(filter_wax9)
key_times = [0.1,0.2,0.3,0.4,0.5,0.6]
comparison_plot_velocity = eht.Plot(wax9, head_MP, head_TDDFA).plot_property(property='velocity', key_times=key_times,xlim=(0.1,0.6), ylim=(-300,300))
comparison_plot_velocity.summarise()

# d45_5
camera = eht.Camera()
tracking_video = eht.Video(filename='phantom/1. Phantom videos/d45_5.mp4')
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False, refineLandmarks=False)
tddfa = eht.TDDFA_V2(video=tracking_video, camera=camera, show=False)
filter = eht.Filter().low_pass_butterworth(fs=24000, lowcut=200, order=4)
head_MP = eht.Head(posedetector=mediapipe, id='MP').apply_filter(filter)
head_TDDFA = eht.Head(posedetector=tddfa, id='TDDFA').apply_filter(filter)
# comparison_plot_pose = eht.Plot(head_MP, head_TDDFA).plot_property(property='pose')
wax9 = eht.Wax9(filename='phantom/3. Sensor data/d45_5.csv', time_offset=-15.33, id='WAX-9').apply_filter(filter_wax9)
key_times = [0.5,0.6,0.7,0.8,0.9,1.0]
comparison_plot_velocity = eht.Plot(wax9, head_MP, head_TDDFA).plot_property(property='velocity', key_times=key_times,xlim=(0.5,1), ylim=(-250,250))
comparison_plot_velocity.summarise()
