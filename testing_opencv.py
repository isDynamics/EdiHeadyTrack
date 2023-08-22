import EdiHeadyTrack as eht
# Camera and video
camera = eht.Camera()
tracking_video = eht.Video(filename='resources/deliberate_motion.mp4')
# # Facedetectors
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False, refineLandmarks=False)
tddfa = eht.TDDFA_V2(video=tracking_video, camera=camera, show=False)
# # Create filter
filter = eht.Filter().low_pass_butterworth(fs=4000, lowcut=160, order=4)
# # Head objects
head_MP = eht.Head(posedetector=mediapipe, id='MP').apply_filter(filter)
head_TDDFA = eht.Head(posedetector=tddfa, id='TDDFA').apply_filter(filter)
# IMU device
wax9 = eht.Wax9(filename='resources/240_del_sensor.csv', time_offset=-18.652, id='WAX-9')
# Plotting
key_times = [4.344, 5.339, 6.333, 7.495, 8.681, 9.873, 10.953, 12.252]
# comparison_plot = eht.Plot(wax9, head_MP, head_TDDFA).plot_property(property='velocity', 
#                                                      xlim=(0, 13), 
#                                                      ylim=(-220, 220), 
#                                                      key_times=[],
#                                                      show=True) 
comparison_plot = eht.Plot(wax9, head_MP, head_TDDFA).plot_property(property='velocity', 
                                                        xlim=(3,13),
                                                        key_times=key_times,
                                                        show=True) 
comparison_plot.summarise()