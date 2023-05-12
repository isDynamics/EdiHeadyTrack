import EdiHeadyTrack as EHT

calibration_video = EHT.Video(filename='EdiHeadyTrack/resources/calibration_example.mp4')
calibrated_camera = EHT.Camera().calibrate(checkerboard=(9,6), video=calibration_video)
# uncalibrated_camera = EHT.Camera()
tracking_video = EHT.Video(filename='EdiHeadyTrack/resources/header1.mp4')
mediapipe = EHT.MediaPipe(video=tracking_video, camera=calibrated_camera)
filter = EHT.Filter().low_pass_butterworth(fs=4000, lowcut=160, order=4)
# # # # # head = EHT.Head(facedetector=sensor_data_1, id='head')
head = EHT.Head(facedetector=mediapipe, id='MP').apply_filter(filter)

wax9 = EHT.Wax9(filename='EdiHeadyTrack/resources/example_imu.csv', time_offset=-59.335, id='WAX-9')
key_times = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
comparison_plot = EHT.Plot().plot_comparison('velocity', (0.85, 1.35), (-220, 220), key_times, head, wax9)