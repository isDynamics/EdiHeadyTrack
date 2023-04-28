import EdiHeadyTrack as EHT

calibration_video = EHT.Video(filename='EdiHeadyTrack/resources/calibration_example.mp4')
# calibrated_camera = EHT.Camera().calibrate(checkerboard=(9,6), video=calibration_video)
# uncalibrated_camera = EHT.Camera()
# tracking_video = EHT.Video(filename='EdiHeadyTrack/resources/header1.mp4')
# sensor_data_1 = EHT.MediaPipe(video=tracking_video, camera=uncalibrated_camera)
# filter = EHT.Filter()
# .low_pass_butterworth(fs=4000, lowcut=160, order=4)
# head = EHT.Head(facedetector=sensor_data_1).apply_filter(filter)
# headplot = EHT.Plot().plot_head(head)
# headkinematics = EHT.HeadKinematics(head=head)