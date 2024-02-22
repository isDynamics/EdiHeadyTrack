import EdiHeadyTrack as eht

camera = eht.Camera()
tracking_video = eht.Video(filename='resources/tom_curry_red_card.mov')
mediapipe = eht.MediaPipe(video=tracking_video, camera=camera, show=False, refineLandmarks=False)
tddfa = eht.TDDFA_V2(video=tracking_video, camera=camera, show=False)
head_TDDFA = eht.Head(posedetector=tddfa, id='TDDFA')
head_MP = eht.Head(posedetector=mediapipe, id='MP')
key_times = [0.1, 0.5, 1.0, 1.4]
comparison_plot = eht.Plot(head_MP, head_TDDFA).plot_property(property='pose', 
                                                    key_times=key_times,
                                                    show=True) 