import EdiHeadyTrack as EHT

participant = 'P003024'
trial = 'L1'
camera_names = ['S', 'F']

videos = [EHT.Video(f'../Data/Videos/{participant}_{trial}_{camera}.avi') for camera in camera_names]

# Initialize a face detector from the DeepFace package, specify the model to use
head_detector = EHT.HeadDetector('retinaface')

# Predict head bounding boxes from video, output as cropped images
head_detections = [head_detector.predict(video, output_dir='../Data/HeadDetection/retinaface') for video in videos]

# Initialize a head pose model, specify the method to use
head_pose_model = EHT.HeadPoseModel('6drepnet')

# Load camera extrinsics from a JSON file
cameras = [EHT.Camera(f'../Data/PoseEstimation/extrinsics/{participant}_{camera}.json') for camera in camera_names]

# Run head pose estimation on the cropped faces, output the predictions as images with overlaid axes
pose_predictions = [head_pose_model.predict(head_detections[i], cameras[i], 
                                            viz_output=f'../Data/PoseEstimation/out/6drepnet/viz/{participant}_{trial}_{camera}', 
                                            data_output=f'../Data/PoseEstimation/out/6drepnet/data/{participant}_{trial}_{camera}') for i, camera in enumerate(camera_names)]

# Load head pose from sensor data
gt_pose = EHT.SensorDataLoader('../Data/PoseEstimation/GT/downsampled/P003024_L2/')

# # Plot time series of the roll, pitch and yaw angles as subplots, where the sensor data is ground truth
EHT.Plot.plot_time_series(gt_pose, pose_predictions[0], output= '../Data/PoseEstimation/out/plot/{participant}_{trial}_{camera_names[0]}')
EHT.Plot.plot_time_series(gt_pose, pose_predictions[1], output= '../Data/PoseEstimation/out/plot/{participant}_{trial}_{camera_names[1]}')