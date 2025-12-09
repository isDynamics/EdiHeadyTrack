import EdiHeadyTrack as EHT

participant = 'P003024'
trial = 'R1'
camera_name = 'S'

# participant = 'P003336'
# trial = 'L2'
# camera_name = 'F'

video = EHT.Video(f'../Data/Videos/{participant}_{trial}_{camera_name}.avi')

# Initialize a face detector from the DeepFace package, specify the model to use
# head_detector = EHT.HeadDetector('retinaface')

# Predict head bounding boxes from video, output as cropped images
# head_detections = head_detector.predict(video, output_dir='../Data/HeadDetection/retinaface')
head_detections = EHT.HeadDetector.load_detections(f'../Data/HeadDetection/GT/{participant}_{trial}_{camera_name}/')

# Initialize a head pose model, specify the method to use
head_pose_model = EHT.HeadPoseModel('6drepnet')

# Load camera extrinsics from a JSON file
camera = EHT.Camera(f'../Data/PoseEstimation/extrinsics/{participant}_{camera_name}.json')

# Run head pose estimation on the cropped faces, output the predictions as images with overlaid axes
pose_predictions_local, pose_predictions_global = head_pose_model.predict(
    head_detections,
    camera,
    data_output=f'../Data/PoseEstimation/out/6drepnet/data/{participant}_{trial}_{camera_name}'
)

kf = EHT.KalmanFilter()
pose_predictions_filtered = kf.filter(pose_predictions_local)

# Load head pose from sensor data
gt_pose = EHT.SensorDataLoader(f'../Data/PoseEstimation/GT/downsampled/{participant}_{trial}', offset_to_zero=False)

head_detections = EHT.HeadDetector.load_detections(f'../Data/HeadDetection/GT/{participant}_{trial}_{camera_name}/')
EHT.Plot.plot_head_orientation(
    poses=pose_predictions_local,
    head_detections=head_detections,
    output=f'../Data/PoseEstimation/out/6drepnet/viz/{participant}_{trial}_{camera_name}',
    show=True
)
EHT.Plot.plot_head_orientation(
    poses=pose_predictions_filtered,
    head_detections=head_detections,
    # camera=camera,
    output=f'../Data/PoseEstimation/out/6drepnet/viz/{participant}_{trial}_{camera_name}',
    show=True
)

# Plot time series of the roll, pitch and yaw angles as subplots, where the sensor data is ground truth
EHT.Plot.plot_time_series(
    output=f'../Data/PoseEstimation/out/6drepnet/plot/{participant}_{trial}_{camera_name}',
    show=True,
    # mocap = gt_pose.get_all_poses(),
    hpe = pose_predictions_local,
    filtered_hpe = pose_predictions_filtered
)