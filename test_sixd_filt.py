import EdiHeadyTrack as EHT
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from scipy.spatial.transform import Rotation as R
import json
import time

participants = ['P003024', 'P003272', 'P003336', 'P003489', 'P003540', 'P003682', 'P003726', 'P003730', 'P003940', 'P004298']
trials = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
camera_names = ['F', 'S']
# participants = ['P003272']
# trials = ['L6']
# camera_names = ['F']


participant_maes = defaultdict(list)
camera_maes = defaultdict(list)
trial_type_maes = defaultdict(list)
overall_maes = []

# List to collect rows before building the DataFrame
rows = []

# List to collect all JSONL entries for consolidated output
all_predictions = []

# Add a list to collect inference times
inference_times = []

for participant in participants:
    for trial in trials:
        for camera_name in camera_names:

            video_path = f'../../../Data/Videos/Processed/{participant}_{trial}_{camera_name}.avi'
            head_detections_path = f'../../../Data/HeadDetection/vggheads/successful/{participant}_{trial}_{camera_name}'
            camera_extrinsics_path = f'../../../Data/Extrinsics/Parsed/{participant}_{camera_name}.json'
            
            # Parse trial into type and repeat for JSONL lookup
            trial_type = trial[0]  # 'L' or 'R'
            trial_repeat = int(trial[1])  # 1, 2, 3, etc.

            if not (os.path.exists(video_path) and 
                    os.path.exists(head_detections_path) and 
                    os.path.exists(camera_extrinsics_path)):
                print(f"Skipping {participant}, {trial}, {camera_name} due to missing files.")
                continue
            
            # Load GT data from dataset.jsonl
            gt_entry = None
            with open('../../../Data/dataset.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if (data.get('Participant') == participant and 
                        data.get('Type') == trial_type and 
                        data.get('Repeat') == trial_repeat):
                        gt_entry = data
                        break
            
            if gt_entry is None:
                print(f"Skipping {participant}, {trial}, {camera_name}: No GT data found")
                continue
                
            # Extract and downsample GT pose data
            gt_euler = np.column_stack([gt_entry['roll'], gt_entry['pitch'], gt_entry['yaw']])
            gt_rmat = np.array(gt_entry['rotation_matrix'])  
            gt_rmat_ds = gt_rmat[::20]  
            gt_euler_ds = gt_euler[::20]

            # Initialize EdiHeadyTrack components
            video = EHT.Video(video_path)
            head_detections = EHT.HeadDetector.load_detections(head_detections_path)
            head_pose_model = EHT.HeadPoseModel('6drepnet360')
            camera = EHT.Camera(camera_extrinsics_path)

            # Calculate occlusion ratios
            occlusion_ratios = EHT.HeadDetector.calculate_occlusions(participant, trial, camera_name, '../../../Data/HeadDetection/GT/annotations.xml')

            # Calculate face visibility
            face_visibility = EHT.HeadDetector.calculate_visible_face_area(participant, trial, camera_name, '../../../Data/face_annotations.xml')

            # Run prediction
            start_time = time.time()
            rmat_local, rmat_global, euler_local, euler_global = head_pose_model.predict(
                head_detections,
                camera
            )
            inference_time = time.time() - start_time
            inference_times.append(inference_time)  # Collect inference time

            # Create a geodesic Gaussian filter
            filt = EHT.SO3Filter(sigma=1.5, sample_rate_hz=50.0, spike_deg=10)

            Sg = filt.smooth(rmat_global, input_type="matrix")
            Sl = filt.smooth(rmat_local,  input_type="matrix")

            # Use smoothed rotations from here on
            rmat_global = [m for m in Sg["R_smooth"]]
            rmat_local = [m for m in Sl["R_smooth"]]
            
            # Align global predictions to GT
            mask = None  # No mask applied for now

            rmat_global_aligned, A_sys = EHT.align_to_gt(
                np.stack(rmat_global),     # (N,3,3)
                gt_rmat_ds,                # (N,3,3)
                mask=mask,
                tol=1e-4,
                max_iter=100000,
                return_delta=True
            )

            # Use aligned predictions from here on
            rmat_global = [m for m in rmat_global_aligned]

            # Recompute Euler for plots AFTER alignment
            euler_global = (R.from_matrix(np.stack(rmat_global)).as_euler('xyz', degrees=False)).tolist()
            euler_local  = (R.from_matrix(np.stack(rmat_local)).as_euler('xyz', degrees=False)).tolist()

            # Collect pose data for consolidated JSONL file
            num_frames = len(euler_local)
            times = [i / 50.0 for i in range(num_frames)]  # 50 Hz sample rate

            # Extract roll, pitch, yaw arrays (smoothed)
            rolls   = [frame[0] for frame in euler_local]
            pitches = [frame[1] for frame in euler_local]
            yaws    = [frame[2] for frame in euler_local]

            # GT data is now loaded from dataset.jsonl above
            if len(rmat_global) != len(gt_rmat_ds):
                print(f"Skipping {participant}, {trial}, {camera_name} due to mismatched lengths.")
                continue
            

            # Compute GE to identity
            # create a time-series of 3x3 identity matrices matching rmat_global length
            identity = [np.eye(3) for _ in range(len(rmat_global))]
            
            ge_identity_pred = EHT.Analyser(rmat_global, identity).compute_geodesic_distances()
            ge_identity_gt = EHT.Analyser(gt_rmat_ds, identity).compute_geodesic_distances()

            # Compute relative rotations
            analyser_p = EHT.Analyser(rmat_global)
            analyser_gt = EHT.Analyser(gt_rmat_ds) 
            frame2frame_p, from_start_p = analyser_p.compute_relative_rotations(data_output=None)
            frame2frame_gt, from_start_gt = analyser_gt.compute_relative_rotations(data_output=None)
            
            # Compute geodesic distances between prediction and GT
            analyser_p_gt = EHT.Analyser(rmat_global, gt_rmat_ds)
            d = analyser_p_gt.compute_geodesic_distances()
            d_t0 = np.abs(np.array(from_start_p) - np.array(from_start_gt))
            d_deltat = np.abs(np.array(frame2frame_p) - np.array(frame2frame_gt))

            # Compute MAEs
            mae_d = np.mean(d)
            mae_d_t0 = np.mean(d_t0)
            mae_d_deltat = np.mean(d_deltat)

            # Store MAEs for aggregation
            mae_vector = np.array([mae_d, mae_d_t0, mae_d_deltat])
            participant_maes[participant].append(mae_vector)
            camera_maes[camera_name].append(mae_vector)
            trial_type_maes[trial[0]].append(mae_vector)
            overall_maes.append(mae_vector)

            # Plots
            # EHT.Plot.plot_euler_time_series(
            #     output=f'../../../Data/Plots/6drepnet360/filtered/eulers/{participant}_{trial}_{camera_name}',
            #     show=False,
            #     MoCap=gt_euler,
            #     SixDRepNet=euler_global
            # )

            # EHT.Plot.plot_geodesic_errors(
            #     to_identity=[ge_identity_gt, ge_identity_pred],
            #     framewise=[frame2frame_gt, frame2frame_p],
            #     from_start=[from_start_gt, from_start_p],
            #     output=f'../../../Data/Plots/6drepnet360/filtered/geodesics/{participant}_{trial}_{camera_name}',
            #     show=False
            # )

            # EHT.Plot.plot_head_orientation(
            #     poses=euler_local,
            #     head_detections=head_detections,
            #     output=f'../../../Data/Plots/6drepnet360/filtered/head_orientation/{participant}_{trial}_{camera_name}',
            #     show=False
            # )

            # Create JSONL entry for this trial
            jsonl_entry = {
                "Impact ID": f"{participant}_{trial}_{camera_name}",
                "Data source": "6DRepNet360",
                "Sample rate (Hz)": 50.0,
                "Participant": participant,
                "Camera": camera_name,
                "Type": trial[0],  # 'L' or 'R'
                "Repeat": trial[1],      # '1', '2', etc.
                "filtered": True,
                "times": list(np.ravel(times)),
                "roll": list(np.ravel(rolls)),
                "pitch": list(np.ravel(pitches)),
                "yaw": list(np.ravel(yaws)),
                "gt_roll": list(np.ravel(gt_euler_ds[:, 0])),
                "gt_pitch": list(np.ravel(gt_euler_ds[:, 1])),
                "gt_yaw": list(np.ravel(gt_euler_ds[:, 2])),
                "geodesic_error": list(np.ravel(d)),
                "geodesic_to_identity_prediction": list(np.ravel(ge_identity_pred)),
                "geodesic_to_identity_ground_truth": list(np.ravel(ge_identity_gt)),
                "from_start_prediction": list(np.ravel(from_start_p)),
                "from_start_ground_truth": list(np.ravel(from_start_gt)),
                "from_start_deviation": list(np.ravel(d_t0)),
                "frame_to_frame_prediction": list(np.ravel(frame2frame_p)),
                "frame_to_frame_ground_truth": list(np.ravel(frame2frame_gt)),
                "frame_to_frame_deviation": list(np.ravel(d_deltat)),
                "mae_from_start_deviation": mae_d_t0,
                "mae_frame_to_frame_deviation": mae_d_deltat,
                "occlusion_ratio": list(np.ravel(occlusion_ratios)) if isinstance(occlusion_ratios, np.ndarray) else occlusion_ratios,
                "face_visibility": list(np.ravel(face_visibility)) if isinstance(face_visibility, np.ndarray) else face_visibility,
                "omega_x": None,
                "omega_y": None,
                "omega_z": None,
                "quaternion": None,
                "rotation_matrix": None
            }

            with open('../../../Data/predictions.jsonl', 'a') as f:
                f.write(json.dumps(jsonl_entry) + '\n')

            
# ---------- Aggregated MAEs ----------
print("\n----- Aggregated Geodesic Distances (MAEs) -----")
for participant, maes in participant_maes.items():
    avg = np.mean(maes, axis=0)
    print(f"Participant {participant}: d={avg[0]:.4f}, d_t0={avg[1]:.4f}, d_deltat={avg[2]:.4f}")

for camera, maes in camera_maes.items():
    avg = np.mean(maes, axis=0)
    print(f"Camera {camera}: d={avg[0]:.4f}, d_t0={avg[1]:.4f}, d_deltat={avg[2]:.4f}")

for trial_type, maes in trial_type_maes.items():
    avg = np.mean(maes, axis=0)
    print(f"Trial type {trial_type}: d={avg[0]:.4f}, d_t0={avg[1]:.4f}, d_deltat={avg[2]:.4f}")

if overall_maes:
    overall_avg = np.mean(overall_maes, axis=0)
    print(f"\nOverall Mean Geodesic Distance: d={overall_avg[0]:.4f}, d_t0={overall_avg[1]:.4f}, d_deltat={overall_avg[2]:.4f}")


# ---------- Save aggregated MAE results in JSONL format ----------
aggregated_results = {
    "analysis_date": pd.Timestamp.now().isoformat(),
    "model_name": "6drepnet360",
    "data_format": "JSONL",
    "sample_rate": 50.0,
    "participants_analyzed": participants,
    "trials_analyzed": trials,
    "cameras_analyzed": camera_names,
    "participant_maes": {p: np.mean(maes, axis=0).tolist() for p, maes in participant_maes.items()},
    "camera_maes": {c: np.mean(maes, axis=0).tolist() for c, maes in camera_maes.items()},
    "trial_type_maes": {t: np.mean(maes, axis=0).tolist() for t, maes in trial_type_maes.items()},
    "overall_mae": np.mean(overall_maes, axis=0).tolist() if overall_maes else [0, 0, 0],
    "mae_metrics": ["geodesic_distance", "from_start_deviation", "frame_to_frame_deviation"],
    "total_trials_processed": len(overall_maes)
}

# Save aggregated results

with open('../../../Data/analysis_results.jsonl', 'a') as f:
    f.write(json.dumps(aggregated_results) + '\n')

print(f"\nAnalysis complete! Results saved to:")
print(f"  - Aggregated results: ../../../Data/analysis_results.jsonl")
print(f"Total trials processed: {len(overall_maes)}")

# Compute and print the mean inference time at the end
if inference_times:
    mean_inference_time = np.mean(inference_times)
    print(f"\nMean inference time: {mean_inference_time:.4f} seconds")