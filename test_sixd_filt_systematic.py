import EdiHeadyTrack as EHT
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import csv

# ----------------------- experiment controls -----------------------

# Filters to run: (human_readable_label, filter_instance_or_None, sigma_value_or_None)
def fmt_sigma(s):
    if s is None:
        return "none"
    # make filesystem-friendly
    return f"{s}".replace(".", "p")

filters_to_test = [
    ("None", None, None),
    *[(f"Gaussian {s:.2f}",
       EHT.SO3Filter(sigma=s, spike_deg=10.0, sample_rate_hz=50),
       s)
      for s in np.arange(0.25, 5.25, 0.25)]
]

    
participants = ['P003024', 'P003272', 'P003336', 'P003489', 'P003540', 'P003682', 'P003726', 'P003730', 'P003940', 'P004298']
trials = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
camera_names = ['F', 'S']

SAMPLE_RATE = 50.0

# base output folders
PLOT_BASE = Path("../../../Data/Plots/6drepnet360/filtered_gaussian")
JSONL_PRED_PATH = Path("../../../Data/predictions_gaussian.jsonl")
JSONL_ANALYSIS_PATH = Path("../../../Data/analysis_results_gaussian.jsonl")

PLOT_BASE.mkdir(parents=True, exist_ok=True)

# Per-filter aggregations
results_by_filter = defaultdict(list)

# ----------------------- processing -----------------------

for participant in participants:
    for trial in trials:
        for camera_name in camera_names:

            video_path = f'../../../Data/Videos/Processed/{participant}_{trial}_{camera_name}.avi'
            head_detections_path = f'../../../Data/HeadDetection/vggheads/successful/{participant}_{trial}_{camera_name}'
            camera_extrinsics_path = f'../../../Data/Extrinsics/Parsed/{participant}_{camera_name}.json'

            # Parse trial into type and repeat for JSONL lookup
            trial_type = trial[0]      # 'L' or 'R'
            trial_repeat = int(trial[1])

            if not (os.path.exists(video_path) and 
                    os.path.exists(head_detections_path) and 
                    os.path.exists(camera_extrinsics_path)):
                print(f"Skipping {participant}, {trial}, {camera_name} due to missing files.")
                continue

            # Load GT entry
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

            # Extract and downsample GT pose data (1000 Hz -> 50 Hz)
            gt_euler = np.column_stack([gt_entry['roll'], gt_entry['pitch'], gt_entry['yaw']])
            gt_rmat = np.array(gt_entry['rotation_matrix'])
            gt_rmat_ds = gt_rmat[::20]
            gt_euler_ds = gt_euler[::20]

            # Initialize EdiHeadyTrack components
            video = EHT.Video(video_path)
            head_detections = EHT.HeadDetector.load_detections(head_detections_path)
            head_pose_model = EHT.HeadPoseModel('6drepnet360')
            camera = EHT.Camera(camera_extrinsics_path)

            # Aux signals (optional but you had them in your pipeline)
            occlusion_ratios = EHT.HeadDetector.calculate_occlusions(
                participant, trial, camera_name, '../../../Data/HeadDetection/GT/annotations.xml'
            )
            face_visibility = EHT.HeadDetector.calculate_visible_face_area(
                participant, trial, camera_name, '../../../Data/face_annotations.xml'
            )

            # Run prediction (raw)
            rmat_local_raw, rmat_global_raw, euler_local_raw, euler_global_raw = head_pose_model.predict(
                head_detections,
                camera
            )

            # Ensure lengths match GT (post-downsample)
            if len(rmat_global_raw) != len(gt_rmat_ds):
                print(f"Skipping {participant}, {trial}, {camera_name} due to mismatched lengths.")
                continue

            # ------------ run all filters ------------
            for filter_label, filt, sigma_val in filters_to_test:

                # Make output folders per filter
                sigma_tag = fmt_sigma(sigma_val)
                plot_dir_eulers = PLOT_BASE / f"eulers_sigma_{sigma_tag}"
                plot_dir_geos   = PLOT_BASE / f"geodesics_sigma_{sigma_tag}"
                plot_dir_eulers.mkdir(parents=True, exist_ok=True)
                plot_dir_geos.mkdir(parents=True, exist_ok=True)

                # Apply filter (or pass-through)
                if filt is None:
                    rmat_global = rmat_global_raw
                    rmat_local  = rmat_local_raw
                    filtered_flag = False
                else:
                    Sg = filt.smooth(rmat_global_raw, input_type="matrix")
                    Sl = filt.smooth(rmat_local_raw,  input_type="matrix")
                    rmat_global = [m for m in Sg["R_smooth"]]
                    rmat_local  = [m for m in Sl["R_smooth"]]
                    filtered_flag = True

                # Align global predictions to GT (systematic misalignment)
                mask = None
                rmat_global_aligned, A_sys = EHT.align_to_gt(
                    np.stack(rmat_global),   # (N,3,3)
                    gt_rmat_ds,              # (N,3,3)
                    mask=mask,
                    tol=1e-4,
                    max_iter=100000,
                    return_delta=True
                )
                rmat_global = [m for m in rmat_global_aligned]

                # Recompute Euler for plots AFTER alignment
                euler_global = R.from_matrix(np.stack(rmat_global)).as_euler('xyz', degrees=False).tolist()
                euler_local  = R.from_matrix(np.stack(rmat_local)).as_euler('xyz', degrees=False).tolist()

                # Timebase
                num_frames = len(euler_local)
                times = [i / SAMPLE_RATE for i in range(num_frames)]

                # Extract roll, pitch, yaw (smoothed or raw)
                rolls   = [frame[0] for frame in euler_local]
                pitches = [frame[1] for frame in euler_local]
                yaws    = [frame[2] for frame in euler_local]

                # ------------ metrics ------------
                # GE to identity
                identity = [np.eye(3) for _ in range(len(rmat_global))]
                ge_identity_pred = EHT.Analyser(rmat_global, identity).compute_geodesic_distances()
                ge_identity_gt   = EHT.Analyser(gt_rmat_ds, identity).compute_geodesic_distances()

                # Relative rotations (frame-to-frame & from-start)
                analyser_p  = EHT.Analyser(rmat_global)
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

                # # Store MAEs for aggregation
                # mae_vector = np.array([mae_d, mae_d_t0, mae_d_deltat])
                # participant_maes[participant].append(mae_vector)
                # camera_maes[camera_name].append(mae_vector)
                # trial_type_maes[trial[0]].append(mae_vector)
                # overall_maes.append(mae_vector)

                # results_by_filter[filter_label].append(np.array([mae_d, mae_d_t0, mae_d_deltat]))
                # Store full per-frame metrics (not per-video means)
                results_by_filter[filter_label].append({
                    "d": np.ravel(d),                  # geodesic distances
                    "d_t0": np.ravel(d_t0),            # deviation from start
                    "d_deltat": np.ravel(d_deltat),    # frame-to-frame deviation
                })

                # ------------ plots (optional; requires your Plot module) ------------
                # If you want to re-enable plotting with per-sigma folders, uncomment these:
                # EHT.Plot.plot_euler_time_series(
                #     output=str(plot_dir_eulers / f'{participant}_{trial}_{camera_name}'),
                #     show=False,
                #     MoCap=gt_euler_ds,
                #     SixDRepNet=euler_global
                # )
                # EHT.Plot.plot_geodesic_errors(
                #     to_identity=[ge_identity_gt, ge_identity_pred],
                #     framewise=[frame2frame_gt, frame2frame_p],
                #     from_start=[from_start_gt, from_start_p],
                #     output=str(plot_dir_geos / f'{participant}_{trial}_{camera_name}'),
                #     show=False
                # )

                # ------------ JSONL: per-trial predictions ------------
                jsonl_entry = {
                    "Impact ID": f"{participant}_{trial}_{camera_name}",
                    "Data source": "6DRepNet360",
                    "Sample rate (Hz)": SAMPLE_RATE,
                    "Participant": participant,
                    "Camera": camera_name,
                    "Type": trial_type,       # 'L' or 'R'
                    "Repeat": trial_repeat,   # 1..6
                    "filtered": filtered_flag,
                    "filter": {
                        "type": "gaussian" if sigma_val is not None else "none",
                        "sigma": sigma_val
                    },
                    "times": list(np.ravel(times)),
                    "roll": list(np.ravel(rolls)),
                    "pitch": list(np.ravel(pitches)),
                    "yaw": list(np.ravel(yaws)),
                    "gt_roll": list(np.ravel(gt_euler_ds[:, 0])),
                    "gt_pitch": list(np.ravel(gt_euler_ds[:, 1])),
                    "gt_yaw": list(np.ravel(gt_euler_ds[:, 2])),
                    "geodesic_error": list(np.ravel(d)) if isinstance(d, np.ndarray) else d,
                    "geodesic_to_identity_prediction": list(np.ravel(ge_identity_pred)) if isinstance(ge_identity_pred, np.ndarray) else ge_identity_pred,
                    "geodesic_to_identity_ground_truth": list(np.ravel(ge_identity_gt)) if isinstance(ge_identity_gt, np.ndarray) else ge_identity_gt,
                    "from_start_prediction": list(np.ravel(from_start_p)) if isinstance(from_start_p, np.ndarray) else from_start_p,
                    "from_start_ground_truth": list(np.ravel(from_start_gt)) if isinstance(from_start_gt, np.ndarray) else from_start_gt,
                    "from_start_deviation": list(np.ravel(d_t0)) if isinstance(d_t0, np.ndarray) else d_t0,
                    "frame_to_frame_prediction": list(np.ravel(frame2frame_p)) if isinstance(frame2frame_p, np.ndarray) else frame2frame_p,
                    "frame_to_frame_ground_truth": list(np.ravel(frame2frame_gt)) if isinstance(frame2frame_gt, np.ndarray) else frame2frame_gt,
                    "frame_to_frame_deviation": list(np.ravel(d_deltat)) if isinstance(d_deltat, np.ndarray) else d_deltat,
                    "occlusion_ratio": list(np.ravel(occlusion_ratios)) if isinstance(occlusion_ratios, np.ndarray) else occlusion_ratios,
                    "face_visibility": list(np.ravel(face_visibility)) if isinstance(face_visibility, np.ndarray) else face_visibility,
                    # Keep placeholders if you want to populate later
                    "omega_x": None,
                    "omega_y": None,
                    "omega_z": None,
                    "quaternion": None,
                    "rotation_matrix": None
                }

                with open(JSONL_PRED_PATH, 'a') as f:
                    f.write(json.dumps(jsonl_entry) + '\n')


# ----------------------- aggregated summaries -----------------------

# Prepare CSV file path
CSV_ANALYSIS_PATH = Path("../../../Data/filter_results.csv")

# Open CSV file for writing
with open(CSV_ANALYSIS_PATH, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "6DRepNet360",
        "sigma",
        "avg_d",
        "avg_d_deltat",
        "median_d",
        "median_d_deltat",
        "n_frames"
    ])

    for filter_label, rows in results_by_filter.items():

        # Concatenate ALL framewise values across ALL videos
        all_d        = np.concatenate([row["d"]        for row in rows])
        all_d_deltat = np.concatenate([row["d_deltat"] for row in rows])

        # Global (frame-weighted) metrics
        avg_d        = float(np.nanmean(all_d))
        avg_d_deltat = float(np.nanmean(all_d_deltat))
        median_d        = float(np.nanmedian(all_d))
        median_d_deltat = float(np.nanmedian(all_d_deltat))

        sigma = filter_label.split()[-1] if "Gaussian" in filter_label else "none"

        writer.writerow([
            "6DRepNet360",
            sigma,
            avg_d,
            avg_d_deltat,
            median_d,
            median_d_deltat
        ])

print(f"\nFrame-weighted filter analysis written to: {CSV_ANALYSIS_PATH}")


print(f"\nAnalysis complete!\n  - Predictions JSONL: {JSONL_PRED_PATH}\n  - Aggregated results JSONL: {JSONL_ANALYSIS_PATH}\n  - Aggregated results CSV: {CSV_ANALYSIS_PATH}")

# Save per-filter aggregated results
analysis_bundle = []
for name, mae_list in results_by_filter.items():
    arr = np.array(mae_list)
    avg = np.nanmean(arr, axis=0)
    analysis_bundle.append({
        "filter_name": name,
        "metrics": ["geodesic_distance", "from_start_deviation", "frame_to_frame_deviation"],
        "mae_mean": avg.tolist(),
        "n_trials": int(arr.shape[0]),
        "sample_rate": SAMPLE_RATE,
        "model_name": "6drepnet360",
        "analysis_date": pd.Timestamp.now().isoformat()
    })

with open(JSONL_ANALYSIS_PATH, 'a') as f:
    for rec in analysis_bundle:
        f.write(json.dumps(rec) + '\n')

print(f"\nAnalysis complete!\n  - Predictions JSONL: {JSONL_PRED_PATH}\n  - Aggregated results JSONL: {JSONL_ANALYSIS_PATH}")
