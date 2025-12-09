import EdiHeadyTrack as EHT
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------------
# Experiment setup
# ------------------------------------------------------------------
participants = ['P003024', 'P003272', 'P003336', 'P003489', 'P003540',
                'P003682', 'P003726', 'P003730', 'P003940', 'P004298']
trials = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6',
          'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
camera_names = ['F', 'S']
# participants = ['P003024']
# trials = ['L1']
# camera_names = ['F']

SAMPLE_RATE = 50.0
DATA_ROOT = Path("../../../Data")

# Output paths
PLOT_BASE = DATA_ROOT / "Plots/vggheads/filtered_gaussian"
JSONL_PRED_PATH = DATA_ROOT / "predictions_vggheads_filterstudy.jsonl"
JSONL_ANALYSIS_PATH = DATA_ROOT / "analysis_results_vggheads_filterstudy.jsonl"
PLOT_BASE.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Gaussian filters to test systematically
# ------------------------------------------------------------------
def fmt_sigma(s):
    if s is None:
        return "none"
    return f"{s}".replace(".", "p")

filters_to_test = [
    ("None", None, None),
    *[(f"Gaussian {s:.2f}",
       EHT.SO3Filter(sigma=s, spike_deg=10.0, sample_rate_hz=SAMPLE_RATE),
       s)
      for s in np.arange(0.25, 5.25, 0.25)]
]

results_by_filter = defaultdict(list)

# ------------------------------------------------------------------
# Loop over participants / trials / cameras
# ------------------------------------------------------------------
for participant in participants:
    for trial in trials:
        for camera_name in camera_names:

            video_path = DATA_ROOT / f"Videos/Processed/{participant}_{trial}_{camera_name}.avi"
            head_detections_path = DATA_ROOT / f"HeadDetection/vggheads/successful/{participant}_{trial}_{camera_name}"
            frames_path = DATA_ROOT / f"Frames/Sorted/{participant}_{trial}_{camera_name}/"
            camera_extrinsics_path = DATA_ROOT / f"Extrinsics/Parsed/{participant}_{camera_name}.json"

            trial_type = trial[0]  # 'L' or 'R'
            trial_repeat = int(trial[1])

            if not (video_path.exists() and head_detections_path.exists() and camera_extrinsics_path.exists() and frames_path.exists()):
                print(f"Skipping {participant} {trial} {camera_name} — missing files.")
                continue

            # Load GT data from dataset.jsonl
            gt_entry = None
            with open(DATA_ROOT / "dataset.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if (data.get('Participant') == participant and
                        data.get('Type') == trial_type and
                        data.get('Repeat') == trial_repeat):
                        gt_entry = data
                        break

            if gt_entry is None:
                print(f"Skipping {participant} {trial} {camera_name} — no GT found.")
                continue

            # Downsample GT (1000 Hz → 50 Hz)
            gt_rmat = np.array(gt_entry['rotation_matrix'])
            gt_rmat_ds = gt_rmat[::20]
            gt_euler_ds = np.column_stack([gt_entry['roll'], gt_entry['pitch'], gt_entry['yaw']])[::20]

            # Initialise EdiHeadyTrack components
            video = EHT.Video(str(video_path))
            frames = EHT.HeadDetector.load_detections(frames_path)
            head_detections = EHT.HeadDetector.load_detections(str(head_detections_path))
            head_pose_model = EHT.HeadPoseModel('vggheads')
            camera = EHT.Camera(str(camera_extrinsics_path))

            # Run prediction
            rmat_local_raw, rmat_global_raw, euler_local_raw, euler_global_raw = head_pose_model.predict(
                frames, camera
            )

            if len(rmat_global_raw) != len(gt_rmat_ds):
                print(f"Skipping {participant} {trial} {camera_name} — length mismatch.")
                continue

            # ------------------------------------------------------------------
            # Systematic Gaussian filtering
            # ------------------------------------------------------------------
            for filter_label, filt, sigma_val in filters_to_test:
                
                sigma_tag = fmt_sigma(sigma_val)
                plot_dir_eulers = PLOT_BASE / f"eulers_sigma_{sigma_tag}"
                plot_dir_geos   = PLOT_BASE / f"geodesics_sigma_{sigma_tag}"
                plot_dir_eulers.mkdir(parents=True, exist_ok=True)
                plot_dir_geos.mkdir(parents=True, exist_ok=True)

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

                # Align global predictions to GT
                rmat_global_aligned, A_sys = EHT.align_to_gt(
                    np.stack(rmat_global),
                    gt_rmat_ds,
                    mask=None,
                    tol=1e-4,
                    max_iter=100000,
                    return_delta=True
                )
                rmat_global = [m for m in rmat_global_aligned]

                # Recompute Euler angles after alignment
                euler_global = R.from_matrix(np.stack(rmat_global)).as_euler('xyz', degrees=False).tolist()
                euler_local  = R.from_matrix(np.stack(rmat_local)).as_euler('xyz', degrees=False).tolist()

                # ------------------------------------------------------------------
                # Metrics
                # ------------------------------------------------------------------
                identity = [np.eye(3) for _ in range(len(rmat_global))]
                ge_identity_pred = EHT.Analyser(rmat_global, identity).compute_geodesic_distances()
                ge_identity_gt   = EHT.Analyser(gt_rmat_ds, identity).compute_geodesic_distances()

                analyser_p  = EHT.Analyser(rmat_global)
                analyser_gt = EHT.Analyser(gt_rmat_ds)
                frame2frame_p, from_start_p = analyser_p.compute_relative_rotations(data_output=None)
                frame2frame_gt, from_start_gt = analyser_gt.compute_relative_rotations(data_output=None)

                analyser_p_gt = EHT.Analyser(rmat_global, gt_rmat_ds)
                d = analyser_p_gt.compute_geodesic_distances()

                d_t0 = np.abs(np.array(from_start_p) - np.array(from_start_gt))
                d_deltat = np.abs(np.array(frame2frame_p) - np.array(frame2frame_gt))

                mae_d = float(np.mean(d))
                mae_d_t0 = float(np.mean(d_t0[0][1:])) if len(d_t0) > 0 else float('nan')
                mae_d_deltat = float(np.mean(d_deltat[0][1:])) if len(d_deltat) > 0 else float('nan')

                results_by_filter[filter_label].append(np.array([mae_d, mae_d_t0, mae_d_deltat]))

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
                # ------------------------------------------------------------------
                # JSONL output per trial & filter
                # ------------------------------------------------------------------
                sigma_tag = fmt_sigma(sigma_val)
                times = [i / SAMPLE_RATE for i in range(len(rmat_global))]
                rolls, pitches, yaws = zip(*euler_local)

                jsonl_entry = {
                    "Impact ID": f"{participant}_{trial}_{camera_name}",
                    "Data source": "VGGHeads",
                    "Sample rate (Hz)": SAMPLE_RATE,
                    "Participant": participant,
                    "Camera": camera_name,
                    "Type": trial_type,
                    "Repeat": trial_repeat,
                    "filtered": filtered_flag,
                    "filter": {
                        "type": "gaussian" if sigma_val is not None else "none",
                        "sigma": sigma_val
                    },
                    "times": list(times),
                    "roll": list(rolls),
                    "pitch": list(pitches),
                    "yaw": list(yaws),
                    "gt_roll": list(gt_euler_ds[:, 0]),
                    "gt_pitch": list(gt_euler_ds[:, 1]),
                    "gt_yaw": list(gt_euler_ds[:, 2]),
                    "geodesic_error": list(np.ravel(d)),
                    "from_start_deviation": list(np.ravel(d_t0)),
                    "frame_to_frame_deviation": list(np.ravel(d_deltat)),
                    "mae_from_start_deviation": mae_d_t0,
                    "mae_frame_to_frame_deviation": mae_d_deltat
                }

                with open(JSONL_PRED_PATH, 'a') as f:
                    f.write(json.dumps(jsonl_entry) + '\n')

# ------------------------------------------------------------------
# Aggregated summaries
# ------------------------------------------------------------------
import csv
# Prepare CSV file path
CSV_ANALYSIS_PATH = Path("../../../Data/filter_results.csv")

# Open CSV file for writing
with open(CSV_ANALYSIS_PATH, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header
    
    # Write data for each filter
    for name, mae_list in results_by_filter.items():
        arr = np.array(mae_list)  # shape (n_trials, 3)
        avg = np.nanmean(arr, axis=0)
        median = np.nanmedian(arr, axis=0)
        
        # Extract sigma value from filter name if applicable
        sigma = name.split()[-1] if "Gaussian" in name else "none"
        
        # Write row
        csvwriter.writerow(["VGGHeads", sigma, avg[0], avg[2], median[0], median[2]])

# print(f"\nAnalysis complete!\n  - Predictions JSONL: {JSONL_PRED_PATH}\n  - Aggregated results JSONL: {JSONL_ANALYSIS_PATH}\n  - Aggregated results CSV: {CSV_ANALYSIS_PATH}")

# with open(JSONL_ANALYSIS_PATH, 'a') as f:
#     for rec in analysis_bundle:
#         f.write(json.dumps(rec) + '\n')

# print(f"\nAnalysis complete!\n  - Predictions JSONL: {JSONL_PRED_PATH}\n  - Aggregated results JSONL: {JSONL_ANALYSIS_PATH}")
