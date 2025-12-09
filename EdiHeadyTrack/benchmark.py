# EdiHeadyTrack/EdiHeadyTrack/benchmark.py

import numpy as np
import matplotlib.pyplot as plt

class Benchmark:
    """
    Compares head pose estimation data against motion capture data.
    """
    def __init__(self, prediction_data, ground_truth_data):
        self.prediction = prediction_data
        self.ground_truth = ground_truth_data
    
    def align_data(self):
        """
        Align the prediction and ground truth data to the same time base.
        For simplicity, we assume they are roughly aligned.
        Otherwise, perform interpolation on one dataset.
        """
        # Dummy implementation: assume both are aligned.
        # In practice, you could use self.prediction.align_to(self.ground_truth)
        return self.prediction, self.ground_truth
    
    def compute_errors(self):
        """
        Compute error metrics (e.g., mean absolute error) for each angle.
        """
        pred, gt = self.align_data()
        errors = {}
        for angle in ['roll', 'pitch', 'yaw']:
            pred_vals = np.array(pred.euler_angles.get(angle, []))
            gt_vals = np.array(gt.euler_angles.get(angle, []))
            # Ensure same length; otherwise, use interpolation (not implemented here)
            if len(pred_vals) != len(gt_vals):
                raise ValueError("Data length mismatch; consider aligning the data.")
            mae = np.mean(np.abs(pred_vals - gt_vals))
            errors[f"mae_{angle}"] = mae
        return errors
    
    def plot_comparison(self):
        """
        Plot the prediction and ground truth time series for visual comparison.
        """
        pred, gt = self.align_data()
        time = pred.time  # or gt.time, assuming they are aligned
        plt.figure(figsize=(12, 8))
        for angle in ['roll', 'pitch', 'yaw']:
            plt.plot(time, pred.euler_angles.get(angle, []), label=f"Prediction {angle}")
            plt.plot(time, gt.euler_angles.get(angle, []), label=f"Ground Truth {angle}", linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (units)")
        plt.title("Head Pose Comparison")
        plt.legend()
        plt.show()