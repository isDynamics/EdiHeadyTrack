import numpy as np
from scipy.spatial.transform import Rotation as R

class Test:
    def __init__(self, pred_euler=None, gt_euler=None, pred_rmat=None, gt_rmat=None):
        """
        pred_euler, gt_euler: Nx3 numpy arrays of Euler angles in degrees (roll, pitch, yaw)
        pred_rmat, gt_rmat: Nx3x3 numpy arrays of rotation matrices
        """
        self.pred_euler = np.array(pred_euler) if pred_euler is not None else None
        self.gt_euler = np.array(gt_euler) if gt_euler is not None else None
        self.pred_rmat = np.array(pred_rmat) if pred_rmat is not None else None
        self.gt_rmat = np.array(gt_rmat) if gt_rmat is not None else None

    def mae_euler(self):
        """Compute MAE for each Euler angle: Roll, Pitch, Yaw"""
        if self.pred_euler is None or self.gt_euler is None:
            raise ValueError("Euler angles not provided.")
        return np.mean(np.abs(self.pred_euler - self.gt_euler), axis=0)

    def overall_mae_euler(self):
        """Compute overall MAE across all angles"""
        return np.mean(self.mae_euler())

    def geodesic_distance(self):
        """
        Compute geodesic distance (in degrees) between rotation matrices.
        Formula: theta = arccos((trace(R1^T R2) - 1) / 2)
        """
        if self.pred_rmat is None or self.gt_rmat is None:
            raise ValueError("Rotation matrices not provided.")
        assert self.pred_rmat.shape == self.gt_rmat.shape, "Shape mismatch in rotation matrices"

        angles = []
        for R1, R2 in zip(self.pred_rmat, self.gt_rmat):
            delta_R = R1 @ R2.T
            trace = np.clip(np.trace(delta_R), -1.0, 3.0)
            theta = np.arccos((trace - 1) / 2)
            angles.append(np.degrees(theta))

        return np.mean(angles)
