import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
import os

class Analyser:
    def __init__(self, *rotation_matrices):
        """
        Initialize with a list or array of 3x3 rotation matrices.
        """
        self.rotation_matrices = rotation_matrices

    def compute_relative_rotations(self, data_output=None):
        """
        Compute relative rotations (in degrees):
        - Frame-to-frame changes
        - Cumulative changes from the first frame

        Returns:
            - relative_rotations_framewise: list of geodesic distances between consecutive frames
            - relative_rotations_from_start: list of geodesic distances from the first frame
        """

        os.makedirs(data_output, exist_ok=True) if data_output else None

        series_framewise = []
        series_from_start = []

        if not self.rotation_matrices or len(self.rotation_matrices) == 0:
            return series_framewise, series_from_start

        for series in self.rotation_matrices:
            series_framewise = []
            series_from_start = []

            rmat_first = R.from_matrix(series[0])

            for i in range(1, len(series)):
                R_prev = R.from_matrix(series[i - 1])
                R_curr = R.from_matrix(series[i])

                # Frame-to-frame rotation
                R_rel = R_curr * R_prev.inv()
                angle_framewise = R_rel.magnitude()
                series_framewise.append(angle_framewise)

                # From first frame
                R_cumulative = R_curr * rmat_first.inv()
                angle_cumulative = R_cumulative.magnitude()
                series_from_start.append(angle_cumulative)

        
        return series_framewise, series_from_start

    def compute_geodesic_distances(self):
        """
        Compute the geodesic distances (in radians) between two time series of rotation matrices
        at each time step.

        Returns:
            list: Geodesic distances (in radians) at each time step.
        """
        if len(self.rotation_matrices) != 2:
            raise ValueError("Expected exactly two time series of rotation matrices.")

        series1, series2 = self.rotation_matrices
        if len(series1) != len(series2):
            raise ValueError("The two time series must have the same length.")

        geodesic_distances = []
        for R1, R2 in zip(series1, series2):
            
            R_rel = R1.T @ R2  # Relative rotation matrix
            trace = np.clip(np.trace(R_rel), -1.0, 3.0)  # Ensure trace is within valid range
            angle = np.arccos((trace - 1) / 2.0)  # Compute angle using the trace formula
            geodesic_distances.append(angle)
        
        # Convert distances from radians to degrees
        # geodesic_distances = np.degrees(geodesic_distances)

        return geodesic_distances

    def compute_angular_velocity(self, sampling_rate):
        """
        Compute angular velocity (x, y, z components in rad/s) using matrix logarithm of relative rotation matrices.

        Args:
            sampling_rate: Sampling rate in Hz (e.g., 50 or 1000)

        Returns:
            np.ndarray of shape (N-1, 3): rotational velocities in rad/s
        """
        if len(self.rotation_matrices) < 2:
            return np.empty((0, 3))

        angular_velocities = []
        dt = 1.0 / sampling_rate

        for i in range(1, len(self.rotation_matrices)):
            R1 = self.rotation_matrices[i - 1]
            R2 = self.rotation_matrices[i]

            # Relative rotation matrix: R1^T * R2
            R_rel = R1.T @ R2

            # Logarithmic map to skew-symmetric matrix
            log_R = logm(R_rel)
            omega_matrix = log_R / dt

            # Extract angular velocity vector from skew-symmetric matrix
            wx = omega_matrix[2, 1]
            wy = omega_matrix[0, 2]
            wz = omega_matrix[1, 0]

            angular_velocities.append([wx, wy, wz])

        return np.array(angular_velocities)
    