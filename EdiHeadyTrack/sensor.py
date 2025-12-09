import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class SensorDataLoader:
    def __init__(self, sensor_data_path, offset_to_zero=False):
        """
        Load sensor-based head pose data (Euler angles: pitch, yaw, roll).
        :param sensor_data_path: Path to the directory containing .npy files.
        """
        if not os.path.exists(sensor_data_path):
            raise FileNotFoundError(f"Sensor data directory not found: {sensor_data_path}")
        
        self.sensor_data_path = sensor_data_path
        if offset_to_zero:
            self.offset_to_zero()
    
    def load(self):
        """
        Load Euler angles (pitch, yaw, roll) from .npy files.
        :return: List of tuples (pitch, yaw, roll) sorted by filename.
        """
        sensor_files = sorted([f for f in os.listdir(self.sensor_data_path) if f.endswith('.npy')])
        euler_globals = []
        rmat_globals = []
        
        for file in sensor_files:
            file_path = os.path.join(self.sensor_data_path, file)
            euler_angles = np.load(file_path) 
            euler_globals.append(tuple(euler_angles))
            rmat_globals.append(R.from_euler('xyz', -euler_angles, degrees=True).as_matrix())

        return rmat_globals, euler_globals
    
    def load_velocity(self):
        """
        Load angular velocities from .npy files.
        :return: List of tuples (omega_x, omega_y, omega_z) sorted by filename.
        """
        sensor_files = sorted([f for f in os.listdir(self.sensor_data_path) if f.endswith('.npy')])
        angular_velocities = []
        
        for file in sensor_files:
            file_path = os.path.join(self.sensor_data_path, file)
            angular_velocities.append(np.load(file_path))

        return angular_velocities

    def offset_to_zero(self):
        """
        Offset all sensor readings so that the first pose starts at zero while maintaining relative rotations.
        """
        if self.data:
            # initial_rot = R.from_euler('xyz', self.data[0], degrees=True)
            # self.data = [tuple(R.from_matrix(R.from_euler('xyz', pose, degrees=True).as_matrix() @ initial_rot.inv().as_matrix()).as_euler('xyz', degrees=True)) for pose in self.data]
            initial_pose = np.array(self.data[0])
            self.data = [tuple(np.array(pose) - initial_pose) for pose in self.data]

    def get_pose_at_index(self, index):
        """
        Retrieve Euler angles at a specific index.
        :param index: Index of the pose in the dataset.
        :return: Tuple (pitch, yaw, roll)
        """
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range for sensor data.")
        return self.data[index]
    