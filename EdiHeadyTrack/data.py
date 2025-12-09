# EdiHeadyTrack/EdiHeadyTrack/data.py

class Data:
    """
    Represents time-series data (e.g., Euler angles and angular velocities).
    This instance can be from head pose estimation or from sensor data (motion capture).
    """
    def __init__(self, time=None, euler_angles=None, angular_velocities=None):
        self.time = time if time is not None else []
        self.euler_angles = euler_angles if euler_angles is not None else {}  # e.g., {'roll': [...], 'pitch': [...], 'yaw': [...]}
        self.angular_velocities = angular_velocities if angular_velocities is not None else {}  # e.g., {'omega_x': [...], ...}
    
    def compute_velocities(self):
        """If only Euler angles are available, compute angular velocities (dummy example)."""
        if self.euler_angles and not self.angular_velocities:
            n = len(self.time)
            self.angular_velocities = {
                'omega_x': [0.0] * n,
                'omega_y': [0.0] * n,
                'omega_z': [0.0] * n,
            }
    
    def align_to(self, other):
        """
        Optionally, interpolate this Data instance so that its time vector matches 'other'.
        Implement interpolation as needed.
        """
        # Dummy implementation; in practice, you might use scipy.interpolate or pandas:
        # For now, simply return self.
        return self
    
    def get_data(self):
        """Return a dictionary representation of the data."""
        return {
            'time': self.time,
            'euler_angles': self.euler_angles,
            'angular_velocities': self.angular_velocities
        }
