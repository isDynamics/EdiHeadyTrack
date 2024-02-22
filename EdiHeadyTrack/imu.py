from .sensordata import IMU
import pandas as pd

class Wax9(IMU):
    """
    A class representing a Wax9 IMU: https://axivity.com/downloads/wax9
    
    ...
    
    Attributes
    ----------
    columns : list
        list of sensor data column headings

    Methods
    -------
    extract_from_file()
        Extracts kinematic data from file provided
    """
    def __init__(self, filename, time_offset=0, id=False):
        """
        Parameters
        ----------
        filename : str
            file containing IMU sensor data
        id : int, float, str
            unique identifier given to IMU
        time_offset : float
            time offset applied to IMU data to sync with head pose data
        """
        super().__init__(filename, time_offset, id)
        self.columns = ['sensor', 
                        'received time','sample number','sample time',
                        'accelX','accelY','accelZ',
                        'gyroX','gyroY','gyroZ',
                        'magX','magY','magZ']
        self.extract_from_file()

    def extract_from_file(self):
        """
        Extracts kinematic data from file provided
        """
        data = pd.read_csv(self.filename)
        data.columns = self.columns
        data['adjusted time'] = data['sample time'] - data['sample time'][0] + self.time_offset
        data = data.dropna()
        import numpy as np
        # Velocity
        self.velocity['time'] = data['adjusted time']
        self.velocity['yaw'] = data['gyroX']
        self.velocity['pitch'] = data['gyroY']
        self.velocity['roll'] = data['gyroZ']
        # Acceleration
        self.acceleration['time'] = data['adjusted time']
        self.acceleration['yaw'] = data['accelX']
        self.acceleration['pitch'] = data['accelY']
        self.acceleration['roll'] = data['accelZ']

        return
