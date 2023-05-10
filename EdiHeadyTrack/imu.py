# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    imu.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/26 16:03:23 by taston            #+#    #+#              #
#    Updated: 2023/05/10 11:38:08 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from .sensordata import SensorData
import pandas as pd


class IMU(SensorData):
    """
    A class used to represent an IMU

    ...
    
    Attributes
    ----------
    filename : str
        file containing IMU sensor data
    id : int, float, str
        unique identifier given to IMU
    time_offset : float
        time offset applied to IMU data to sync with head pose data
    """
    counter = 0
    def __init__(self, filename=None, time_offset=0, id=counter):
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
        super().__init__()
        IMU.counter += 1
        self.filename = filename
        self.time_offset = time_offset
        if id:
            self.id = id
        else:
            self.id = IMU.counter


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

        self.velocity['time'] = data['adjusted time']
        self.velocity['yaw'] = data['gyroX']
        self.velocity['pitch'] = data['gyroY']
        self.velocity['roll'] = data['gyroZ']