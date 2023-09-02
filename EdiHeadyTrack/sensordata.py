# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sensordata.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/25 15:35:24 by taston            #+#    #+#              #
#    Updated: 2023/09/01 11:33:29 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import numpy as np
from datetime import datetime
import pandas as pd
from .posedetector import PoseDetector
from .filter import Filter

class SensorData:
    """
    A class representing SensorData

    Attributes
    ----------
    velocity : dict
        time history of rotational velocities
    acceleration : dict
        time history of rotational accelerations
    """
    def __init__(self):
        self.velocity = {'time':    [],
                         'yaw':     [],
                         'pitch':   [],
                         'roll':    []}
        self.acceleration = {'time':    [],
                             'yaw':     [],
                             'pitch':   [],
                             'roll':    []}
        

class Head(SensorData):
    """
    A class used to represent a Head obtained using head pose estimation
    techniques

    ...
    
    Attributes
    ----------
    posedetector : PoseDetector
        the PoseDetector object used to find the Head
    filter : Filter
        Filter object used for data filtering
    id : str, int, float
        unique identifier given to Head
    pose : dict
        dict containing Head pose time history

    Methods
    -------
    apply_filter(filter)
        applies filter to head pose data
    calculate_kinematics()
        calculates kinematic data from pose time history
    calculate_pose()
        computes HeadPose from detected facial landmarks
    """
    _counter = 0
    
    def __init__(self, posedetector=PoseDetector(), id=_counter):
        """
        Parameters
        ----------
        posedetector : PoseDetector, optional
            the PoseDetector object used to find the Head (default PoseDetector())
        id : str, int, float, optional
            unique identifier for each Head object (defaults to count of existing heads)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Creating Head object for {posedetector}:', timestamp))
        print('-'*120)
        super().__init__()
        Head._counter += 1
        self.posedetector = posedetector
        self.pose = posedetector.pose
        if id:
            self.id = id
        else:
            self.id = Head._counter
        
        self.calculate_kinematics()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Head object complete!', timestamp))
        print('-'*120)

    def apply_filter(self, filter=Filter()):
        """Applies filter to head pose data and updates pose 
        
        Parameters
        ----------
        filter : Filter, optional
            Filter object used to filter data (default Filter())

        Returns
        -------
        self
        """
        # print('Filtering data...')
        self.filter = filter
        data = pd.DataFrame.from_dict(self.posedetector.pose)
        properties = ['yaw', 'pitch', 'roll']
        for property in properties:
            signal = np.array(data[property])
            filtered_signal = self.filter.apply(signal)
            self.posedetector.pose[property] = list(filtered_signal)
        
        self.calculate_kinematics()
        
        return self
    

    def calculate_kinematics(self):
        """
        Calculates kinematic data from pose time history
        """


        self.velocity['time'] = self.posedetector.pose['time'][1:]
        self.acceleration['time'] = self.posedetector.pose['time'][2:]

        for key in list(self.posedetector.pose.keys())[2:]:
            self.velocity[key] = np.diff(np.array(self.posedetector.pose[key])) / np.diff(np.array(self.posedetector.pose['time']))
            self.acceleration[key] = np.diff(np.array(self.velocity[key])) / np.diff(np.array(self.velocity['time']))

        return self


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
    _counter = 0
    def __init__(self, filename=None, time_offset=0, id=_counter):
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
        IMU._counter += 1
        self.filename = filename
        self.time_offset = time_offset
        if id:
            self.id = id
        else:
            self.id = IMU._counter

    def apply_filter(self, filter=Filter()):
        """Applies filter to sensor data and updates
        
        Parameters
        ----------
        filter : Filter, optional
            Filter object used to filter data (default Filter())

        Returns
        -------
        self
        """
        # print('Filtering data...')
        self.filter = filter
        data = pd.DataFrame.from_dict(self.velocity)
        # print(data)
        properties = ['yaw', 'pitch', 'roll']
        for property in properties:
            signal = np.array(data[property])
            filtered_signal = self.filter.apply(signal)
            self.velocity[property] = list(filtered_signal)
        # self.velocity['time'] = self.velocity['time'][~np.isnan(signal)]
        # print(signal)
        # print(filtered_signal)
        return self