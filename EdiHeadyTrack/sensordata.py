# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sensordata.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/25 15:35:24 by taston            #+#    #+#              #
#    Updated: 2023/05/30 14:13:46 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import numpy as np
from datetime import datetime
import pandas as pd
from .facedetector import FaceDetector
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
    facedetector : FaceDetector
        the FaceDetector object used to find the Head
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
    
    def __init__(self, facedetector=FaceDetector(), id=_counter):
        """
        Parameters
        ----------
        facedetector : FaceDetector, optional
            the FaceDetector object used to find the Head (default FaceDetector())
        id : str, int, float, optional
            unique identifier for each Head object (defaults to count of existing heads)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Creating Head object for {facedetector}:', timestamp))
        print('-'*120)
        super().__init__()
        Head._counter += 1
        self.facedetector = facedetector
        if id:
            self.id = id
        else:
            self.id = Head._counter
        self.pose = {'time':    [],
                     'yaw':     [],
                     'pitch':   [],
                     'roll':    []}
        
        self.calculate_pose()
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
        data = pd.DataFrame.from_dict(self.pose)
        properties = ['yaw', 'pitch', 'roll']
        for property in properties:
            signal = np.array(data[property])
            filtered_signal = self.filter.apply(signal)
            self.pose[property] = list(filtered_signal)
        
        self.calculate_kinematics()
        
        return self
    
    def calculate_pose(self):
        """Calculates head pose from detected facial landmarks using 
        Perspective-n-Point (PnP) pose computation:
        
        https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html
        """
        print('Computing head pose from tracking data...')
        for idx, time in enumerate(self.facedetector.face2d['time']):
            self.pose['time'].append(time)
            face2d = self.facedetector.face2d['key landmark positions'][idx]
            face2d = np.array(face2d, dtype=np.float64)
            face3d = np.array(self.facedetector.face3d, dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face3d,
                                                       face2d,
                                                       self.facedetector.camera.internal_matrix,
                                                       self.facedetector.camera.distortion_matrix,
                                                       flags=cv2.SOLVEPNP_ITERATIVE)
            
            rmat = cv2.Rodrigues(rot_vec)[0]

            P = np.hstack((rmat, np.zeros((3, 1), dtype=np.float64)))
            eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
            yaw = eulerAngles[1, 0]
            pitch = eulerAngles[0, 0]
            roll = eulerAngles[2,0]
            
            if pitch < 0:
                pitch = - 180 - pitch
            elif pitch >= 0: 
                pitch = 180 - pitch
            
            self.pose['yaw'].append(yaw)
            self.pose['pitch'].append(pitch)
            self.pose['roll'].append(roll)

            if self.facedetector.nose2d:
                    nose2d = self.facedetector.nose2d
                    p1 = (int(nose2d[0]), int(nose2d[1]))
                    p2 = (int(nose2d[0] - yaw * 2), int(nose2d[1] - pitch * 2))

        return

    def calculate_kinematics(self):
        """
        Calculates kinematic data from pose time history
        """
        # print('Calculating kinematic data from head pose...')

        self.velocity['time'] = self.pose['time'][1:]
        self.acceleration['time'] = self.pose['time'][2:]

        for key in list(self.pose.keys())[1:]:
            self.velocity[key] = np.diff(np.array(self.pose[key])) / np.diff(np.array(self.pose['time']))
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