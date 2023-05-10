# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    head.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/26 11:14:49 by taston            #+#    #+#              #
#    Updated: 2023/05/10 11:29:01 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import numpy as np
from datetime import datetime
import pandas as pd
from .facedetector import FaceDetector
from .filter import Filter
from .sensordata import SensorData

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
    counter = 0
    def __init__(self, facedetector=FaceDetector(), id=counter):
        """
        Parameters
        ----------
        facedetector : FaceDetector, optional
            the FaceDetector object used to find the Head (default FaceDetector())
        id : str, int, float, optional
            unique identifier for each Head object (defaults to count of existing heads)
        """
        super().__init__()
        Head.counter += 1
        self.facedetector = facedetector
        # if id:
        self.id = id
        # # else:
        #     self.id = id
        self.pose = {'time':    [],
                     'yaw':     [],
                     'pitch':   [],
                     'roll':    []}
        
        self.calculate_pose()
        self.calculate_kinematics()

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
        print('Filtering data...')
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

    def calculate_kinematics(self):
        """
        Calculates kinematic data from pose time history
        """
        print('Calculating kinematic data from head pose...')

        self.velocity['time'] = self.pose['time'][1:]
        self.acceleration['time'] = self.pose['time'][2:]

        for key in list(self.pose.keys())[1:]:
            self.velocity[key] = np.diff(np.array(self.pose[key])) / np.diff(np.array(self.pose['time']))
            self.acceleration[key] = np.diff(np.array(self.velocity[key])) / np.diff(np.array(self.velocity['time']))





    
    