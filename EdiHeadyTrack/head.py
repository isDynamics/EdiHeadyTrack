# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    head.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/26 11:14:49 by taston            #+#    #+#              #
#    Updated: 2023/04/28 11:52:19 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from .facedetector import FaceDetector
import cv2
import numpy as np
from datetime import datetime

class Head:
    def __init__(self, facedetector=FaceDetector()):
        self.facedetector = facedetector
        self.pose = {'time':    [],
                     'yaw':     [],
                     'pitch':   [],
                     'roll':    []}
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Creating Head object for {self.facedetector}:', timestamp))
        print('-'*120)
        self.calculate_pose()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Head object complete!', timestamp))
        print('-'*120)

    def calculate_pose(self):
        print('Computing head pose from tracking data...')
        for idx, time in enumerate(self.facedetector.face2d['time']):
            self.pose['time'].append(time)
            face2d = self.facedetector.face2d['landmark positions'][idx]
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

        return 
    
    # def apply_filter(self, filter=Filter()):
    #     print('Filtering data...')
    #     self.filter = filter
    #     data = pd.DataFrame.from_dict(self.pose)
    #     time = data['time']
    #     properties = ['yaw', 'pitch', 'roll']
    #     filtered_pose = {'time':    [],
    #                      'yaw':     [],
    #                      'pitch':   [],
    #                      'roll':    []}
    #     filtered_pose['time'] = time
        
    #     for property in properties:
    #         signal = np.array(data[property])
    #         filtered_signal = self.filter.apply(signal)
    #         filtered_pose[property] = filtered_signal

    #     self.pose = filtered_pose

