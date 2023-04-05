# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    facedetector.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/10 16:43:13 by taston            #+#    #+#              #
#    Updated: 2023/03/25 13:09:00 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import mediapipe as mp
import numpy as np
from numpy import genfromtxt
import math

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, refineLandmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
        '''
        Initialise object
        '''
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.refineLandmarks,self.minDetectionCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        '''
        Finds face mesh and draws it
        '''
        img.flags.writeable = False
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ih, iw, ic = img.shape
        self.results = self.faceMesh.process(self.imgRGB)
        face = {
                    "x": [],
                    "y": [],
                    "z": [],
                    "x_av": None,
                    "y_av": None,
                    "yaw": None,
                    "pitch": None,
                    "roll": None
                }
        p1 = (0, 0)
        p2 = (0, 0)
        face_2d = []
        face_3d = [[0, -1.126865, 7.475604], # 1
                   [-4.445859, 2.663991, 3.173422], # 33
                   [-2.456206,	-4.342621, 4.283884], # 61
                   [0, -9.403378, 4.264492], # 199
                   [4.445859, 2.663991, 3.173422], # 263
                   [2.456206, -4.342621, 4.283884]] # 291

        key_landmarks = [33, 263, 1, 61, 291, 199]
        # If something detected
        if self.results.multi_face_landmarks:
            faces = []
            for faceLandmarks in self.results.multi_face_landmarks:
                # self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_TESSELATION,
                #                             self.drawSpec,self.drawSpec)
                self.mpDraw.draw_landmarks(img, 
                                           faceLandmarks, 
                                           self.mpFaceMesh.FACEMESH_TESSELATION,
                                           None,
                                           mp.solutions.drawing_styles
                                           .get_default_face_mesh_tesselation_style())

                
                # Get landmarks
                for idx, lm in enumerate(faceLandmarks.landmark):
                    
                    face["x"].append(int(lm.x*iw))
                    face["y"].append(int(lm.y*ih))  
                    face["z"].append(lm.z)
                    if idx in key_landmarks:
                        if idx == 1:
                            nose_2d = (lm.x * iw, lm.y * ih)
                            nose_3d = (lm.x * iw, lm.y*ih, lm.z * 3000)

                        x, y = int(lm.x * iw), int(lm.y * ih)
                        face_2d.append([x, y])
                        # face_3d.append([x, y, lm.z])

                faces.append(face)
                # face_2d = [face["x"], face["y"]]
                # face_3d = [face["x"], face["y"], face["z"]]
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                # print(face_2d)
                # print(face_3d)
                # Camera properties
                # camera_matrix = genfromtxt('camera_matrix.csv', delimiter=',')
                # print(camera_matrix)
                # distortion_matrix = genfromtxt('camera_distortion.csv', delimiter=',')
                
                focal_length = ih * 1.28
                camera_matrix = np.array([[focal_length, 0, iw/2],
                                          [0, focal_length, ih/2],
                                          [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)
    
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, camera_matrix, distortion_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
                    
                # Rotational matrix
                rmat = cv2.Rodrigues(rot_vec)[0]

                # Get angles
                P = np.hstack((rmat, np.zeros((3, 1), dtype=np.float64)))
                eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
                yaw   = eulerAngles[1, 0]
                pitch = eulerAngles[0, 0]
                roll  = eulerAngles[2, 0] 
                
                # Angle correction performed to centre around zero
                if pitch < 0:
                    pitch = - 180 - pitch
                elif pitch >= 0: 
                    pitch = 180 - pitch

                # Define points for direction line drawn on face
                if nose_2d:
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] - yaw * 2), int(nose_2d[1] - pitch * 2))

        
        else:
            faces = None 
            
        if faces:
            for face in faces:
                if face["x"]:
                    face["x_av"] = np.average(face["x"])
                    face["y_av"] = np.average(face["y"])
                    face["yaw"] = yaw
                    face["pitch"] = pitch
                    face["roll"] = roll
        
        
        return img, face, faces, p1, p2