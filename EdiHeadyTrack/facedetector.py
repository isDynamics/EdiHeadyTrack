# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    facedetector.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/10 16:43:13 by taston            #+#    #+#              #
#    Updated: 2023/05/10 11:29:21 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import mediapipe as mp
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm
from datetime import datetime
from .camera import Camera
from .video import Video
from .filter import Filter

class FaceDetector:
    """
    A class for representing a FaceDetector
    
    ...
    
    Attributes
    ----------
    camera : Camera, optional
        Camera object to be used with FaceDetector
    face2d : dict
        dict of detected face points in 2d
    face3d : list
        list of known 3d face points (from mesh model)
    """
    def __init__(self, video=Video(), camera=Camera()):
        self.camera = camera
        self.video = video
        self.face2d = {'time': [],
                       'frame': [],
                       'key landmark positions':    [],
                       'all landmark positions':    []}
        self.face3d = []

class MediaPipe(FaceDetector):
    """
    A class for representing a MediaPipe FaceDetector

    ...
    
    Attributes
    ----------
    drawSpec : DrawingSpec
        Drawing specifications for face mesh
    face3d : list
        list of known 3d face points (from mesh model)
    faceMesh : FaceMesh
        MediaPipe FaceMesh object 
    key_landmarks : list
        list of key landmark positions used for pose estimation
    maxFaces : int
        int number of maximum faces to be detected in video
    minDetectionCon : float
        float representing minimum detection confidence
    minTrackCon : float
        float representing minimum tracking confidence
    mpDraw : module
        MediaPipe drawing utilities
    mpFaceMesh : module
        MediaPipe face mesh utilities
    refineLandmarks : bool
        bool for choosing if landmarks will be refined
    staticMode : bool
        bool representing if searching for landmarks on static frame
        or non static video file
    tracking_frames : list
        list of frame numbers which have successfully been tracked
        
    Methods
    -------
    find_faces(img)
        search for faces in a given frame
    run()
        run through the tracking procedure using MediaPipe face mesh
    """
    def __init__(self, video=Video(), camera=Camera(),
                 staticMode=False, maxFaces=1, refineLandmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
        """
        Parameters
        ----------
        video : Video, optional
            video to perform head pose estimation on (default is empty Video)
        camera : Camera, optional
            camera used to capture video (default is uncalibrated Camera)
        staticMode : bool, optional
            flag specifying if tracking is static or not (default False)
        maxFaces : int, optional
            maximum number of faces allowed to be detected in video (default 1)
        refineLandmarks : bool, optional
            flag setting if landmarks should be refined or not (default False)
        minDetectionCon : float, optional
            minimum detection confidence (default 0.5)
        minTrackCon : float, optional
            minimum tracking confidence (default 0.5)
        """
        super().__init__(video, camera)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Creating MediaPipe object for video {self.video.filename}:', timestamp))
        print('-'*120)
        print(self.video)
        self.staticMode = staticMode
        self.refineLandmarks = refineLandmarks
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        print(type(self.mpDraw), type(self.mpFaceMesh))
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,
                                                 self.maxFaces,
                                                 self.refineLandmarks,
                                                 self.minDetectionCon,
                                                 self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.key_landmarks = [33, 263, 1, 61, 291, 199]
        self.run()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'MediaPipe object complete!', timestamp))
        print('-'*120)
    
    def run(self):
        """Runs through MediaPipe tracking process, calling relevant
        functions
        
        More information found here:

        https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
        """
        print('Running MediaPipe Face Mesh on selected video...')
        self.face3d = [[0, -1.126865, 7.475604], # 1
                       [-4.445859, 2.663991, 3.173422], # 33
                       [-2.456206,	-4.342621, 4.283884], # 61
                       [0, -9.403378, 4.264492], # 199
                       [4.445859, 2.663991, 3.173422], # 263
                       [2.456206, -4.342621, 4.283884]] # 291
        
        progress_bar = tqdm(range(self.video.total_frames))
        self.tracking_frames = []
        while True:
            success, img = self.video.cap.read()
            if success:
                progress_bar.update(1)
                # current_frame = int(self.video.cap.get(1))
                self.find_faces(img)
                cv2.namedWindow("EdiHeadyTrack", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("EdiHeadyTrack", 640, 360)
                cv2.imshow("EdiHeadyTrack", img)
                # cv2.imwrite(f'tracking frames/{current_frame}.png', img)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    progress_bar.close()
                    print('Face tracking interuppted...')
                    break     
            else:
                progress_bar.close()
                print('Face tracking complete...')
                break

    def find_faces(self, img):
        """Finds faces in a supplied image

        Parameters
        ----------
        img : ndarray
            ndarray representing the image in which faces should be found
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        frame_number = int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES))
        time = frame_number / self.video.fps
        
        if results.multi_face_landmarks:
            self.face2d['time'].append(time)
            self.face2d['frame'].append(frame_number)
            landmark_positions=[]
            key_landmark_positions=[]
            for faceLandmarks in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img,
                                           faceLandmarks,
                                           self.mpFaceMesh.FACEMESH_TESSELATION,
                                           None,
                                           mp.solutions.drawing_styles
                                           .get_default_face_mesh_tesselation_style())
                
                for idx, lm in enumerate(faceLandmarks.landmark):
                    x = int(lm.x * self.video.width)
                    y = int(lm.y * self.video.height)

                    landmark_position = [x,y]
                    landmark_positions.append(landmark_position)
                    
                    if idx in self.key_landmarks:
                        if idx == 1:
                            nose2d = (lm.x * self.video.width, lm.y * self.video.height)
                            nose3d = (lm.x * self.video.width, lm.y * self.video.height, lm.z * 3000)

                        key_landmark_positions.append(landmark_position)
                
                self.face2d['all landmark positions'].append(landmark_positions)
                self.face2d['key landmark positions'].append(key_landmark_positions)

        self.tracking_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return
    
    def __str__(self):
        return f'MediaPipe Face Detector with video {self.video.filename}'


# class OpenPose(FaceDetector):
#     def __init__(self):
#         ...

# class FaceMeshDetector():
#     def __init__(self, staticMode=False, maxFaces=2, refineLandmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
#         '''
#         Initialise object
#         '''
#         self.staticMode = staticMode
#         self.maxFaces = maxFaces
#         self.refineLandmarks = refineLandmarks
#         self.minDetectionCon = minDetectionCon
#         self.minTrackCon = minTrackCon

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpFaceMesh = mp.solutions.face_mesh
#         self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,
#                                                  self.maxFaces,
#                                                  self.refineLandmarks,
#                                                  self.minDetectionCon,
#                                                  self.minTrackCon)
#         self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

#     def findFaceMesh(self, img, draw=True):
#         '''
#         Finds face mesh and draws it
#         '''
#         img.flags.writeable = False
#         self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         ih, iw, ic = img.shape
#         self.results = self.faceMesh.process(self.imgRGB)
#         face = {
#                     "x": [],
#                     "y": [],
#                     "z": [],
#                     "x_av": None,
#                     "y_av": None,
#                     "yaw": None,
#                     "pitch": None,
#                     "roll": None
#                 }
#         p1 = (0, 0)
#         p2 = (0, 0)
#         face_2d = []
#         face_3d = [[0, -1.126865, 7.475604], # 1
#                    [-4.445859, 2.663991, 3.173422], # 33
#                    [-2.456206,	-4.342621, 4.283884], # 61
#                    [0, -9.403378, 4.264492], # 199
#                    [4.445859, 2.663991, 3.173422], # 263
#                    [2.456206, -4.342621, 4.283884]] # 291

#         key_landmarks = [33, 263, 1, 61, 291, 199]
#         # If something detected
#         if self.results.multi_face_landmarks:
#             faces = []
#             for faceLandmarks in self.results.multi_face_landmarks:
#                 # self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_TESSELATION,
#                 #                             self.drawSpec,self.drawSpec)
#                 self.mpDraw.draw_landmarks(img, 
#                                            faceLandmarks, 
#                                            self.mpFaceMesh.FACEMESH_TESSELATION,
#                                            None,
#                                            mp.solutions.drawing_styles
#                                            .get_default_face_mesh_tesselation_style())

                
#                 # Get landmarks
#                 for idx, lm in enumerate(faceLandmarks.landmark):
                    
#                     face["x"].append(int(lm.x*iw))
#                     face["y"].append(int(lm.y*ih))  
#                     face["z"].append(lm.z)
#                     if idx in key_landmarks:
#                         if idx == 1:
#                             nose_2d = (lm.x * iw, lm.y * ih)
#                             nose_3d = (lm.x * iw, lm.y*ih, lm.z * 3000)

#                         x, y = int(lm.x * iw), int(lm.y * ih)
#                         face_2d.append([x, y])
#                         # face_3d.append([x, y, lm.z])

#                 faces.append(face)
#                 # face_2d = [face["x"], face["y"]]
#                 # face_3d = [face["x"], face["y"], face["z"]]
#                 face_2d = np.array(face_2d, dtype=np.float64)
#                 face_3d = np.array(face_3d, dtype=np.float64)
#                 # print(face_2d)
#                 # print(face_3d)
#                 # Camera properties
#                 # camera_matrix = genfromtxt('camera_matrix.csv', delimiter=',')
#                 # print(camera_matrix)
#                 # distortion_matrix = genfromtxt('camera_distortion.csv', delimiter=',')
                
#                 focal_length = ih * 1.28
#                 camera_matrix = np.array([[focal_length, 0, iw/2],
#                                           [0, focal_length, ih/2],
#                                           [0, 0, 1]])
#                 distortion_matrix = np.zeros((4, 1), dtype=np.float64)
    
#                 # Solve PnP
#                 success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, camera_matrix, distortion_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
                    
#                 # Rotational matrix
#                 rmat = cv2.Rodrigues(rot_vec)[0]

#                 # Get angles
#                 P = np.hstack((rmat, np.zeros((3, 1), dtype=np.float64)))
#                 eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
#                 yaw   = eulerAngles[1, 0]
#                 pitch = eulerAngles[0, 0]
#                 roll  = eulerAngles[2, 0] 
                
#                 # Angle correction performed to centre around zero
#                 if pitch < 0:
#                     pitch = - 180 - pitch
#                 elif pitch >= 0: 
#                     pitch = 180 - pitch

#                 # Define points for direction line drawn on face
#                 if nose_2d:
#                     p1 = (int(nose_2d[0]), int(nose_2d[1]))
#                     p2 = (int(nose_2d[0] - yaw * 2), int(nose_2d[1] - pitch * 2))

        
#         else:
#             faces = None 
            
#         if faces:
#             for face in faces:
#                 if face["x"]:
#                     face["x_av"] = np.average(face["x"])
#                     face["y_av"] = np.average(face["y"])
#                     face["yaw"] = yaw
#                     face["pitch"] = pitch
#                     face["roll"] = roll
        
        
#         return img, face, faces, p1, p2
    
