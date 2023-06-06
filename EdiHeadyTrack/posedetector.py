# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    posedetector.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/10 16:43:13 by taston            #+#    #+#              #
#    Updated: 2023/06/06 14:39:14 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import mediapipe as mp
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm
from datetime import datetime

from EdiHeadyTrack.camera import Camera
from EdiHeadyTrack.video import Video
from .camera import Camera
from .video import Video
from .filter import Filter

class PoseDetector:
    """
    A class for representing a PoseDetector
    
    ...
    
    Attributes
    ----------
    camera : Camera, optional
        Camera object to be used with PoseDetector
    face2d : dict
        dict of detected face points in 2d
    face3d : list
        list of known 3d face points (from mesh model)
    """
    def __init__(self, video=Video(), camera=Camera(), show=True):
        self.camera = camera
        self.video = video
        self.face2d = {'time': [],
                       'frame': [],
                       'key landmark positions':    [],
                       'all landmark positions':    []}
        self.pose = {'frame':   [],
                     'time':    [],
                     'yaw':     [],
                     'pitch':   [],
                     'roll':    []}
        self.show = show

class MediaPipe(PoseDetector):
    """
    A class for representing a MediaPipe PoseDetector

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
    face2d : dict
        dict of detected face points in 2d
    face3d : list
        list of known 3d face points (from mesh model)
        
    Methods
    -------
    find_faces(img)
        search for faces in a given frame
    run()
        run through the tracking procedure using MediaPipe face mesh
    """
    def __init__(self, video=Video(), camera=Camera(), show=True,
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
        super().__init__(video, camera, show)
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
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,
                                                 self.maxFaces,
                                                 self.refineLandmarks,
                                                 self.minDetectionCon,
                                                 self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.key_landmarks = [33, 263, 1, 61, 291, 199]
        self.face3d = []
        self.run()
        self.calculate_pose()
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
        out = cv2.VideoWriter('tracking.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              self.video.fps,
                              (self.video.width,self.video.height))
        # self.video.create_writer('tracking.mp4')
        self.tracking_frames = []
        while True:
            success, img = self.video.cap.read()
            if success:
                progress_bar.update(1)
                self.find_faces(img)
                if self.show == True:
                    cv2.namedWindow("EdiHeadyTrack", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("EdiHeadyTrack", int(self.video.width/2), int(self.video.height/2))
                    cv2.imshow("EdiHeadyTrack", img)
                out.write(img)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    self.video.cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    progress_bar.close()
                    print('Face tracking interuppted...')
                    break     
            else:
                self.video.cap.release()
                out.release()
                cv2.destroyAllWindows()
                progress_bar.close()
                print('Face tracking complete...')
                break

        return

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
        # print(time)
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
                            self.nose2d = (lm.x * self.video.width, lm.y * self.video.height)
                            nose3d = (lm.x * self.video.width, lm.y * self.video.height, lm.z * 3000)

                        key_landmark_positions.append(landmark_position)

                self.face2d['all landmark positions'].append(landmark_positions)
                self.face2d['key landmark positions'].append(key_landmark_positions)

        self.tracking_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        return
    
    def calculate_pose(self):
        """Calculates head pose from detected facial landmarks using 
        Perspective-n-Point (PnP) pose computation:
        
        https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html
        """
        print('Computing head pose from tracking data...')
        for idx, time in enumerate(self.face2d['time']):
            # print(time)
            self.pose['time'].append(time)
            self.pose['frame'].append(self.face2d['frame'][idx])
            face2d = self.face2d['key landmark positions'][idx]
            face2d = np.array(face2d, dtype=np.float64)
            face3d = np.array(self.face3d, dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face3d,
                                                       face2d,
                                                       self.camera.internal_matrix,
                                                       self.camera.distortion_matrix,
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

            if self.nose2d:
                nose2d = self.nose2d
                p1 = (int(nose2d[0]), int(nose2d[1]))
                p2 = (int(nose2d[0] - yaw * 2), int(nose2d[1] - pitch * 2))

        return
    
    def __str__(self):
        return f'MediaPipe Face Detector with video {self.video.filename}'
    


class TDDFA_V2(PoseDetector):
    """
    A class for representing a 3DDFA_V2 PoseDetector

    ...
    
    Attributes
    ----------
    
        
    Methods
    -------
    
    """

    def __init__(self, video=Video(), camera=Camera(), show=True):
        super().__init__(video, camera, show)
        self.run()

    def run(self):
        # import TDDFA_v2 as TDDFA_v2
        from .TDDFA_v2.FaceBoxes import FaceBoxes
        from .TDDFA_v2.TDDFA import TDDFA
        from .TDDFA_v2.utils.render import render
        from .TDDFA_v2.utils.pose import viz_pose
        from .TDDFA_v2.utils.functions import cv_draw_landmark, get_suffix
        
        import os 
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

