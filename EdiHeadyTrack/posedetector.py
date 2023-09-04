# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    posedetector.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/10 16:43:13 by taston            #+#    #+#              #
#    Updated: 2023/09/04 10:54:07 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import mediapipe as mp
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm
from datetime import datetime
import time

from EdiHeadyTrack.camera import Camera
from EdiHeadyTrack.video import Video
from .camera import Camera
from .video import Video
from .filter import Filter

class PoseDetector:
    """
    Abstract class for representing a PoseDetector
    
    ...
    
    Attributes
    ----------
    camera : Camera, optional
        Camera object to be used with PoseDetector
    face2d : dict
        dict of detected face points in 2d
    face3d : list
        list of known 3d face points (from mesh model)
    show : bool, optional
            flag for displaying video output (default True)
    tracking_frames : list, ndarray
        list containing frames which have successfully been tracked
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
        self.tracking_frames = []

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
                 staticMode=False, maxFaces=1, refineLandmarks=True, minDetectionCon=0.5, minTrackCon=0.5):
        """
        Parameters
        ----------
        video : Video, optional
            video to perform head pose estimation on (default is empty Video)
        camera : Camera, optional
            camera used to capture video (default is uncalibrated Camera)
        show : bool, optional
            flag for displaying video output (default True)
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
        # self.calculate_pose()
        # offset values
        self.pose['yaw'] = [val - self.pose['yaw'][0] for val in self.pose['yaw']]
        self.pose['pitch'] = [val - self.pose['pitch'][0] for val in self.pose['pitch']]
        self.pose['roll'] = [val - self.pose['roll'][0] for val in self.pose['roll']]
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
        self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
                    # self.video.cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    progress_bar.close()
                    print('Face tracking interuppted...')
                    break     
            else:
                # self.video.cap.release()
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
        from .TDDFA_v2.utils.pose import viz_pose
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        frame_number = int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES))
        time = frame_number / self.video.fps
        # print(time)
        if results.multi_face_landmarks:
            self.face2d['time'].append(time)
            self.pose['time'].append(time)
            self.face2d['frame'].append(frame_number)
            self.pose['frame'].append(frame_number)
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
                
                yaw, pitch, roll, p1, p2 = self.calculate_pose(key_landmark_positions)
                
                # if self.nose2d:
                #     nose2d = self.nose2d
                #     p1 = (int(nose2d[0]), int(nose2d[1]))
                #     p2 = (int(nose2d[0] - yaw * 2), int(nose2d[1] - pitch * 2))

                # cv2.line(img, p1, p2, (255,0,0), 3)
                self.pose['yaw'].append(yaw)
                self.pose['pitch'].append(pitch)
                self.pose['roll'].append(roll)
  

                # res, pose = viz_pose(res, param_lst, [key_landmark_positions]) 
                # self.pose['yaw'].append(pose[0])
                # self.pose['pitch'].append(pose[1])
                # self.pose['roll'].append(pose[2])   

                
        
        self.tracking_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        return
    
    def calculate_pose(self, face2d):
        """Calculates head pose from detected facial landmarks using 
        Perspective-n-Point (PnP) pose computation:
        
        https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html
        """
        # print('Computing head pose from tracking data...')
        # for idx, time in enumerate(self.face2d['time']):
        #     # print(time)
        #     self.pose['time'].append(time)
        #     self.pose['frame'].append(self.face2d['frame'][idx])
        #     face2d = self.face2d['key landmark positions'][idx]
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
        
        yaw *= -1
        pitch *= -1
        
        if self.nose2d:
            nose2d = self.nose2d
            p1 = (int(nose2d[0]), int(nose2d[1]))
            p2 = (int(nose2d[0] - yaw * 2), int(nose2d[1] - pitch * 2))
        
        return yaw, pitch, roll, p1, p2
    
    def __str__(self):
        return f'MediaPipe Face Detector with video {self.video.filename}'
    


class TDDFA_V2(PoseDetector):
    """
    A class for representing a 3DDFA_V2 PoseDetector

    ...
    
    Attributes
    ----------
    current_path : str
        string for tracking file path of 3DDFA source files, required
        due configuration of 3DDFA module.
    tracking_frames : list, ndarray
        list containing frames which have successfully been tracked
        
    Methods
    -------
    run(args)
        run through the tracking procedure using 3DDFA_v2
    run_smooth(args)
        run through the tracking procedure using 3DDFA_v2 with smoothing
    """

    def __init__(self, video=Video(), camera=Camera(), show=True, smooth=False, dense=False):
        """
        Parameters
        ----------
        video : Video, optional
            video to perform head pose estimation on (default is empty Video)
        camera : Camera, optional
            camera used to capture video (default is uncalibrated Camera)
        show : bool, optional
            flag for displaying video output (default True)
        smooth : bool
            flag for using smooth tracking by looking n frames ahead (default False)
        dense : bool
            flag for using dense facial landmark model with 38,365 landmarks (default False, with 68 landmarks)
        
        """
        super().__init__(video, camera, show)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Creating TDDFA_v2 object for video {self.video.filename}:', timestamp))
        print('-'*120)
        print(self.video)
        import argparse
        import os.path as osp
        parser = argparse.ArgumentParser(description='The demo of video of 3DDFA_V2')
        self.current_path = osp.dirname(osp.abspath(__file__))

        # Dummy arguments to avoid ipykernel errors
        # parser.add_argument(
        #     "-i", "--ip", help="a dummy argument to fool ipython", default="1")
        # parser.add_argument(
        #     "-s", "--stdin", help="a dummy argument to fool ipython", default="1")
        # parser.add_argument(
        #     "-c", "--control", help="a dummy argument to fool ipython", default="1")
        # parser.add_argument(
        #     "-b", "--hb", help="a dummy argument to fool ipython", default="1")
        # parser.add_argument(
        #     "-K", "--Session.key", help="a dummy argument to fool ipython", default="1")
        # parser.add_argument(
        #     "-S", "--Session.signature_scheme", help="a dummy argument to fool ipython", default="1")
        # parser.add_argument(
        #     "-l", "--shell", help="a dummy argument to fool ipython", default="1")
        # parser.add_argument(
        #     "-t", "--transport", help="a dummy argument to fool ipython", default="1")
        # parser.add_argument(
        #     "-o", "--iopub", help="a dummy argument to fool ipython", default="1")
        
        # Actual arguments
        parser.add_argument('-c', '--config', type=str, default=f'{self.current_path}/TDDFA_v2/configs/mb1_120x120.yml')
        parser.add_argument('-f', '--video_fp', type=str, default=self.video.filename)
        parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
        if dense:
            parser.add_argument('-o', '--opt', type=str, default='dense', choices=['sparse', 'dense'])
        else:
            parser.add_argument('-o', '--opt', type=str, default='sparse', choices=['sparse', 'dense'])
        parser.add_argument('--onnx', action='store_true', default=False)
        if smooth:
            parser.add_argument('-n_pre', default=5, type=int, help='the pre frames of smoothing')
            parser.add_argument('-n_next', default=5, type=int, help='the next frames of smoothing')
            parser.add_argument('-s', '--start', default=-1, type=int, help='the started frames')
            parser.add_argument('-e', '--end', default=-1, type=int, help='the end frame')
            args = parser.parse_args(args=[])
            print(args)
            self.run_smooth(args)
        else:
            args = parser.parse_args(args=[])
            self.run(args)

        # offset values
        self.pose['yaw'] = [val - self.pose['yaw'][0] for val in self.pose['yaw']]
        self.pose['pitch'] = [val - self.pose['pitch'][0] for val in self.pose['pitch']]
        self.pose['roll'] = [val - self.pose['roll'][0] for val in self.pose['roll']]
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'3DDFA_v2 object complete!', timestamp))
        print('-'*120)
        
    
    def run(self, args):
        """Runs through 3DDFA_v2 tracking process, calling relevant
        functions
        
        More information found here:

        https://github.com/cleardusk/3DDFA_V2
        """
        # import EdiHeadyTrack.TDDFA_v2 as TDDFA_v2
        # from .TDDFA_v2.FaceBoxes import FaceBoxes
        # from .TDDFA_v2.TDDFA import TDDFA
    
        from .TDDFA_v2.utils.render import render
        from .TDDFA_v2.utils.pose import viz_pose, plot_pose_box
        from .TDDFA_v2.utils.functions import cv_draw_landmark, get_suffix
    
        import os 
        import imageio
        from tqdm import tqdm
        import yaml
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from .TDDFA_v2.TDDFA_ONNX import TDDFA_ONNX
        from .TDDFA_v2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX

        cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)

        # Given a video path
        fn = args.video_fp.split('/')[-1]
        reader = imageio.get_reader(args.video_fp)
        
        fps = reader.get_meta_data()['fps']

        suffix = get_suffix(args.video_fp)
        # video_wfp = f'{self.current_path}/TDDFA_v2/examples/results/videos/{fn.replace(suffix, "")}_{args.opt}.mp4'
        video_wfp = f'TDDFA_tracking.mp4'
        # writer = imageio.get_writer(video_wfp, fps=fps)
        writer = cv2.VideoWriter(video_wfp,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              self.video.fps,
                              (self.video.width,self.video.height))

        # run
        dense_flag = args.opt in ('dense',)
        pre_ver = None

        progress_bar = tqdm(range(self.video.total_frames))
        
        self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            i = int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES))
            success, frame_bgr = self.video.cap.read()
            if success:
                progress_bar.update(1)
                if i == 0:
                    # the first frame, detect face, here we only use the first face, you can change depending on your need
                    boxes = face_boxes(frame_bgr)
                    boxes = [boxes[0]]
                    param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                    # refine
                    param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                else:
                    param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

                    roi_box = roi_box_lst[0]
                    # todo: add confidence threshold to judge the tracking is failed
                    if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                        boxes = face_boxes(frame_bgr)
                        boxes = [boxes[0]]
                        param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                pre_ver = ver  # for tracking
                
                # Adding landmarks to face2d
                self.face2d['frame'].append(i)
                self.face2d['time'].append(i/self.video.fps)

                # print(list(ver[:-1][0]))
                x = list(ver[:-1][0])
                y = list(ver[:-1][1])
                top = [int(round(x[y.index(max(y))])), int(round(max(y)))]
                bottom = [int(round(x[y.index(min(y))])), int(round(min(y)))]
                left = [int(round(min(x))), int(round(y[x.index(min(x))]))]
                right = [int(round(max(x))), int(round(y[x.index(max(x))]))]
                landmark_positions = [top, bottom, left, right]
                self.face2d['all landmark positions'].append(landmark_positions)

                if args.opt == 'sparse':
                    res = cv_draw_landmark(frame_bgr, ver)
                    res, pose = viz_pose(res, param_lst, [ver]) 
                    self.pose['frame'].append(i)
                    self.pose['time'].append(i/self.video.fps)
                    self.pose['yaw'].append(pose[0]*-1)
                    self.pose['pitch'].append(pose[1])
                    self.pose['roll'].append(pose[2]*-1)   
                    if self.show == True:
                        cv2.namedWindow("EdiHeadyTrack", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("EdiHeadyTrack", int(self.video.width/2), int(self.video.height/2))
                        cv2.imshow('EdiHeadyTrack', res)
                    writer.write(res)
                    self.tracking_frames.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        # self.video.cap.release()
                        writer.release()
                        cv2.destroyAllWindows()
                        progress_bar.close()
                        print('Face tracking interrupted...')
                        break
                elif args.opt == 'dense':
                    res = render(frame_bgr, [ver], tddfa.tri, show_flag=False)
                    res, pose = viz_pose(res, param_lst, [ver]) 
                    self.pose['frame'].append(i)
                    self.pose['time'].append(i/self.video.fps)
                    self.pose['yaw'].append(pose[0]*-1)
                    self.pose['pitch'].append(pose[1])
                    self.pose['roll'].append(pose[2]*-1)   
                    if self.show == True:
                        cv2.namedWindow("EdiHeadyTrack", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("EdiHeadyTrack", int(self.video.width/2), int(self.video.height/2))
                        cv2.imshow('EdiHeadyTrack', res)
                    writer.write(res)
                    self.tracking_frames.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        # self.video.cap.release()
                        writer.release()
                        cv2.destroyAllWindows()
                        progress_bar.close()
                        print('Face tracking interrupted...')
                        break
                    
                else:
                    raise ValueError(f'Unknown opt {args.opt}')
            else:
                # self.video.cap.release()
                writer.release()
                cv2.destroyAllWindows()
                progress_bar.close()
                print('Face tracking complete...')
                break

            # time.sleep(0.5)
        
        print(f'Dump to {video_wfp}')


    def run_smooth(self, args):
        """Runs through 3DDFA_v2 SMOOTH tracking process, calling relevant
        functions

        'Smooth' looks n frames ahead to perform tracking. 
        
        More information found here:

        https://github.com/cleardusk/3DDFA_V2

        """
    
        from .TDDFA_v2.utils.render import render
        from .TDDFA_v2.utils.pose import viz_pose, plot_pose_box
        from .TDDFA_v2.utils.functions import cv_draw_landmark, get_suffix
        
        import os 
        import imageio
        from tqdm import tqdm
        import yaml
        from collections import deque
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from .TDDFA_v2.TDDFA_ONNX import TDDFA_ONNX
        from .TDDFA_v2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX

        cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)

        # Given a video path
        fn = args.video_fp.split('/')[-1]
        reader = imageio.get_reader(args.video_fp)

        fps = reader.get_meta_data()['fps']
        suffix = get_suffix(args.video_fp)
        video_wfp = f'TDDFA_tracking_smooth.mp4'
        writer = cv2.VideoWriter(video_wfp,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              self.video.fps,
                              (self.video.width,self.video.height))

        # the simple implementation of average smoothing by looking ahead by n_next frames
        # assert the frames of the video >= n
        n_pre, n_next = args.n_pre, args.n_next
        n = n_pre + n_next + 1
        queue_ver = deque()
        queue_frame = deque()

        # run
        dense_flag = args.opt in ('dense',)
        pre_ver = None

        progress_bar = tqdm(range(self.video.total_frames))
        
        self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            i = int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if args.start > 0 and i < args.start:
                continue
            if args.end > 0 and i > args.end:
                break

            success, frame_bgr = self.video.cap.read()
            
            if success:
                progress_bar.update(1)
                if i == 0:
                    # the first frame, detect face, here we only use the first face, you can change depending on your need
                    boxes = face_boxes(frame_bgr)
                    boxes = [boxes[0]]
                    param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                    # refine
                    param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                    # padding queue
                    for _ in range(n_pre):
                        queue_ver.append(ver.copy())
                    queue_ver.append(ver.copy())

                    for _ in range(n_pre):
                        queue_frame.append(frame_bgr.copy())
                    queue_frame.append(frame_bgr.copy())
                    
                else:
                    param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

                    roi_box = roi_box_lst[0]
                    # todo: add confidence threshold to judge the tracking is failed
                    if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                        boxes = face_boxes(frame_bgr)
                        boxes = [boxes[0]]
                        param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                    queue_ver.append(ver.copy())
                    queue_frame.append(frame_bgr.copy())
                    
                pre_ver = ver  # for tracking

                # Adding landmarks to face2d
                self.face2d['frame'].append(i)
                self.face2d['time'].append(i/self.video.fps)

                # print(list(ver[:-1][0]))
                x = list(ver[:-1][0])
                y = list(ver[:-1][1])
                top = [int(round(x[y.index(max(y))])), int(round(max(y)))]
                bottom = [int(round(x[y.index(min(y))])), int(round(min(y)))]
                left = [int(round(min(x))), int(round(y[x.index(min(x))]))]
                right = [int(round(max(x))), int(round(y[x.index(max(x))]))]
                landmark_positions = [top, bottom, left, right]
                self.face2d['all landmark positions'].append(landmark_positions)

                if len(queue_ver) >= n:
                    ver_ave = np.mean(queue_ver, axis=0)

                    if args.opt == 'sparse':
                        res = cv_draw_landmark(frame_bgr, ver)
                        res, pose = viz_pose(res, param_lst, [ver]) 
                        self.pose['frame'].append(i)
                        self.pose['time'].append(i/self.video.fps)
                        self.pose['yaw'].append(pose[0]*-1)
                        self.pose['pitch'].append(pose[1])
                        self.pose['roll'].append(pose[2]*-1)   
                        if self.show == True:
                            cv2.namedWindow("EdiHeadyTrack", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("EdiHeadyTrack", int(self.video.width/2), int(self.video.height/2))
                            cv2.imshow('EdiHeadyTrack', res)
                        writer.write(res)
                        self.tracking_frames.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            # self.video.cap.release()
                            writer.release()
                            cv2.destroyAllWindows()
                            progress_bar.close()
                            print('Face tracking interrupted...')
                            break
                    elif args.opt == 'dense':
                        res = render(frame_bgr, [ver], tddfa.tri, show_flag=False)
                        res, pose = viz_pose(res, param_lst, [ver]) 
                        self.pose['frame'].append(i)
                        self.pose['time'].append(i/self.video.fps)
                        self.pose['yaw'].append(pose[0]*-1)
                        self.pose['pitch'].append(pose[1])
                        self.pose['roll'].append(pose[2]*-1)   
                        if self.show == True:
                            cv2.namedWindow("EdiHeadyTrack", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("EdiHeadyTrack", int(self.video.width/2), int(self.video.height/2))
                            cv2.imshow('EdiHeadyTrack', res)
                        writer.write(res)
                        self.tracking_frames.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            # self.video.cap.release()
                            writer.release()
                            cv2.destroyAllWindows()
                            progress_bar.close()
                            print('Face tracking interrupted...')
                            break    
                    else:
                        raise ValueError(f'Unknown opt {args.opt}')

            
            else:
                # self.video.cap.release()
                writer.release()
                cv2.destroyAllWindows()
                progress_bar.close()
                print('Face tracking complete...')
                break
        
        print(n_next)
        
        for _ in range(n_next):
            queue_ver.append(ver.copy())
            queue_frame.append(res.copy())  # the last frame

            ver_ave = np.mean(queue_ver, axis=0)

            if args.opt == 'sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif args.opt == 'dense':
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            else:
                raise ValueError(f'Unknown opt {args.opt}')


            queue_ver.popleft()
            queue_frame.popleft()
        
        
        print(f'Dump to {video_wfp}')
        