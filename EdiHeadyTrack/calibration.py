# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    calibration.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/25 10:20:08 by taston            #+#    #+#              #
#    Updated: 2023/04/28 11:33:14 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2 
import numpy as np
from datetime import datetime
from .video import Video

class Checkerboard:
    def __init__(self, dimensions = (9,6)):
        print('Checkerboard created')
        self.dimensions = dimensions
        self.min_points = 50
        self.threedpoints = []
        self.twodpoints = [] 
        self.objectp3d = np.zeros((1, self.dimensions[0]
                            * self.dimensions[1],
                            3), np.float32)
        self.objectp3d[0, :, :2] = np.mgrid[0:self.dimensions[0],
                                    0:self.dimensions[1]].T.reshape(-1, 2)
        
    def get_corners(self, gray_frame):
        ret, corners = cv2.findChessboardCorners(
                            gray_frame, self.dimensions,
                            cv2.CALIB_CB_ADAPTIVE_THRESH
                            + cv2.CALIB_CB_FAST_CHECK +
                            cv2.CALIB_CB_NORMALIZE_IMAGE)

        return ret, corners

class Calibrator:
    def __init__(self, checkerboard, video=Video()):
        self.video = video
        self.checkerboard = Checkerboard(checkerboard)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Creating Calibrator object for video {self.video.filename}:', timestamp))
        print('-'*120)
        print(self.video)
        print(f'Checkerboard dimensions: {self.checkerboard.dimensions[0]} x {self.checkerboard.dimensions[1]}')
        self.criteria = (cv2.TERM_CRITERIA_EPS + 
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.matrix, self.distortion, self.r_vecs, self.t_vecs = self.calibrate()
        self.save_outputs()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Calibrator object complete!', timestamp))
        print('-'*120)

    def calibrate(self):
        '''
        Obtain the camera intrinsic parameters for the chosen video.
        '''
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.video.create_writer()
        print('Displaying video...')
        while True:
            ret, self.frame = self.video.cap.read()
            frame_number = int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            ret, corners = self.checkerboard.get_corners(self.gray_frame)
            if ret:
                complete, image = self.draw_corners(corners)
            if complete: break
            cv2.imshow('Calibrating...', self.frame)
            self.video.writer.write(self.frame)
            k = cv2.waitKey(1)
            if k == 27:
                self.video.cap.release()
                self.video.writer.release()
                cv2.destroyAllWindows()
                break
        h, w = image.shape[:2] 
    
        # Perform camera calibration by given threedpoints and twodpoints
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(self.checkerboard.threedpoints, 
                                                                      self.checkerboard.twodpoints,
                                                                      self.gray_frame.shape[::-1], None, None)
        print(f'Number of frames used for calibration: {frame_number}')
        return matrix, distortion, r_vecs, t_vecs
    
    def draw_corners(self, corners):
        '''
        Draw corners of checkerboard onto frame to verify calibration is working
        '''
        complete = False
        
        self.checkerboard.threedpoints.append(self.checkerboard.objectp3d)
        # Refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(
            self.gray_frame, corners, self.checkerboard.dimensions, (-1, -1), self.criteria)
        self.checkerboard.twodpoints.append(corners2)
        # When we have minimum number of data points, stop:
        if len(self.checkerboard.twodpoints) > self.checkerboard.min_points:
            self.video.cap.release()
            self.video.writer.release()
            cv2.destroyAllWindows()
            complete=True

        # Draw and display the corners:
        frame = cv2.drawChessboardCorners(self.frame,
                                        self.checkerboard.dimensions,
                                        corners2, True)
            
        return complete, frame
    
    def save_outputs(self):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('Saving outputs...')
        from numpy import savetxt
        savetxt('EdiHeadyTrack/resources/camera_matrix.csv', self.matrix, delimiter=',')
        savetxt('EdiHeadyTrack/resources/camera_distortion.csv', self.distortion, delimiter=',')