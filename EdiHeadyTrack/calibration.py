# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    calibration.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/25 10:20:08 by taston            #+#    #+#              #
#    Updated: 2023/05/10 10:03:59 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2 
import numpy as np
from datetime import datetime
from .video import Video

class Checkerboard:
    """
    A class used to represent a calibration Checkerboard

    ...
    
    Attributes
    ----------
    dimensions : tuple
        tuple of checkerboard pattern dimensions
    min_points : int
        integer threshold of minimum detected points for 
        checkerboard to be considered found
    objectp3d : ndarray
        array of checkerboard points in three dimensions
    threedpoints : list
        list of checkerboard points in three dimensions
        for each frame where a checkerboard is found
    twodpoints : list 
        list of detected checkerboard points in two
        dimensions for each frame
        
    Methods
    -------
    get_corners(gray_frame)
        Finds checkerboard corners in a given grayscale frame
    """
    def __init__(self, dimensions = (9,6)):
        """
        Parameters
        ----------
        dimensions : tuple, optional
            Checkerboard pattern used in camera calibration (default is 9x6)
        """
        
        print('Checkerboard created')
        self.dimensions = dimensions
        self.min_points = 50
        self.twodpoints = [] 
        self.threedpoints = [] 
        self.objectp3d = np.zeros((1, self.dimensions[0]
                            * self.dimensions[1],
                            3), np.float32)
        self.objectp3d[0, :, :2] = np.mgrid[0:self.dimensions[0],
                                    0:self.dimensions[1]].T.reshape(-1, 2)
        
    def get_corners(self, gray_frame):
        """
        Looks for checkerboard corners in a given grayscale video
        frame.

        Parameters
        ----------
        gray_frame : ndarray
            ndarray representing grayscale frame from video

        Returns
        -------
        ret : bool
            bool representing if corner search was successful
        corners : ndarray
            ndarray containing coordinates of corners
        """

        ret, corners = cv2.findChessboardCorners(
                            gray_frame, self.dimensions,
                            cv2.CALIB_CB_ADAPTIVE_THRESH
                            + cv2.CALIB_CB_FAST_CHECK +
                            cv2.CALIB_CB_NORMALIZE_IMAGE)

        return ret, corners

class Calibrator:
    """
    A class used to represent a camera Calibrator

    ...
    
    Attributes
    ----------
    checkerboard : Checkerboard

    criteria : tuple
        tuple of criteria for successful camera calibration
    distortion : ndarray
        ndarray of distortion parameters
    frame : ndarray
        ndarray representing video frame
    gray_frame : ndarray
        ndarray representing grayscale video frame
    matrix : ndarray
        ndarray representing camera intrinsic matrix
    r_vecs : ndarray
        ndarray of rotational vectors
    t_vecs : ndarray
        ndarray of translation vectors
        
    Methods
    -------
    calibrate()
        Perform camera calibration process
    draw_corners(corners)
        Draw checkerboard corners on video frame
    save_outputs()
        Save camera parameters to csv files
    """

    def __init__(self, checkerboard, video=Video()):
        """
        Parameters
        ----------
        checkerboard : tuple
            tuple representing Checkerboard pattern
        video : Video, optional
            Video used for camera calibration. If no video specified
            an empty video will be attempted.
        """
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
        self.calibrate()
        # self.save_outputs()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('-'*120)
        print('{:<100} {:>19}'.format(f'Calibrator object complete!', timestamp))
        print('-'*120)

    def calibrate(self):
        """
        Performs the camera calibration procedure outlined here:

        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        """
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
        ret, self.matrix, self.distortion, self.r_vecs, self.t_vecs = cv2.calibrateCamera(self.checkerboard.threedpoints, 
                                                                      self.checkerboard.twodpoints,
                                                                      self.gray_frame.shape[::-1], None, None)
        print(f'Number of frames used for calibration: {frame_number}')
    
    def draw_corners(self, corners):
        '''
        Draws corners of checkerboard onto frame to verify calibration is working

        Parameters
        ----------
        corners : ndarray
            ndarray of the corners found for a given frame

        Returns
        -------
        complete : bool
            bool representing whether search for corners is complete
        frame : ndarray
            new video frame with corners drawn
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
        """
        Saves matrices to csv
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print('Saving outputs...')
        from numpy import savetxt
        savetxt('EdiHeadyTrack/resources/camera_matrix.csv', self.matrix, delimiter=',')
        savetxt('EdiHeadyTrack/resources/camera_distortion.csv', self.distortion, delimiter=',')