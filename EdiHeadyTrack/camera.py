# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    camera.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/25 15:41:23 by taston            #+#    #+#              #
#    Updated: 2023/05/10 09:39:30 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from .video import Video
from .calibration import Calibrator

class Camera:
    """
    A class used to represent a Camera.

    ...

    Attributes
    ----------
    focal_length : float
        float representing the focal length of the Camera in mm
    internal_matrix : ndarray
        array representing the Camera's intrinsic parameters
    distortion_matrix : ndarray
        array representing the Camera's lens distortion parameters
    calibrator : Calibrator
        Calibrator object used for camera calibration
    calibrated : bool
        bool for quick checking if camera has been calibrated
    video : Video
        Video object where the footage has been shot using this Camera
        
    Methods
    -------
    calibrate(checkerboard=(9,6), video=Video()):
        Performs calibration on the camera
    """
    def __init__(self):
        """
        Parameters
        ----------
        ...
        """

        width = 1920
        height = 1080
        self.focal_length = height * 1.28
        self.internal_matrix = np.array([[self.focal_length, 0, width/2],
                                          [0, self.focal_length, height/2],
                                          [0, 0, 1]])
        self.distortion_matrix = np.zeros((4, 1), dtype=np.float64)
        self.calibrated = False

    def calibrate(self, checkerboard=(9,6), video=Video()):
        """Creates a calibrator object and calibrates the Camera.

        If arguments checkerboard and video aren't passed in, the
        default checkerboard pattern and an empty video are used.

        Parameters
        ----------
        checkerboard : tuple, optional
            Checkerboard pattern used in camera calibration (default is 9x6)
        video : Video, optional
            Video used to calibrate camera
        """
        
        self.video = video
        print(video)
        self.calibrator = Calibrator(checkerboard, self.video)
        self.calibrated = True