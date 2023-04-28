# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    camera.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/25 15:41:23 by taston            #+#    #+#              #
#    Updated: 2023/04/27 13:45:55 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from .video import Video
from .calibration import Calibrator

class Camera:
    def __init__(self):
        width = 1920
        height = 1080
        self.focal_length = height * 1.28
        self.internal_matrix = np.array([[self.focal_length, 0, width/2],
                                          [0, self.focal_length, height/2],
                                          [0, 0, 1]])
        self.distortion_matrix = np.zeros((4, 1), dtype=np.float64)
        self.calibrated = False

    def calibrate(self, checkerboard=(9,6), video=Video()):
        '''
        Calibrate camera object using chosen video and set calibrated state to True.
        '''
        self.video = video
        print(video)
        self.calibrator = Calibrator(checkerboard, self.video)
        self.calibrated = True