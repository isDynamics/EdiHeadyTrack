# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tracking.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/13 13:55:50 by taston            #+#    #+#              #
#    Updated: 2023/04/28 08:53:38 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import time
import numpy as np
import cv2
import EdiHeadyTrack.math_utils as math_utils
# from EdiHeadyTrack.facedetector import FaceMeshDetector
from tqdm import tqdm


def track(vidfile, maxFaces, show=False):
    print('-'*45)
    print('Tracking started')
    print('-'*45)
    
    cap = cv2.VideoCapture(vidfile)
    # Set up progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # progress_bar = iter(tqdm(range(total_frames), f"Tracking video {vidfile}"))
    progress_bar = tqdm(range(total_frames), f"Tracking video {vidfile}")
    # Initialise time count
    sTime = time.time()
    pTime = 0
    # Initialise detector object
    detector = FaceMeshDetector(maxFaces=maxFaces, minDetectionCon=0.5)
    kinematics = math_utils.Kinematics(maxFaces, sTime=sTime)

    # While video open
    while True:
        # Read frame
        success, img = cap.read()
        if success:
            # next(progress_bar)
            progress_bar.update(1)
            # Find face in current frame and calculate kinematics
            img, face, faces, p1, p2 = detector.findFaceMesh(img)
            current_frame = cap.get(1)
            kinematics.calculate(current_frame, faces) 

            # Calculate fps
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.line(img, p1, p2, (255,0,0), 3)
            
            if show == True:            
                
                cv2.namedWindow("EdiHeadyTrack", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("EdiHeadyTrack", 640, 360)
                cv2.imshow("EdiHeadyTrack", img)
                
            cv2.imwrite(f'tracking frames/{current_frame}.png', img)
            # Terminate with q
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break     
        else:
            break
    
    print('#'*20)
    print('Tracking complete')
    print('#'*20)

    cap.release()
    cv2.destroyAllWindows()