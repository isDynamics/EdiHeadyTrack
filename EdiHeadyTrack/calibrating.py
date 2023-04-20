# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    calibrating.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/17 14:32:56 by taston            #+#    #+#              #
#    Updated: 2023/04/20 14:41:05 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import EdiHeadyTrack
import cv2
import numpy as np
from time import sleep

# Define the dimensions of checkerboard
CHECKERBOARD = (9 ,6)
MIN_POINTS = 50
RECORD = True


def calibrate(vid_file, show=False):
    '''
    Obtain the intrinsic camera parameters for the camera used
    in a specified input video file. 

    Method: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    '''
    min_points = 50
    # Initialise criteria, vectors and matrices
    checkerboard = (9,6)
    criteria, threedpoints, twodpoints, objectp3d = initialise(checkerboard)
    print(objectp3d)
    # Open video file
    cap, FPS = open_vid(vid_file)
        
    if RECORD:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter('calibration.mp4',
                                    cv2.VideoWriter_fourcc(*'DIVX'),
                                    FPS,
                                    (width,height))
       
    while True:
            ret, img = cap.read()
            image = img
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
            # Find the chess board corners
            # if desired number of corners are
            # found in the image then ret = true:
            ret, corners = cv2.findChessboardCorners(
                            grayColor, CHECKERBOARD,
                            cv2.CALIB_CB_ADAPTIVE_THRESH
                            + cv2.CALIB_CB_FAST_CHECK +
                            cv2.CALIB_CB_NORMALIZE_IMAGE)
    
            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:
                threedpoints.append(objectp3d)
    
                # Refining pixel coordinates
                # for given 2d points.
                corners2 = cv2.cornerSubPix(
                    grayColor, corners, CHECKERBOARD, (-1, -1), criteria)

                twodpoints.append(corners2)
                # When we have minimum number of data points, stop:
                if len(twodpoints) > MIN_POINTS:
                    cap.release()
                    if RECORD: writer.release()
                    cv2.destroyAllWindows()
                    break
    
                # Draw and display the corners:
                image = cv2.drawChessboardCorners(image,
                                                CHECKERBOARD,
                                                corners2, ret)

            if show == True:
                cv2.imshow('img', image)
            
            if RECORD:
                writer.write(image)
            
            # wait for ESC key to exit and terminate feed.
            k = cv2.waitKey(1)
            if k == 27:         
                cap.release()
                if RECORD: writer.release()
                cv2.destroyAllWindows()
                break


    h, w = image.shape[:2] 
    
    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints):
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None) 
    
    # Displaying required output
    print("Camera matrix:")
    print(matrix)
    
    print("\n Distortion coefficient:")
    print(distortion)
    
    print("\n Rotation Vectors:")
    print(r_vecs)
    
    print("\n Translation Vectors:")
    print(t_vecs)

    from numpy import savetxt
    from numpy import genfromtxt
    

    mean_r = np.mean(np.asarray(r_vecs), axis=0)
    mean_t = np.mean(np.asarray(t_vecs), axis=0)
    savetxt('data/calibration/rotation_vectors.csv', mean_r, delimiter=',')
    savetxt('data/calibration/translation_vectors.csv', mean_t, delimiter=',')
    savetxt('data/calibration/camera_matrix.csv', matrix, delimiter=',')
    savetxt('data/calibration/camera_distortion.csv', distortion, delimiter=',')


    # return f'{vid_file} calibration success'

def initialise(checkerboard):
    # Stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Vector for the 3D points:
    threedpoints = []
    
    # Vector for 2D points:
    twodpoints = [] 
    
    # 3D points real world coordinates:
    objectp3d = np.zeros((1, checkerboard[0]
                        * checkerboard[1],
                        3), np.float32)

    objectp3d[0, :, :2] = np.mgrid[0:checkerboard[0],
                                0:checkerboard[1]].T.reshape(-1, 2)
    
    return criteria, threedpoints, twodpoints, objectp3d

def open_vid(vid_file):
    cap = cv2.VideoCapture(vid_file)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open chosen video")

    return cap, FPS