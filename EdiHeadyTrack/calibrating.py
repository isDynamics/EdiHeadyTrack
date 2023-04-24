# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    calibrating.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/17 14:32:56 by taston            #+#    #+#              #
#    Updated: 2023/04/24 11:16:16 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import numpy as np
from time import sleep
from datetime import datetime

def calibrate(vid_file, show=True, record=False):
    '''
    Obtain the intrinsic camera parameters for the camera used
    in a specified input video file. 

    Method follows: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    '''
    timestamp = datetime.now().strftime("%H:%M:%S")
    print('-'*45)
    print('{:<30} {:>14}'.format('Camera calibration started:', timestamp))
    print('-'*45)
    min_points = 50
    # Initialise criteria, vectors and matrices
    checkerboard = (9,6)
    criteria, threedpoints, twodpoints, objectp3d = initialise(checkerboard)
    # Open video file
    cap, FPS = open_vid(vid_file)
    # Set up writer object for recording
    width, height, writer = setup_recording(cap, FPS)
    # Perform calibration on each frame
    matrix, distortion, r_vecs, t_vecs = calibrate_frames(criteria,
                                                        min_points,
                                                        cap, 
                                                        checkerboard, 
                                                        twodpoints, 
                                                        threedpoints, 
                                                        objectp3d,
                                                        record,
                                                        writer,
                                                        show)
    # Save matrices to csv files
    save_outputs(matrix, distortion)

    timestamp = datetime.now().strftime("%H:%M:%S")
    print('-'*45)
    print('{:<30} {:>14}'.format('Camera calibration complete:', timestamp))
    print('-'*45)
    # return f'{vid_file} calibration success'

def initialise(checkerboard):
    '''
    Initialise parameters required for calibration process.
    '''
    timestamp = datetime.now().strftime("%H:%M:%S")
    print('{:<30} {:>14}'.format('Initialising parameters...', timestamp))
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
    '''
    Open a specified video file and return capture object and frames
    per second.
    '''
    timestamp = datetime.now().strftime("%H:%M:%S")
    print('{:<30} {:>14}'.format('Opening calibration video...', timestamp))
    cap = cv2.VideoCapture(vid_file)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        timestamp = datetime.now().strftime("%H:%M:%S")
        raise IOError("Cannot open chosen video")

    return cap, FPS

def setup_recording(cap, FPS):
    '''
    Create video writer object if record set to true
    '''
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter('calibration.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            FPS,
                            (width,height))
    
    return width, height, writer

def calibrate_frames(criteria,
                     min_points,
                     cap, 
                     checkerboard, 
                     twodpoints, 
                     threedpoints, 
                     objectp3d,
                     record,
                     writer,
                     show):
    '''
    Perform calibration process frame by frame, searching for checkerboard and 
    each individual frame and ending the process once calibration is complete.
    '''
    timestamp = datetime.now().strftime("%H:%M:%S")
    print('{:<30} {:>14}'.format('Performing calibration...', timestamp))
    while True:
            ret, img = cap.read()
            image = img
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners (ret=True if corners found)
            ret, corners = cv2.findChessboardCorners(
                            grayColor, checkerboard,
                            cv2.CALIB_CB_ADAPTIVE_THRESH
                            + cv2.CALIB_CB_FAST_CHECK +
                            cv2.CALIB_CB_NORMALIZE_IMAGE)
    
            # If corners found, display them on image of the checkerboard.
            complete, image = draw_corners(image,
                                           ret, 
                                            threedpoints,
                                            objectp3d,
                                            grayColor,
                                            corners,
                                            checkerboard,
                                            criteria,
                                            twodpoints,
                                            min_points,
                                            cap,
                                            record,
                                            writer)
            if complete: break
            # Show frame and write to writer 
            if show == True:
                cv2.imshow('img', image)
            if record:
                writer.write(image)
            
            # wait for ESC key to exit and terminate feed.
            k = cv2.waitKey(1)
            if k == 27:         
                cap.release()
                if record: writer.release()
                cv2.destroyAllWindows()
                break

    h, w = image.shape[:2] 
    
    # Perform camera calibration by given threedpoints and twodpoints
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    
    return matrix, distortion, r_vecs, t_vecs

def draw_corners(image,
                 ret, 
                 threedpoints,
                 objectp3d,
                 grayColor,
                 corners,
                 checkerboard,
                 criteria,
                 twodpoints,
                 min_points,
                 cap,
                 record,
                 writer):
    complete = False
    if ret == True:
        threedpoints.append(objectp3d)
        # Refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(
            grayColor, corners, checkerboard, (-1, -1), criteria)
        twodpoints.append(corners2)
        # When we have minimum number of data points, stop:
        if len(twodpoints) > min_points:
            cap.release()
            if record: writer.release()
            cv2.destroyAllWindows()
            complete=True

        # Draw and display the corners:
        image = cv2.drawChessboardCorners(image,
                                        checkerboard,
                                        corners2, ret)
        
    return complete, image
        
def save_outputs(matrix, distortion):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print('{:<30} {:>14}'.format('Saving outputs...', timestamp))
    from numpy import savetxt
    savetxt('EdiHeadyTrack/resources/camera_matrix.csv', matrix, delimiter=',')
    savetxt('EdiHeadyTrack/resources/camera_distortion.csv', distortion, delimiter=',')