# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    math_utils.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/10 15:23:30 by taston            #+#    #+#              #
#    Updated: 2023/04/05 13:52:02 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import time
from pathlib import Path
import os

class Kinematics():
    def __init__(self, maxFaces, sTime):
        self.maxFaces = maxFaces
        self.fieldnames = ["frame", "time", "x", "y", 
                           "x_vel", "y_vel", 
                           "x_acc", "y_acc", 
                           "yaw", "pitch", "roll",
                           "yaw_vel", "pitch_vel", "roll_vel",
                           "yaw_acc", "pitch_acc", "roll_acc"]
        
        # for idx in range(maxFaces):
        #     # If file already exists, delete it so that fresh file can be written to
        #     if Path(f"face{idx}.csv").is_file():
        #         os.remove(f"face{idx}.csv")                
                 
        #     with open(f"face{idx}.csv", "w") as csv_file:
        #         csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames) 
        #         csv_writer.writeheader()
        
        if Path("data/tracking data/output.csv").is_file():
            os.remove("data/tracking data/output.csv")
        with open(f"data/tracking data/output.csv", "w") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames) 
            csv_writer.writeheader()

        self.info = {
            "frame":        None,
            "time":         None,
            "x":            None,
            "y":            None,
            "x_vel":        None,
            "y_vel":        None,
            "x_acc":        None,
            "y_acc":        None,
            "yaw":          None,
            "pitch":        None,
            "roll":         None,
            "yaw_vel":      None,
            "pitch_vel":    None,
            "roll_vel":     None,
            "yaw_acc":      None,
            "pitch_acc":    None,
            "roll_acc":     None
        }

        self.sTime = sTime
        self.pTime = 0    
        self.frame = 0
        self.x_prev = None
        self.y_prev = None
        self.x_vel_prev = None
        self.y_vel_prev = None
        self.yaw_prev = None
        self.roll_prev = None
        self.pitch_prev = None
        self.yaw_vel_prev = None
        self.pitch_vel_prev = None
        self.roll_vel_prev = None

    def calculate(self, frame, faces):
        '''
        Calculate head kinematics from input information from FaceMeshDetector
        '''
        cTime = time.time()
        sTime = self.sTime

        # If faces detected, look for kinematics of each face
        if faces:
            for idx, face in enumerate(faces):    
                # with open(f"face{idx}.csv", "a") as csv_file:
                with open("data/tracking data/output.csv", "a") as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames) 
                    
                    self.info["time"] = frame/240
                    self.info["frame"] = frame         
                    self.info["x"] = face["x_av"]
                    self.info["y"] = face["y_av"]
                    self.info["yaw"] = face["yaw"]
                    self.info["pitch"] = face["pitch"]
                    self.info["roll"] = face["roll"]

                    # If face detected in previous frame, calculate time dependent kinematics
                    if self.x_prev and self.info["x"]:                   
                        # Linear velocities  
                        self.info["x_vel"] = (self.info["x"] - self.x_prev)/(cTime - self.pTime)
                        self.info["y_vel"] = (self.info["y"] - self.y_prev)/(cTime - self.pTime)
                        # Angular velocities
                        self.info["yaw_vel"] = (self.info["yaw"] - self.yaw_prev)/(cTime - self.pTime)
                        self.info["pitch_vel"] = (self.info["pitch"] - self.pitch_prev)/(cTime - self.pTime)
                        self.info["roll_vel"] = (self.info["roll"] - self.roll_prev)/(cTime - self.pTime)

                        if self.info["x_vel"] and self.x_vel_prev:
                            # Linear accelerations
                            self.info["x_acc"] = (self.info["x_vel"] - self.x_vel_prev)/(cTime - self.pTime)
                            self.info["y_acc"] = (self.info["y_vel"] - self.y_vel_prev)/(cTime - self.pTime)
                            # Angular accelerations
                            self.info["yaw_acc"] = (self.info["yaw_vel"] - self.yaw_vel_prev)/(cTime - self.pTime) 
                            self.info["pitch_acc"] = (self.info["pitch_vel"] - self.yaw_vel_prev)/(cTime - self.pTime) 
                            self.info["roll_acc"] = (self.info["roll_vel"] - self.yaw_vel_prev)/(cTime - self.pTime) 
                        # else:
                            # Linear accelerations
                            # self.info["x_acc"] = None
                            # self.info["y_acc"] = None
                            # # Angular accelerations
                            # self.info["yaw_acc"] = None
                            # self.info["pitch_acc"] = None
                            # self.info["roll_acc"] = None
                    # else:
                    #     # Linear
                    #     self.info["x_vel"] = None
                    #     self.info["y_vel"] = None 
                    #     self.info["x_acc"] = None
                    #     self.info["y_acc"] = None
                    #     # Angular
                    #     self.info["yaw_vel"] = None
                    #     self.info["pitch_vel"] = None
                    #     self.info["roll_vel"] = None
                    #     self.info["yaw_acc"] = None
                    #     self.info["pitch_acc"] = None
                    #     self.info["roll_acc"] = None
                    
                    
                    # Linear
                    self.x_prev = self.info["x"]
                    self.y_prev = self.info["y"]
                    self.x_vel_prev = self.info["x_vel"]
                    self.y_vel_prev = self.info["y_vel"]
                    # Angular
                    self.yaw_prev = self.info["yaw"]
                    self.pitch_prev = self.info["pitch"]
                    self.roll_prev = self.info["roll"]
                    self.yaw_vel_prev = self.info["yaw_vel"]
                    self.pitch_vel_prev = self.info["pitch_vel"]
                    self.roll_vel_prev = self.info["roll_vel"]

                    # Write to file
                    csv_writer.writerow(self.info)
                    self.pTime = cTime
        # If no faces, just write None entries for kinematics
        else:
            for idx in range(self.maxFaces):
                # with open(f"face{idx}.csv", "a") as csv_file:
                with open("data/tracking data/output.csv", "a") as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames) 
                    self.info["time"] = frame/240
                    self.info["frame"] = frame
                    # self.info["x"] = None
                    # self.info["y"] = None
                    # self.info["yaw"] = None
                    # self.info["pitch"] = None
                    # self.info["roll"] = None
                    # self.info["x_vel"] = None
                    # self.info["y_vel"] = None 
                    # self.info["x_acc"] = None
                    # self.info["y_acc"] = None
                    # self.info["yaw_vel"] = None
                    # self.info["pitch_vel"] = None
                    # self.info["roll_vel"] = None
                    # self.info["yaw_acc"] = None
                    # self.info["pitch_acc"] = None
                    # self.info["roll_acc"] = None
                    
                    self.x_prev = self.info["x"]
                    self.y_prev = self.info["y"]
                    self.x_vel_prev = self.info["x_vel"]
                    self.y_vel_prev = self.info["y_vel"]

                    self.yaw_prev = self.info["yaw"]
                    self.pitch_prev = self.info["pitch"]
                    self.roll_prev = self.info["roll"]
                    self.yaw_vel_prev = self.info["yaw_vel"]
                    self.pitch_vel_prev = self.info["pitch_vel"]
                    self.roll_vel_prev = self.info["roll_vel"]

                    csv_writer.writerow(self.info)
                    self.pTime = cTime

        return self.info