# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_utils.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/09 12:54:56 by taston            #+#    #+#              #
#    Updated: 2023/04/19 14:32:15 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os.path


def plot():
    def animate(i):
        '''
        Function which reads live .csv file to be used by FuncAnimation
        '''
        import time
        while not os.path.exists("data/tracking data/output.csv"):
            time.sleep(1)

        if os.path.isfile("data/tracking data/output.csv"):
            data = pd.read_csv("data/tracking data/output.csv")
        else:
            raise ValueError("%s isn't a file!" % "output.csv")
        
        time = data["time"]
        x = data["x"]
        y = data["y"]
        x_vel = data["x_vel"]
        y_vel = data["y_vel"]
        x_acc = data["x_acc"]
        y_acc = data["y_acc"]
        yaw = data["yaw"]
        pitch = data["pitch"]
        roll = data["roll"]
        yaw_vel = data["yaw_vel"]
        pitch_vel = data["pitch_vel"]
        roll_vel = data["roll_vel"]
        yaw_acc = data["yaw_acc"]
        pitch_acc = data["pitch_acc"]
        roll_acc = data["roll_acc"]

        # TODO: face numbering
        plt.subplot(3,2,1)
        plt.cla()
        plt.plot(time, x, label="x position")
        plt.plot(time, y, label = "y position")
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.subplot(3,2,3)
        plt.cla()
        plt.plot(time, x_vel, label="x velocity")
        plt.plot(time, y_vel, label="y velocity")
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.subplot(3,2,5)
        plt.cla()
        plt.plot(time, x_acc, label="x acceleration")
        plt.plot(time, y_acc, label="y acceleration")
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.subplot(3,2,2)
        plt.cla()
        plt.plot(time, yaw, label="yaw")
        plt.plot(time, pitch, label="pitch")
        plt.plot(time, roll, label="roll")
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.subplot(3,2,4)
        plt.cla()
        plt.plot(time, yaw_vel, label="yaw vel")
        plt.plot(time, pitch_vel, label="pitch vel")
        plt.plot(time, roll_vel, label="roll vel")
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.subplot(3,2,6)
        plt.cla()
        plt.plot(time, yaw_acc, label="yaw acc")
        plt.plot(time, pitch_acc, label="pitch acc")
        plt.plot(time, roll_acc, label="roll acc")
        plt.legend(loc="upper left")
        plt.tight_layout()
        

        
            
    ani = FuncAnimation(plt.gcf(), animate, interval = 10)
    plt.gcf().canvas.manager.set_window_title('EdiHeadyTrack - Kinematics')
    # plt.savefig('plot images/raw.pdf')
    plt.show()