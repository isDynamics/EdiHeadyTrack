# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/27 10:18:49 by taston            #+#    #+#              #
#    Updated: 2023/04/28 11:42:29 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


from .sensordata import SensorData
from .head import Head
import matplotlib.pyplot as plt

class Plot:
    def __init__(self):
        ...

    def plot_head(self, head=Head()):
        self.head = head
        # self.head.pose
        plt.plot(self.head.pose['time'], self.head.pose['yaw'])
        plt.show()

    def plot_kinematics(self, kinematics=SensorData()):
        self.kinematics = kinematics
        