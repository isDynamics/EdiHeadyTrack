# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    headkinematics.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/26 15:55:54 by taston            #+#    #+#              #
#    Updated: 2023/04/28 11:47:39 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from .sensordata import SensorData
from .head import Head

class HeadKinematics(SensorData):
    def __init__(self, head=Head):
        super().__init__()
        self.head = head