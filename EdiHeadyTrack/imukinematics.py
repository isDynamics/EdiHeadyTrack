# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    imukinematics.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/26 16:03:23 by taston            #+#    #+#              #
#    Updated: 2023/04/26 16:05:47 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from .sensordata import SensorData

class IMUKinematics(SensorData):
    def __init__(self):
        super().__init__()