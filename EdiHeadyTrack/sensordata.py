# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sensordata.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/25 15:35:24 by taston            #+#    #+#              #
#    Updated: 2023/05/10 11:51:39 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


class SensorData:
    """
    A class representing SensorData

    Attributes
    ----------
    velocity : dict
        time history of rotational velocities
    acceleration : dict
        time history of rotational accelerations
    """
    def __init__(self):
        self.velocity = {'time':    [],
                         'yaw':     [],
                         'pitch':   [],
                         'roll':    []}
        self.acceleration = {'time':    [],
                             'yaw':     [],
                             'pitch':   [],
                             'roll':    []}