# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    processing.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/03/23 14:44:29 by taston            #+#    #+#              #
#    Updated: 2023/04/05 13:56:15 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import genfromtxt

def process(filtered_data, show=False):
    '''
    Process the filtered data to add data for velocities and accelerations
    '''
    
    data = genfromtxt(filtered_data, delimiter=',')
    time = data[1:, 0]
    yaw = data[1:, 1]
    pitch = data[1:, 2]
    roll = data[1:, 3]

    omega_yaw = np.insert(differentiate(yaw, time), 0, None)
    omega_pitch = np.insert(differentiate(pitch, time), 0, None)
    omega_roll = np.insert(differentiate(roll, time), 0, None)

    plt.plot(time, omega_yaw, label='omega_yaw')
    plt.plot(time, omega_pitch, label='omega_pitch')
    plt.plot(time, omega_roll, label='omega_roll')
    plt.legend(loc='upper right')
    plt.savefig('data/plot images/processed.pdf')

    df = pd.read_csv(filtered_data)
    df['omega_yaw'] = omega_yaw
    df['omega_pitch'] = omega_pitch
    df['omega_roll'] = omega_roll

    df.to_csv('data/tracking data/output_final.csv', index=False)

    if show == True:
        plt.show()


def differentiate(property, time):
    '''
    Differentiate a given property with respect to time
    '''
    return np.diff(property)/np.diff(time)