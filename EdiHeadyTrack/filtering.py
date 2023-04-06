# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    filtering.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/03/22 13:33:00 by taston            #+#    #+#              #
#    Updated: 2023/04/06 15:04:19 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

def filter(file, show=False):
    '''
    Apply selected filter to data from a specified input file
    '''
    data = pd.read_csv(file)
    filtered_data = []
    # Properties to be filtered
    properties = ['yaw', 'pitch', 'roll']
    # Corresponding colours to be used in plots
    plot_colours = [['b', 'darkblue'], 
                    ['g', 'darkgreen'],
                    ['r', 'darkred']]

    for idx, property in enumerate(properties):
        
        sensor_data = data[property]
        sensor_data = np.array(sensor_data)

        time = data['time']
        time_filtered = time[~np.isnan(sensor_data)]
        
        plt.plot(time, sensor_data, label=property, color=plot_colours[idx][0])
        filtered_signal = lowPassFilter(sensor_data)
        
        plt.plot(time_filtered, filtered_signal, label=f'{property} filtered', color=plot_colours[idx][1])
        
        filtered_data.append(filtered_signal)
    
    filtered_data.insert(0, np.array(time_filtered))

    data_out = pd.DataFrame(np.transpose(filtered_data),
                            columns=['time', 'yaw', 'pitch', 'roll'])

    data_out.to_csv('data/tracking data/output_filtered.csv', index=False)

    plt.xlabel('Time (s)')
    plt.ylabel('Angular velocity')
    plt.legend(loc='upper right')
    plt.savefig('data/plot images/filtered.pdf')
    if show == True:
        plt.show()


def lowPassFilter(signal):
    '''
    Low pass filter parameters and utility
    '''
    fs = 4000
    lowcut = 160

    nyq = 0.5 * fs
    low = lowcut / nyq

    order = 4

    b, a = scipy.signal.butter(order, low, analog=False)
    y = scipy.signal.filtfilt(b, a, signal[~np.isnan(signal)])

    return y

