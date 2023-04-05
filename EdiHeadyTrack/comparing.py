# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    comparing.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/03/23 13:53:20 by taston            #+#    #+#              #
#    Updated: 2023/04/05 14:14:07 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

from matplotlib import gridspec
# import numpy as np


def compare(video_file, sensor_file, show=False):
    '''
    Compare outputs between head kinematic data obtained using markerless 
    head pose detection and WAX-9 sensor
    '''

    # Read files into dataframes
    video_data = pd.read_csv(video_file)
    sensor_data = pd.read_csv(sensor_file)
    sensor_data.columns = ['sensor', 
                         'receivedTime','sampleNumber','sampleTime',
                         'accelX','accelY','accelZ',
                         'gyroX','gyroY','gyroZ',
                         'magX','magY','magZ']

    sensor_data['adjustedTime'] = sensor_data['sampleTime'] - sensor_data['sampleTime'][0] - 59.335
    # fig = plt.figure()
    # plt.plot(sensor_data['adjustedTime'], sensor_data['gyroY'])
    # plt.show()

    video_data_restricted = video_data.query('time < 1.25')
    video_data_restricted = video_data_restricted.query('time > 0.4')
    sensor_data_restricted = sensor_data.query('adjustedTime > 0.4')
    sensor_data_restricted = sensor_data_restricted.query('adjustedTime < 1.25')
    video_data_restricted['time'] = video_data['time']
    # Compare maximum values...
    print('#'*40)
    print('Video')
    print('#'*40)
    print(f'max omega_yaw = {video_data_restricted["omega_yaw"][2:3750].abs().max()}')
    print(f'max omega_pitch = {video_data_restricted["omega_pitch"][2:3750].abs().max()}')
    print(f'max omega_roll = {video_data_restricted["omega_roll"][2:3750].abs().max()}')
    print('#'*40)
    print('Sensor')
    print('#'*40)
    print(f'max omega_yaw = {sensor_data_restricted["gyroX"].abs().max()}')
    print(f'max omega_pitch = {sensor_data_restricted["gyroY"].abs().max()}')
    print(f'max omega_roll = {sensor_data_restricted["gyroZ"].abs().max()}')

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure('Data Comparison',figsize=[7,7])    

    ax2 = plt.subplot(412)
    ax2.plot(sensor_data_restricted['adjustedTime'], sensor_data_restricted['gyroX'], 'k', label='WAX-9')
    ax2.plot(video_data_restricted['time'], video_data_restricted['omega_yaw'], 'c', label='HPE',linestyle='dashed')
    ax2.grid(color='0.95')
    ax2.set_ylabel(r'$ \omega_{yaw} $ (deg/s)')
    plt.xlim(0.85, 1.35)
    
    ax3 = plt.subplot(413, sharex=ax2)
    ax3.plot(sensor_data_restricted['adjustedTime'], sensor_data_restricted['gyroY'], 'k')
    ax3.plot(video_data_restricted['time'], video_data_restricted['omega_pitch'], 'c', linestyle='dashed')
    ax3.grid(color='0.95')
    ax3.set_ylabel(r'$ \omega_{pitch} $ (deg/s)')
    # plt.ylim(-300, 300)

    ax4 = plt.subplot(414, sharex=ax2)
    ax4.plot(sensor_data_restricted['adjustedTime'], sensor_data_restricted['gyroZ'], 'k')
    ax4.plot(video_data_restricted['time'], video_data_restricted['omega_roll'], 'c', linestyle='dashed')
    ax4.grid(color='0.95')
    ax4.set_ylabel(r'$ \omega_{roll} $ (deg/s)')
    plt.xlabel(r'Time (s)')

    ax1 = plt.subplot(411, sharex=ax2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.setp(ax1.get_yticks(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # Vertical line plotting
    axes = [ax1, ax2, ax3, ax4]
    key_times = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    # key_times = []
    for time in key_times:
        for axis in axes:
            axis.axvline(x=time, color='green', ls=':')

    # Get key frames from line times
    key_frames = [int(240*time) for time in key_times]
    
    # Add images to ax1
    for idx, frame in enumerate(key_frames):
        img = plt.imread(f'data/tracking frames/{key_frames[idx]}.0.png', format='png')
        img = img[0:800, 0:450, :]

        imagebox = OffsetImage(img, zoom=0.08)
        imagebox.image.axes = ax1

        ab = AnnotationBbox(imagebox, [key_times[idx],0.5],
                            xybox=(1, 1),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.01,
                            bboxprops =dict(edgecolor='white')
                            )

        ax1.add_artist(ab)
    
    ax1.set_yticks([])

    handles, labels = ax2.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='right')
    ax3.legend(handles, labels, bbox_to_anchor=(1.0, 0.85), loc=2)
    # plt.tight_layout()
    plt.savefig('data/plot images/comparison.svg', bbox_inches='tight')
    plt.savefig('data/plot images/comparison.png', bbox_inches='tight')
    
    if show == True:
        plt.show()