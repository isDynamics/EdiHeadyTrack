# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/27 10:18:49 by taston            #+#    #+#              #
#    Updated: 2023/06/01 13:47:57 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


from .imu import Wax9
from .sensordata import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

class Plot:
    """
    A class representing a Plot

    ...
    
    Attributes
    ----------
    colors : list
        list of colors used in plots
    sensors : list, SensorData
        list of SensorData objects to be plotted
    lines : list, dict 
        list of dicts containing data of each line being plotted 

    Methods
    -------
    plot_propertyplot_property(self, property='velocity', xlim=None, ylim=None, key_times=[], show=True):
        plots a time comparison between sensor data
    """
    def __init__(self, *sensors):
        self.colors = []
        self.sensors = []
        self.lines = []
        for sensor in sensors:
            if not isinstance(sensor, SensorData):
                raise TypeError(f'Attempted to plot type {type(sensor)}. Only SensorData objects may be plotted here!')
            else:
                self.sensors.append(sensor)


    def plot_property(self, property='velocity', 
                      xlim=None, ylim=None, key_times=[], show=True):
        """Plots comparison between sensor data
        
        Parameters
        ----------
        property : str
            specifies which property to be plotted (from ('pose', 'velocity', 'acceleration')
        xlim : tuple, float
            range of times to be plotted over (lower limit, upper limit)
        ylim : tuple, float
            range of y values to be plotted over (lower limit, upper limit)
        key_times : list, float, optional
            key time points to display frame images for 
        *sensors : SensorData
            sensor data objects to be added to plot
        """
        plt.rcParams.update({'font.size': 14})
        fig = plt.figure('EdiHeadyTrack - Head Data Plotting',figsize=[6,8])    

        # if key_times:
        #     key_frames = [int(240*time) for time in key_times]

        for sensor in self.sensors:
            if hasattr(sensor, property):
                property_dict = getattr(sensor, property)
                for idx, key in enumerate(list(property_dict.keys())[1:]):
                    
                    ax = plt.subplot(len(list(property_dict.keys())), 1, idx+2)
                    ax.grid(color='0.95')
                    if property == 'pose':
                        ax.set_ylabel(fr'$\theta_{{{key}}}$ (deg)')
                    elif property == 'velocity':
                        ax.set_ylabel(fr'$\omega_{{{key}}}$ (deg/s)')
                    elif property =='acceleration':
                        ax.set_ylabel(fr'$\alpha_{{{key}}}$ (deg/s$^2$)')

                    if xlim:
                        ax.set_xlim(xlim)

                    # TODO: improve method of using different colors.
                    #       Maybe use 
                    lims = []
                    if isinstance(sensor, Head):
                        x_data = property_dict['time']
                        y_data = property_dict[key]
                        if xlim:
                            for lim in xlim:
                                for idx, time in enumerate(x_data):
                                    if time > lim:
                                        lims.append(idx)
                                        break
                            x_data = x_data[lims[0]:lims[1]]
                            y_data = y_data[lims[0]:lims[1]]
                        line, = ax.plot(x_data, 
                                        y_data, 
                                        label=f'{sensor.id}', 
                                        color='c', ls='dashed')
        
                        self.lines.append({'sensor':    sensor.id,
                                           'property':  f'{key}, {property}',
                                           'values':    y_data})
                        if key_times:
                            key_frames = [int(sensor.facedetector.video.fps*time) for time in key_times]
                            
                    elif isinstance(sensor, IMU):
                        x_data = property_dict['time']
                        y_data = property_dict[key]
                        if xlim:
                            for lim in xlim:
                                for idx, time in enumerate(x_data):
                                    if time > lim:
                                        lims.append(idx)
                                        break
                            x_data = x_data[lims[0]:lims[1]]
                            y_data = y_data[lims[0]:lims[1]]
                        line, = ax.plot(x_data, 
                                        y_data, 
                                        label=f'{sensor.id}', 
                                        color='k')
                        self.lines.append({'sensor':    sensor.id,
                                           'property':  f'{key}, {property}',
                                           'values':    y_data})
            
                    for time in key_times:
                        ax.axvline(x=time, color='green', ls=':')
                        
                    if idx + 2 != len(list(property_dict.keys())):
                        plt.setp(ax.get_xticklabels(), visible=False)
                    else:
                        ax.set_xlabel(r'Time (s)')
                    
                    if idx + 2 == 3: 
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend(handles, labels, bbox_to_anchor=(1.0, 0.85), loc=2)

        ax1 = plt.subplot(411, sharex=ax)
        for time in key_times:
            ax1.axvline(x=time, color='green', ls=':')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)

        if key_times:
            for sensor in self.sensors:
                if isinstance(sensor, Head):
                    for idx, frame in enumerate(key_frames):
                        img = sensor.facedetector.tracking_frames[frame]

                        frame_index = sensor.facedetector.face2d['frame'].index(frame)
                        x_list = [pos[0] for pos in sensor.facedetector.face2d['all landmark positions'][frame_index]]
                        y_list = [pos[1] for pos in sensor.facedetector.face2d['all landmark positions'][frame_index]]
                        
                        top_bound = max(0, min(y_list[:]) - 200)
                        bottom_bound = min(sensor.facedetector.video.height, max(y_list[:]) + 200)
                        left_bound = max(0, min(x_list[:]) - 200)
                        right_bound = min(sensor.facedetector.video.width, max(x_list[:]) + 200)
                        
                        img = img[top_bound:bottom_bound, left_bound:right_bound, :]

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
        
        # plt.savefig('resources/comparison.png', bbox_inches='tight')
        if show == True:
            plt.show()
        
        return self

    def summarise(self):
        '''
        Produces a summary of the maximum values of each line in a 
        Plot object.
        '''
        sensor_prev = None
        for line in self.lines:
            sensor = line['sensor']

            if sensor != sensor_prev:
                print('-'*46)
                print(f'{sensor} plot summary')
                print('-'*46)
                print('{:<30} {:>15}'.format(line['property'], 
                                             round(max(abs(line['values'])), 2)))
            else:
                print('{:<30} {:>15}'.format(line['property'], 
                                             round(max(abs(line['values'])), 2)))
            
            sensor_prev = sensor
        
        print('-'*46)