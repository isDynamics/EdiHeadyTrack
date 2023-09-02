# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/27 10:18:49 by taston            #+#    #+#              #
#    Updated: 2023/09/01 14:51:54 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


from .imu import Wax9
from .sensordata import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import matplotlib.colors as mcolors

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
        self.colors = ['k', 'c', 'orange', 'r', 'g']
        self.linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
        self.sensors = []
        self.heads = []
        self.lines = []
        for sensor in sensors:
            if not isinstance(sensor, SensorData):
                raise TypeError(f'Attempted to plot type {type(sensor)}. Only SensorData objects may be plotted here!')
            else:
                self.sensors.append(sensor)
                if isinstance(sensor, Head):
                    self.heads.append(sensor)
                
        # print(self.heads)


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
        gridspec_kw={'height_ratios': [2, 1, 1, 1]}
        fig, axs = plt.subplots(4,1,figsize=[6,8], gridspec_kw=gridspec_kw)    

        for sensor_idx, sensor in enumerate(self.sensors):
            if hasattr(sensor, property):
                property_dict = getattr(sensor, property)
                if isinstance(sensor, Head) and property=='pose':
                    start=2
                else:
                    start=1
                for ax_idx, key in enumerate(list(property_dict.keys())[start:]):
                    # axs[ax_idx+1] = plt.subplot(len(list(property_dict.keys())), 1, ax_idx+1)
                    axs[ax_idx+1].grid(color='0.95')
                    if property == 'pose':
                        axs[ax_idx+1].set_ylabel(fr'$\theta_{{{key}}}$ (deg)')
                    elif property == 'velocity':
                        axs[ax_idx+1].set_ylabel(fr'$\omega_{{{key}}}$ (deg/s)')
                    elif property =='acceleration':
                        axs[ax_idx+1].set_ylabel(fr'$\alpha_{{{key}}}$ (deg/s$^2$)')

                    if xlim:
                        axs[ax_idx+1].set_xlim(xlim)
                    if ylim:
                        axs[ax_idx+1].set_ylim(ylim)
                    lims = []

                    x_data = property_dict['time']
                    y_data = property_dict[key]

                    # cropping data
                    if xlim:
                        for lim in xlim:
                            for time_idx, time in enumerate(x_data):
                                if time > lim:
                                    lims.append(time_idx)
                                    break
                        
                        x_data = x_data[lims[0]:lims[1]]
                        y_data = y_data[lims[0]:lims[1]]

                    # print(f'plotting line with color: {self.colors[sensor_idx]}')
                    line, = axs[ax_idx+1].plot(x_data,
                                    y_data,
                                    label=f'{sensor.id}',
                                    color=self.colors[sensor_idx],
                                    ls=self.linestyles[sensor_idx])
                    
                    if isinstance(sensor, Head):
                        if key_times:
                            key_frames = [int(sensor.posedetector.video.fps*time) for time in key_times]
                    
                    self.lines.append({'sensor':    sensor.id,
                                       'property':  f'{key}, {property}',
                                       'values':    y_data})
                    
                    for time in key_times:
                        axs[ax_idx+1].axvline(x=time, color='green', ls=':')
                        
                    if ax_idx != 2:
                        plt.setp(axs[ax_idx+1].get_xticklabels(), visible=False)
                    else:
                        axs[ax_idx+1].set_xlabel(r'Time (s)')
                    
                    if ax_idx == 1: 
                        handles, labels = axs[ax_idx+1].get_legend_handles_labels()
                        axs[ax_idx+1].legend(handles, labels, bbox_to_anchor=(1.0, 0.85), loc=2)
            else:
                print(f'{sensor} does NOT have property {property}')
                
        axs[0] = plt.subplot(411, sharex=axs[1])
        for time in key_times:
            axs[0].axvline(x=time, color='green', ls=':')
        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.setp(axs[0].get_yticklabels(), visible=False)

        if key_times:
            for sensor_idx, sensor in enumerate(self.heads):
                for idx, frame in enumerate(key_frames):
                    img = sensor.posedetector.tracking_frames[frame]

                    frame_index = sensor.posedetector.pose['frame'].index(frame)
                    x_list = [pos[0] for pos in sensor.posedetector.face2d['all landmark positions'][frame_index]]
                    y_list = [pos[1] for pos in sensor.posedetector.face2d['all landmark positions'][frame_index]]
                    # print(x_list)
                    # print(y_list)
                    top_bound = max(0, min(y_list[:]) - 200)
                    bottom_bound = min(sensor.posedetector.video.height, max(y_list[:]) + 200)
                    left_bound = max(0, min(x_list[:]) - 200)
                    right_bound = min(sensor.posedetector.video.width, max(x_list[:]) + 200)
                    
                    img = img[top_bound:bottom_bound, left_bound:right_bound, :]

                    imagebox = OffsetImage(img, zoom=0.08)
                    imagebox.image.axes = axs[0]
                    
                    if len(self.heads) != 1:
                        ab = AnnotationBbox(imagebox, [key_times[idx],0.7-0.4*sensor_idx/(len(self.heads)-1)],
                                            xybox=(1,  1),
                                            xycoords='data',
                                            boxcoords="offset points",
                                            pad=0.01,
                                            bboxprops =dict(edgecolor=self.colors[sensor_idx+1])
                                            )

                        axs[0].add_artist(ab)
        

        plt.savefig('resources/comparison.png', bbox_inches='tight')
        plt.savefig('resources/comparison.svg', bbox_inches='tight')
        if show == True:
            plt.show()
            plt.tight_layout
        
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
                                             round(abs(max(line['values'], key=abs)), 2)))
            else:
                print('{:<30} {:>15}'.format(line['property'], 
                                             round(abs(max(line['values'], key=abs)), 2)))
            
            sensor_prev = sensor
        
        print('-'*46)