# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/05 13:36:12 by taston            #+#    #+#              #
#    Updated: 2023/04/05 13:36:12 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# TODO: add tests...
# TODO: covering face with mesh?
# TODO: Github organisation

import EdiHeadyTrack.calibrating    # for calibrating camera, extracting intrinsic parameters
import EdiHeadyTrack.logging        # for opening csv files to write data to
import EdiHeadyTrack.tracking       # for tracking specified video files
import EdiHeadyTrack.filtering      # for filtering measured signals with Butterworth filter
import EdiHeadyTrack.processing     # for calculating kinematics from measured displacements
import EdiHeadyTrack.comparing      # for comparing values with sensor data 
import EdiHeadyTrack.archiving      # for archiving outputs to ordered folder system

def main():
    '''
    Worked example use of EdiHeadyTrack.
    '''
    EdiHeadyTrack.calibrating.calibrate('videos/calibration_xiaomi.mp4')
    EdiHeadyTrack.logging.log(show=True)
    EdiHeadyTrack.tracking.track('videos/head_240_final.mp4', maxFaces=1, show=True)
    EdiHeadyTrack.filtering.filter('data/tracking data/output.csv', show=True)
    EdiHeadyTrack.processing.process('data/tracking data/output_filtered.csv', show=True)
    EdiHeadyTrack.comparing.compare('data/tracking data/output_final.csv', 
                                    'data/sensor data/head_240_final_sensor.csv', show=True)
    EdiHeadyTrack.archiving.archive(['data/tracking data/output_final.csv'], 
                                    ['data/data archives/240_head_4.csv'])

if __name__ == "__main__":
    main()