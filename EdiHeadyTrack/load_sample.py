# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    load_sample.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/05/16 08:40:02 by taston            #+#    #+#              #
#    Updated: 2023/05/16 08:56:28 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2

def load_vid(filename):
    """
    Return a video object from the sample data
    """
    video = cv2.VideoCapture(filename)