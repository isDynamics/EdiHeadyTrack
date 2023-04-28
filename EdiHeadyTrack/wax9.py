# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    wax9.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/26 16:03:41 by taston            #+#    #+#              #
#    Updated: 2023/04/26 16:04:43 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from .imukinematics import IMUKinematics

class Wax9(IMUKinematics):
    def __init__(self):
        super().__init__()