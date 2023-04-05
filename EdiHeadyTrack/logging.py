# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logging.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/15 14:16:46 by taston            #+#    #+#              #
#    Updated: 2023/03/22 13:56:31 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from multiprocessing import Process
from .plot_utils import plot

def log(show=False):
    if show == True:
        p = Process(target=plot)
        p.start()
    else:
        ...