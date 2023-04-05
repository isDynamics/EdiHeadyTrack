# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    archiving.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/03/27 11:09:20 by taston            #+#    #+#              #
#    Updated: 2023/03/27 11:16:27 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import shutil

def archive(files_in=[], files_out=[]):
    '''
    Function for archiving most recent plot files
    with new names, allowing for files to be organised.

    TODO: maybe just implement this directly into the tracking,
    filtering, processing and comparing modules directly.
    '''

    for idx, file in enumerate(files_in):
        shutil.copyfile(file, files_out[idx])