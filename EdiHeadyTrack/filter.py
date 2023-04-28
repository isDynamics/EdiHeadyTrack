# # **************************************************************************** #
# #                                                                              #
# #                                                         :::      ::::::::    #
# #    filter.py                                          :+:      :+:    :+:    #
# #                                                     +:+ +:+         +:+      #
# #    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
# #                                                 +#+#+#+#+#+   +#+            #
# #    Created: 2023/04/26 15:59:05 by taston            #+#    #+#              #
# #    Updated: 2023/04/28 11:08:38 by taston           ###   ########.fr        #
# #                                                                              #
# # **************************************************************************** #

# import scipy
# import numpy as np


class Filter:
    def __init__(self):
        ...
#         # nyq = 2000
#         # low = 160 / nyq
#         # self.b, self.a = scipy.signal.butter(4, low, analog=False)
#         print('Filter created')
#         # print(self.b)
        
#     # def low_pass_butterworth(self, fs=4000, lowcut=160, order=4):
#     #     nyq = 0.5 * fs
#     #     low = lowcut / nyq
#     #     self.b, self.a = scipy.signal.butter(order, low, analog=False)

#     # def apply(self, signal):
#     #     filtered_signal = scipy.signal.filtfilt(self.b, self.a, signal[~np.isnan(signal)])
#     #     return filtered_signal