# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    filter.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/26 15:59:05 by taston            #+#    #+#              #
#    Updated: 2023/09/01 11:33:14 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import scipy
import numpy as np


class Filter:
    """
    A class used to represent a Filter for data

    ...
    
    Attributes
    ----------
    a : ndarray
        denominator polynomial of the IIR filter
    b : ndarray
        numerator polynomial of the IIR filter

    More information on IIR filters here:
    https://en.wikipedia.org/wiki/Infinite_impulse_response

    Methods
    -------
    low_pass_butterworth(fs=4000, lowcut=160, order=4)
        creates a low pass butterworth filter
    apply(signal)
        applies filter to a given signal
    """
    def __init__(self):
        nyq = 2000
        low = 160 / nyq
        self.b, self.a = scipy.signal.butter(4, low, analog=False)
        
    def low_pass_butterworth(self, fs=4000, lowcut=160, order=4):
        """Creates a low pass butterworth filter

        Parameters
        ----------
        fs : float, optional
            sampling frequency (default 4000Hz)
        lowcut : float, optional
            lowcut frequency (default 160Hz)
        order : int, optional
            order of the filter (default 4)

        Returns
        -------
        self
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        
        # Create scipy Butterworth filter:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        self.b, self.a = scipy.signal.butter(order, low, analog=False)
        
        return self
    
    def apply(self, signal):
        """Applies filter to a signal
        
        Parameters
        ----------
        signal : list
            signal for filter to be applied to
        
        Returns
        -------
        filtered_signal : list
            new filtered signal
        """
        filtered_signal = scipy.signal.filtfilt(self.b, self.a, signal[~np.isnan(signal)])
        return filtered_signal