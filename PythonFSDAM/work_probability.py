# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""functions to create and elaborate work probabilities
"""

import math

import numpy as np


def make_probability_histogram(values, bin_with=0.1, normalize=True):
    """Creates a normalized histogram of the values

    It uses numpy.histogram

    Parameters
    ------------
    values : numpy.array
        The work values to make the normalized histogram of
    bin_with :
        the with each bin should (more or less) have
        as the number of bins shall be an integer the
        real bin with migh differ a little bit from the input one
        default = 0.1
    normalize : bool, optional, defalut=True
        if true the histogram will be normalized

    Returns
    ----------
    hist : numpy.array
        the normalized histogram
    bin_edges : numpy.array
        Return the bin edges (length(hist)+1) of the histogram
    """

    #get the number of bins
    number_of_bins = abs(np.max(values))
    number_of_bins /= bin_with
    number_of_bins = math.ceil(number_of_bins)

    return np.histogram(values, bins=number_of_bins, density=normalize)
