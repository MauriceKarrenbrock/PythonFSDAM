# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""functions to purge outlier values from a set
"""

import numpy as np
import scipy.stats as stats


def purge_outliers_zscore(values, z_score=3.0):
    """Purges outliers using zscore

    it calculates the z score of `values` respectively
    to `values` mean and STD and removes all the values
    with a z score outside [-`z_score` `zscore`] intervall
    (uses strictly > < and not >= =<), the default z score is 3.

    Parameters
    -----------
    values : array
        a 1-D numpy array or something similar
        it is the array of values to test and
        purge
    z_score : float
        the value of z score that determines an outlier
        default 3.0

    Returns
    -----------
    numpy.array
        if some values have been purged it will be shorter than the
        input one
    """

    if values.size == 0:
        raise ValueError('The input cannot be an empty array')

    if z_score < 0:
        z_score = -(z_score)

    z_scores_array = stats.zscore(values, nan_policy='raise')

    values_to_purge = []

    iterator = np.nditer(z_scores_array, flags=['c_index'])
    for z_value in iterator:

        if z_value < -(z_score) or z_value > z_score:

            values_to_purge.append(iterator.index)

    #free the memory, can be useful with very long arrays
    del z_scores_array

    output_array = np.empty([values.size - len(values_to_purge)],
                            dtype=values.dtype)

    i = 0
    iterator = np.nditer(values, flags=['c_index'], order='C')
    for value in iterator:

        if iterator.index not in values_to_purge:

            output_array[i] = value

            i += 1

    #a maybe useless check but better safe than sorry
    if i != output_array.size:
        raise Exception('Something went wrong, check out your input array')

    return output_array
