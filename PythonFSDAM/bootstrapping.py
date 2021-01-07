# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Contains functions that use bootstrapping methods
"""

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


def standard_deviation(values,
                       alpha=0.05,
                       num_iterations=10000,
                       iteration_batch_size=10,
                       num_threads=1):
    """get standard deviation with bootstrapping

    it is a wrapper of bootstrapped.bootstrap.bootstrap function
    https://github.com/facebookincubator/bootstrapped

    Parameters
    ------------
    values : numpy array (or scipy.sparse.csr_matrix)
        values to bootstrap
    alpha : float, optional
        alpha value representing the confidence interval.
        Defaults to 0.05, i.e., 95th-CI. (2sigma)
    num_iterations: int, optional
        number of bootstrap iterations to run. Default 10000
    iteration_batch_size: int, optional
        The bootstrap sample can generate very large
        matrices. This argument limits the memory footprint by
        batching bootstrap rounds. If unspecified the underlying code
        will produce a matrix of len(values) x num_iterations. If specified
        the code will produce sets of len(values) x iteration_batch_size
        (one at a time) until num_iterations have been simulated.
        Defaults to 10. Passing None will calculate the full simulation in one step.
    num_threads: int, optional
        The number of therads to use. This speeds up calculation of
        the bootstrap. Defaults to 1. If -1 is specified then the number
        of CPUs will be counted automatically
    Returns
    ------------
    bootstrappend_STD, (confidence_intervall_lower_bound, confidence_intervall_upper_bound) :
        float, tuple(float,float)
    """

    bs_object = bs.bootstrap(values=values,
                             stat_func=bs_stats.std,
                             alpha=alpha,
                             num_iterations=num_iterations,
                             iteration_batch_size=iteration_batch_size,
                             num_threads=num_threads)
    print(type(bs_object))
    print(bs_object)

    return bs_object.value, (bs_object.lower_bound, bs_object.upper_bound)
