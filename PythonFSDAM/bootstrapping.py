# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Contains functions that use bootstrapping methods
"""

from multiprocessing import Pool

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np

import PythonFSDAM.combine_works as combine_works


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

    return bs_object.value, (bs_object.lower_bound, bs_object.upper_bound)


def _mix_and_bootstrap_helper_function(values_1, values_2, mixing_function,
                                       stat_function):
    """private

    helper function
    """

    #numpy random number generator
    rng = np.random.default_rng()

    bs_values_1 = rng.choice(values_1, values_1.shape, replace=True)

    bs_values_2 = rng.choice(values_2, values_2.shape, replace=True)

    mixed_values = mixing_function(bs_values_1, bs_values_2)

    #try not to keep to much stuff in memory
    del bs_values_1
    del bs_values_2

    return stat_function(mixed_values)


def mix_and_bootstrap(
        values_1,
        values_2,
        *,
        mixing_function=combine_works.combine_non_correlated_works,
        stat_function=bs_stats.mean,
        num_iterations=10000,
        n_threads=None):
    """complex combining and bootstrapping

    it boostraps values from `values_1` and `values_2`, combines them with the
    given `mixing_fuction` and then calculates the `stat_function` on the obtained
    values. Returns the mean and the STD of the obtained values

    it uses multiprocessing to parallelize the task

    Parameters
    --------------
    values_1 : numpy.array
        first set of values
    values_2 : numpy.array
        second set of values
    mixing_function : function, default=`PythonFSDAM.combine_works.combine_non_correlated_works`
        function to use to mix the bootstrapped values_1 with the bootstrapped
        values_2, must accept 2 numpy.array and return one
        numpy.array
    stat_function : function, default=`bootstrapped.stats_functions.mean`
        the function that will be applied to the mixed values obtained from
        `mixing_function`, must accept a numpy.array and return a float
        for more info on the default function check
        https://github.com/facebookincubator/bootstrapped
    num_iterations : int, default=10000
        the number of bootstrapping iterations
    n_threads : int, optional, default None
        the number of parallel threads to use
        if left None they will be detected automatically
        by multiprocessing.Pool

    Returns
    ------------
    float, float
        the bootstrapped value of the `stat_function` and the standard deviation STD

    Notes
    --------
    this function is can be memory intensive if heavilly parallelized
    """

    if num_iterations < 1:
        raise ValueError(
            f'num_iterations can not be less than one, it is {num_iterations}')

    #make input for parallel section
    input_values = []
    for _ in range(num_iterations):

        input_values.append(
            (values_1, values_2, mixing_function, stat_function))

    #make bootstrap and mix in parallel
    with Pool(n_threads) as pool:

        bootstapped_stat_function = pool.starmap_async(
            _mix_and_bootstrap_helper_function, input_values)

        bootstapped_stat_function = bootstapped_stat_function.get()

    bootstapped_stat_function = np.array(bootstapped_stat_function)

    return np.mean(bootstapped_stat_function), np.std(
        bootstapped_stat_function)


def _fake_mixing_function(values_1, _):
    """dummy function

    I use it to reuse `mix_and_bootstrap`

    Returns
    -----------
    the first argument
    """

    return values_1


def bootstrap_std_of_function_results(values,
                                      *,
                                      function=bs_stats.mean,
                                      num_iterations=10000,
                                      n_threads=None):
    """bootstrap then apply the function then return mean and std

    it boostraps values from `values` and then calculates the `function` on the obtained
    values. Returns the mean and the STD of the obtained values

    Parameters
    -----------
    values : numpy.array
        the values to bootstrap
    function : function(x)
        the function that will process the bootstrapped values
    for the other parameters check `mix_and_bootstrap`

    Returns
    -------------
    mean, std : float, float

    Notes
    ---------
    this function uses `mix_and_bootstrap`
    """

    return mix_and_bootstrap(
        values,
        np.array([1.]),  #dummy argument
        mixing_function=_fake_mixing_function,
        stat_function=function,
        num_iterations=num_iterations,
        n_threads=n_threads)
