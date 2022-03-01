# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Contains functions that use bootstrapping methods
"""

import os
from multiprocessing import Pool

import numpy as np

import PythonFSDAM.combine_works as combine_works


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
        stat_function,
        mixing_function=combine_works.combine_non_correlated_works,
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
        if left None the function will first check for
        the environment variable OMP_NUM_THREADS
        if it is not defined the default of multiprocessing.Pool
        will be used (= the total nuymber of CPUs on the machine)

    Returns
    ------------
    float, float
        the bootstrapped value of the `stat_function` and the standard deviation STD

    Notes
    --------
    this function is can be memory intensive if heavilly parallelized

    the fact that if nothing is given as input the function first looks
    for OMP_NUM_THREADS instead of the total nuymber of CPUs on the machine
    is because it can be a problem on HPC clusters (you cannot simply use all
    resources of the access node to do some post processing)
    """

    if num_iterations < 1:
        raise ValueError(
            f'num_iterations can not be less than one, it is {num_iterations}')

    if n_threads is None:

        n_threads = os.environ.get('OMP_NUM_THREADS', None)

        try:

            n_threads = int(n_threads)

        except TypeError:  #None cannot be casted to int
            pass

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
                                      function,
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
