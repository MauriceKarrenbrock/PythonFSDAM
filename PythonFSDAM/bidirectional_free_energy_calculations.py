# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Functions and classes to do bidirectional (Crooks) free energy calculations
"""

import math

import numpy as np
import scipy.optimize as sp_opt

import PythonFSDAM.bootstrapping as boot
import PythonFSDAM.combine_works as combine


def bar_free_energy(works_1,
                    works_2,
                    temperature=298.15,
                    boltzmann_kappa=0.001985875):
    """Get the free energy of a bi directional process
    with the Bennett acceptance ratio BAR
    (Crooks theorem)

    To avoid float overflow `works_1` should be negative and
    `works_2` positive

    works_1 : numpy.array
        1-D numpy array containing the values of the
        non equilibrium works
        if you don't modify `boltzmann_kappa`
        they should be Kcal (1 Kcal = 1/4.148 KJ)
        To avoid float overflow `works_1` should be negative and
        `works_2` positive
    works_2 : numpy.array
        1-D numpy array containing the values of the
        non equilibrium works
        if you don't modify `boltzmann_kappa`
        they should be Kcal (1 Kcal = 1/4.148 KJ)
        To avoid float overflow `works_1` should be negative and
        `works_2` positive
    temperature : float, optional
        the temperature in Kelvin at which the non equilibrium simulation
        has been done, default 298.15 K
    boltzmann_kappa : float
        the Boltzmann constant, the dafault is 0.001985875 kcal/(mol⋅K)
        and if you keep it the `works` shall be in Kcal

    Returns
    ----------
    free_energy : float
        the value of the free energy
        if you used the default `boltzmann_kappa` and the
        `works` where in Kcal it is in Kcal/mol
    """

    def function_to_minimize(x, works_1, works_2, beta, M):

        def helper(works, x, beta, M):
            # A vectorized and memery efficient way to calculate
            # 1./(1 + EXP( beta * (work + M - x) ))
            summation = works + M
            summation -= x
            summation *= beta
            summation = np.exp(summation)
            summation += 1
            summation = 1 / summation

            # summation -> becomes a float
            summation = np.sum(summation)

            return summation

        sum_1 = helper(works_1, x, beta, M)

        # Here beta has the sign switched!
        sum_2 = helper(works_2, x, -beta, M)

        return (sum_1 - sum_2)**2

    kappa_T = temperature * boltzmann_kappa

    if works_1.size == works_2.size:
        M = 0.
    else:
        M = kappa_T * math.log(works_1.size / works_2.size)

    # Mean of the means as first guess
    x0 = np.mean(works_1) + np.mean(works_2)
    x0 /= 2.

    free_energy = sp_opt.fmin(function_to_minimize,
                              x0=x0,
                              args=(works_1, works_2, 1. / kappa_T, M),
                              disp=False,
                              full_output=False)

    # The function returns a 1X1 array, I want a float
    free_energy = free_energy[0]

    return free_energy


# by using this class I can choose the temp and K inside the bootstrapping function
# had to put it outside the function to allow the use of multiprocessing
class _vDSSBBarErrorPropagationHelperClass(object):
    """helper class
    """

    def __init__(self, temperature, boltzmann_kappa):

        self.temperature = temperature
        self.boltzmann_kappa = boltzmann_kappa

    def calculate_bar(self, works_1, works_2):
        """helper method
        """

        return bar_free_energy(works_1,
                               works_2,
                               temperature=self.temperature,
                               boltzmann_kappa=self.boltzmann_kappa)


def vDSSB_bar_error_propagation(bound_works_1,
                                unbound_works_1,
                                bound_works_2,
                                unbound_works_2,
                                *,
                                temperature=298.15,
                                boltzmann_kappa=0.001985875,
                                num_iterations=10000):
    """get STD of the BAR free energy obtained by convolution of bound and unbound work vDSSB

    starting from the bound and unbound work values calculates the STD of the bar
    free energy estimate (if you use the vDSSB method)
    it uses bootstrapping, can be very time and memory consuming

    Parameters
    -----------
    bound_works_1 : numpy.array
        the first numpy array of the work values
        To avoid float overflow the sum of `works_1` should be negative and
        the sum of `works_2` positive
    unbound_works_1 : numpy.array
        the first numpy array of the work values
        To avoid float overflow the sum of `works_1` should be negative and
        the sum of `works_2` positive
    bound_works_2 : numpy.array
        the second numpy array of the work values
        To avoid float overflow the sum of `works_1` should be negative and
        the sum of `works_2` positive
    unbound_works_2 : numpy.array
        the first numpy array of the work values
        To avoid float overflow the sum of `works_1` should be negative and
        the sum of `works_2` positive
    temperature : float, optional
        the temperature in Kelvin at which the non equilibrium simulation
        has been done, default 298.15 K
    boltzmann_kappa : float
        the Boltzmann constant, the dafault is 0.001985875 kcal/(mol⋅K)
        and if you keep it the `works` shall be in Kcal
    num_iterations : ins, optional, default=1000
        the number of bootstrapping iterations (time and memory consuming)

    Returns
    ----------
    STD, mean : float, float
        the STD of the BAR estimate and it's bootstrapped mean value

    Notes
    -----------
    it uses the `bar_free_energy` present in this module
    """

    helper_obj = _vDSSBBarErrorPropagationHelperClass(
        temperature=temperature, boltzmann_kappa=boltzmann_kappa)

    out_mean, out_std = boot.mix_and_bootstrap_multiple_couples_of_values(
        ((bound_works_1, unbound_works_1), (bound_works_2, unbound_works_2)),
        mixing_function=combine.combine_non_correlated_works,
        stat_function=helper_obj.calculate_bar,
        num_iterations=num_iterations)

    return out_std, out_mean


def plain_bar_error_propagation(works_1,
                                works_2,
                                *,
                                temperature=298.15,
                                boltzmann_kappa=0.001985875,
                                num_iterations=10000):
    """bar STD via bootstrap

    it doesn't use the vDSSB aproach, but only the normal
    one, for the rest it is 100% equal to `vDSSB_bar_error_propagation`

    Parameters
    -------------
    works_1 : numpy.array
        the work values
        To avoid float overflow `works_1` should be negative and
        `works_2` positive
    works_2 : numpy.array
        the work values
        To avoid float overflow `works_1` should be negative and
        `works_2` positive
    temperature : float, optional
        the temperature in Kelvin at which the non equilibrium simulation
        has been done, default 298.15 K
    boltzmann_kappa : float
        the Boltzmann constant, the dafault is 0.001985875 kcal/(mol⋅K)
        and if you keep it the `works` shall be in Kcal
    num_iterations : ins, optional, default=10000
        the number of bootstrapping iterations (time and memory consuming)

    Returns
    ----------
    STD, mean : float, float
        the STD of the BAR estimate and it's bootstrapped mean value
    """

    helper_obj = _vDSSBBarErrorPropagationHelperClass(
        temperature=temperature, boltzmann_kappa=boltzmann_kappa)

    tot_works = works_1, works_2

    out_mean, out_std = boot.bootstrap_std_of_function_results(
        tot_works,
        function=helper_obj.calculate_bar,
        num_iterations=num_iterations,
        number_of_sets_of_values=len(tot_works))

    return out_std, out_mean
