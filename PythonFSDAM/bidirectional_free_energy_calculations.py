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
