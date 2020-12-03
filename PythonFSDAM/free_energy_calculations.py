# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Functions and classes to do free energy calculations
"""

import math

import numpy as np

from . import work_probability


# pylint: disable=anomalous-backslash-in-string
def jarzynski_free_energy(works,
                          temperature=298.15,
                          boltzmann_kappa=0.001985875):
    """Calculates the Jarzynski free energy

    starting from the non equilibrium works obtained
    from for example alchemical transformations
    you get the free energy difference: Delta F (A, G, ...)
    if the `works` are in Kcal and you keep the default `boltzmann_kappa`
    = 0.001985875 kcal/(mol⋅K) the result will be in Kcal/mol
    otherwise it depends on your choice

    DOES NOT DO A VOLUME CORRECTION NOR AN ERROR ESTIMATE!!!!!!!!

    Parameters
    -----------
    works : numpy.array
        1-D numpy array containing the values of the
        non equilibrium works
        if you don't modify `boltzmann_kappa`
        they should be Kcal (1 Kcal = 1/4.148 KJ)
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

    Notes
    -----------
    Normally the Jarzynski free energy is calculated so
    math :: \Delta G = -kT log (\sum_i e^{-\beta W_i} /N)

    but to avoid overflow I calculate it:
    math::
        \Delta G = -kT log (\sum_i e^{-\beta W_i} /N)
        = -kT log [e^ {-\beta W_{min}}(\sum_i e^{-\beta (W_i - W_{min})}/N]
        =  W_{min} - kT log(\sum_i e^{-\beta (W_i-W_{min})}/N]
        = W_{min} + kT (log N - log(\sum_i e^{-\beta (W_i-W_{min})})
    """

    #   To avoid overflow jarzynski average is computed as
    #   following:
    #   DG = -kT log (sum_i e^-bW_i /N)
    #      = -kT log [e^-bWmin(\sum_i e^-b(W_i-Wmin)/N]
    #      =  Wmin -kT log(\sum_i e^-b(W_i-Wmin)/N]
    #      = Wmin + kT (log N - log(\sum_i e^-b(W_i-Wmin))
    #   computes jarzynski correction

    # 1. / beta
    kappa_T = temperature * boltzmann_kappa

    work_min = np.min(works)

    delta_works = works - work_min

    delta_works = delta_works / kappa_T

    delta_works = np.exp(delta_works)

    free_energy = np.sum(delta_works)

    free_energy = math.log(free_energy)

    free_energy = work_min + kappa_T * (math.log(delta_works.size) -
                                        free_energy)

    return free_energy


def volume_correction(distance_values, temperature=298.15):
    """Calculates the volume correction of the free energy

    Parameters
    -------------
    distance_values : numpy.array
        1-D numpy.array containing the distance
        of the ligand from the protein (center of mass - center of mass)
        IN ANGSTROM!
    temperature : float
        Kelvin

    Returns
    ---------
    float
        the volume correction in Kcal/mol

    Notes
    ----------
    the correction is calculated
    math :: \Delta G_{vol} = RT ln (V_{site} / V_{0})
    with math:: V_0 = 1661
    for more info check https://dx.doi.org/10.1021/acs.jctc.0c00634
    """

    #angstrom
    standard_volume = 1661.

    #kcal/mol
    R_gas = 1.9872036 * (10**-3)

    bin_width = 0.1

    histogram = work_probability.make_probability_histogram(
        distance_values, bin_width)

    histogram = histogram[0]

    STD = np.std(histogram)

    site_volume = (4. / 3.) * math.pi * (2 * STD)**3

    Delta_G_vol = R_gas * temperature * math.log(site_volume / standard_volume)

    return Delta_G_vol


# def jarzynski_error_propagation(bound_error, unbound_error):
#     """Get the right confidence intervall of jarzynski

#     starting with the bound and unbound confidence intervall
#     the function propagates it in order to get the right confidence intervall
#     """

#     pass
