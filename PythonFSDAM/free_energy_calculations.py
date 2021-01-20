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

import PythonFSDAM.bootstrapping as boot
import PythonFSDAM.combine_works as combine
import PythonFSDAM.exp_max_gaussmix as em
import PythonFSDAM.work_probability as work_probability


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

    work_min = np.amin(works)

    delta_works = works - work_min

    delta_works = -(delta_works / kappa_T)

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


def jarzynski_error_propagation(works_1,
                                works_2,
                                *,
                                temperature=298.15,
                                boltzmann_kappa=0.001985875,
                                num_iterations=100):
    """get STD of the Jarzynki free energy obtained by convolution of bound and unbound work vDSSB

    starting from the bound and unbound work values calculates the STD of the Jarzynski
    free energy estimate (if you use the vDSSB method)
    it uses bootstrapping, can be very time and memory consuming

    Parameters
    -----------
    works_1 : numpy.array
        the first numpy array of the work values
    works_2 : numpy.array
        the second numpy array of the work values
    temperature : float, optional
        the temperature in Kelvin at which the non equilibrium simulation
        has been done, default 298.15 K
    boltzmann_kappa : float
        the Boltzmann constant, the dafault is 0.001985875 kcal/(mol⋅K)
        and if you keep it the `works` shall be in Kcal
    num_iterations : ins, optional, default=100
        the number of bootstrapping iterations (time and memory consuming)

    Returns
    ----------
    STD, mean : float, float
        the STD of the Jarzynski estimate and it's bootstrapped mean value

    Notes
    -----------
    it uses the `jarzynski_free_energy` present in this module, it uses bootstrapping,
    will take for granted that you mixed the work values with
    `PythonFSDAM.combine_works.combine_non_correlated_works` and uses the
    `PythonFSDAM.bootstrapping.mix_and_bootstrap` function to do the bootstrapping
    """

    # by using this class I can choose the temp and K inside the bootstrapping function
    class HelperClass(object):
        """helper class
        """
        def __init__(self, temperature, boltzmann_kappa):

            self.temperature = temperature
            self.boltzmann_kappa = boltzmann_kappa

        def calculate_jarzynski(self, values):
            """helper method
            """

            return jarzynski_free_energy(values,
                                         temperature=self.temperature,
                                         boltzmann_kappa=self.boltzmann_kappa)

    helper_obj = HelperClass(temperature=temperature,
                             boltzmann_kappa=boltzmann_kappa)

    out_mean, out_std = boot.mix_and_bootstrap(
        works_1,
        works_2,
        mixing_function=combine.combine_non_correlated_works,
        stat_function=helper_obj.calculate_jarzynski,
        num_iterations=num_iterations)

    return out_std, out_mean


def gaussian_mixtures_free_energy(works,
                                  temperature=298.15,
                                  boltzmann_kappa=0.001985875,
                                  n_gaussians=3,
                                  tol=1.E-6,
                                  max_iterations=None):
    """Calculates the free energy with the gaussian mixtures method

    starting from the non equilibrium works obtained
    from for example alchemical transformations
    you get the free energy difference: Delta F (A, G, ...)
    if the `works` are in Kcal and you keep the default `boltzmann_kappa`
    = 0.001985875 kcal/(mol⋅K) the result will be in Kcal/mol
    otherwise it depends on your choice

    uses expectation maximization to fit the data

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
    n_gaussians : int
        the number of gaussians to use for the fit
    tol : float, optional, default=1.E-6
        the tollerance for the convergence
    max_iterations : int, optional, default=`sys.maxsize`
        maximum number of iterations before raising an Exception

    Returns
    ----------
    free_energy : float
        the value of the free energy
        if you used the default `boltzmann_kappa` and the
        `works` where in Kcal it is in Kcal/mol

    Notes
    ----------
    the formula is (for more info check https://dx.doi.org/10.1021/acs.jctc.0c00634
    and http://dx.doi.org/10.1063/1.4918558)
    math :
    \Delta G = k_b T ln( \sum w_i e^( \mu - \beta \sigma^2 / 2))
    """

    kappa_T = boltzmann_kappa * temperature

    beta = 1. / kappa_T

    def crooks_for_gaussian(mean, var):
        """helper function
        """

        return mean - 0.5 * (var**2) * beta

    em_object = em.EMGauss(works,
                           n_gaussians=n_gaussians,
                           tol=tol,
                           max_iterations=max_iterations)

    gaussians = em_object.fit()

    exponents = np.empty(len(gaussians))

    for i, gaussian in enumerate(gaussians):

        exponents[i] = crooks_for_gaussian(gaussian['mu'], gaussian['sigma'])

    #I do it to avoid over and underflow
    max_exponent = np.amax(exponents)

    exponents -= max_exponent

    exponents *= (-beta)

    exponents = np.exp(exponents)

    for i, gaussian in enumerate(gaussians):

        exponents[i] *= gaussian['lambda']

    delta_G = np.sum(exponents)

    del exponents

    delta_G = (-kappa_T) * math.log(delta_G) + max_exponent

    return delta_G
