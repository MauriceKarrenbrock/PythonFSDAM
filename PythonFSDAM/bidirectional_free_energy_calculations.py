# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Functions and classes to do bidirectional (Crooks) free energy calculations
"""

import sys

import numpy as np
import pymbar.other_estimators as _pymbar_other

import PythonFSDAM.bootstrapping as boot
import PythonFSDAM.combine_works as combine
import PythonFSDAM.exp_max_gaussmix as em


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
    works_2 : numpy.array
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
    """

    # Values must be in units of kT
    works_1 /= temperature * boltzmann_kappa
    works_2 /= temperature * boltzmann_kappa

    # If the deafult method fails will try the self-consistent-iteration one
    try:
        free_energy = _pymbar_other.bar(works_1,
                                        works_2,
                                        maximum_iterations=10000)

    except Exception:  # pylint: disable=broad-except
        free_energy = _pymbar_other.bar(works_1,
                                        works_2,
                                        maximum_iterations=10000,
                                        method='self-consistent-iteration')

    # _pymbar_other.bar returns a dict with the DG and the uncertainty
    # I only care about the free energy
    free_energy = free_energy['Delta_f']

    # Back to the right units (default=Kcal/mol)
    free_energy *= temperature * boltzmann_kappa

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


def crossing_of_2_gaussians(mu1, mu2, sigma1, sigma2):
    """Calsculate the crossing point(s) of 2 gaussians

    Parameters
    -----------
    mu1 : float
        mean of gaussian 1
    mu2 : float
        mean of gaussian 2
    sigma1 : float
        std of gaussian 1
    sigma2 : float
        std of gaussian 2

    Returns
    ---------
    np.array
        the length depends on the number of real roots of
        the quadratic equation
    """
    a = 1 / (2 * sigma1**2) - 1 / (2 * sigma2**2)
    b = mu2 / (sigma2**2) - mu1 / (sigma1**2)
    c = mu1**2 / (2 * sigma1**2) - mu2**2 / (2 * sigma2**2) - np.log(
        sigma2 / sigma1)
    return np.roots([a, b, c])


def crooks_gaussian_crossing_free_energy(works_1,
                                         works_2,
                                         max_iterations=None):
    """Calculates the Crooks free energy

    It fits the probability distributions of `works_1` and
    -`works_2` with one gaussian with the expectation maximization
    algorithm and finds where they cross

    Parameters
    -----------
    works_1 : np.array
        1-D numpy array of work values
    works_2 : np.array
        1-D numpy array of work values
        the sign will be changed
    max_iterations : default=sys.maxsize

    Returns:
    ----------
    free_energy : float

    Raises
    --------
    RuntimeError
        if there are no real roots to the equation
    """

    em_obj = em.EMGauss(works_1, n_gaussians=1, max_iterations=max_iterations)

    #gaussians_* are lists of dict:
    # [{'sigma': 0, 'mu': 0, 'lambda': 0}, ...]
    # where 'lambda' is the probability of the gaussian N_gauss / N_tot (in this case = 1)
    gaussian_1, _ = em_obj.fit()
    gaussian_1 = gaussian_1[0]

    em_obj = em.EMGauss(-works_2, n_gaussians=1, max_iterations=max_iterations)

    gaussian_2, _ = em_obj.fit()
    gaussian_2 = gaussian_2[0]

    crossing_points = crossing_of_2_gaussians(gaussian_1['mu'],
                                              gaussian_2['mu'],
                                              gaussian_1['sigma'],
                                              gaussian_2['sigma'])

    # If there are 2 roots I want the one closest to the mean of the means
    if len(crossing_points) == 2:
        mean_of_means = np.mean(np.array([gaussian_1['mu'], gaussian_2['mu']]))

        d0 = abs(crossing_points[0] - mean_of_means)
        d1 = abs(crossing_points[1] - mean_of_means)

        if d0 < d1:
            return crossing_points[0]
        return crossing_points[1]

    # No real roots
    if len(crossing_points) == 0:
        raise RuntimeError(
            'There are no crossing points between the 2 gaussians')

    # There is only one root
    return crossing_points[0]


# by using this class I can choose max_iterations inside the bootstrapping function
# had to put it outside the function to allow the use of multiprocessing
class _vDSSBCrooksGaussianCrossingErrorPropagationHelperClass(object):
    """helper class
    """

    def __init__(self, max_iterations):

        self.max_iterations = max_iterations

    def calculate_crooks_gaussian_crossing(self, works_1, works_2):
        """helper method
        """

        return crooks_gaussian_crossing_free_energy(
            works_1, works_2, max_iterations=self.max_iterations)


def vDSSB_crooks_gaussian_crossing_error_propagation(bound_works_1,
                                                     unbound_works_1,
                                                     bound_works_2,
                                                     unbound_works_2,
                                                     *,
                                                     num_iterations=10000,
                                                     max_iterations=None):
    """get STD of the crooks gaussian crossing free energy obtained by convolution
    of bound and unbound work vDSSB

    starting from the bound and unbound work values calculates the STD of the
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
    num_iterations : ins, optional, default=1000
        the number of bootstrapping iterations (time and memory consuming)
    max_iterations, default=sys.maxsize // 10
        the max number of iterations for the gaussian EM fitting

    Returns
    ----------
    STD, mean : float, float
        the STD of the free energy estimate and it's bootstrapped mean value

    Notes
    -----------
    it uses the `bar_free_energy` present in this module
    """

    if max_iterations is None:
        max_iterations = sys.maxsize // 10

    helper_obj = _vDSSBCrooksGaussianCrossingErrorPropagationHelperClass(
        max_iterations)

    out_mean, out_std = boot.mix_and_bootstrap_multiple_couples_of_values(
        ((bound_works_1, unbound_works_1), (bound_works_2, unbound_works_2)),
        mixing_function=combine.combine_non_correlated_works,
        stat_function=helper_obj.calculate_crooks_gaussian_crossing,
        num_iterations=num_iterations)

    return out_std, out_mean


def plain_crooks_gaussian_crossing_error_propagation(works_1,
                                                     works_2,
                                                     *,
                                                     num_iterations=10000,
                                                     max_iterations=None):
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
    max_iterations, default=sys.maxsize // 10
        the max number of iterations for the gaussian EM fitting

    Returns
    ----------
    STD, mean : float, float
        the STD of the free energy estimate and it's bootstrapped mean value
    """

    if max_iterations is None:
        max_iterations = sys.maxsize // 10

    helper_obj = _vDSSBCrooksGaussianCrossingErrorPropagationHelperClass(
        max_iterations)

    tot_works = works_1, works_2

    out_mean, out_std = boot.bootstrap_std_of_function_results(
        tot_works,
        function=helper_obj.calculate_crooks_gaussian_crossing,
        num_iterations=num_iterations,
        number_of_sets_of_values=len(tot_works))

    return out_std, out_mean
