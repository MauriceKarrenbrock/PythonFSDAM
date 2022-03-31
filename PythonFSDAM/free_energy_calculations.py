# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Functions and classes to do free energy calculations
"""
import math
import sys

import numpy as np
from scipy.stats import norm

import PythonFSDAM.bootstrapping as boot
import PythonFSDAM.combine_works as combine
import PythonFSDAM.exp_max_gaussmix as em


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


def weighted_jarzynski_free_energy(works,
                                   weights,
                                   temperature=298.15,
                                   boltzmann_kappa=0.001985875):
    """Jarzynki free energy of weighted works

    if for some reason you have a weight you want to
    give to your works, like when you use results produced
    by diferent MD programs this function lets you do a
    wheighted Jarzynski exponential average

    this function uses the `jarzynski_free_energy` function
    if you change the implementation of that function you need to
    change this function too!!!!

    Parameters
    --------------
    weights : numpy.array
        the weights you want to give to the work values
        they must be in the same order of the works
        and in the same number
    for all other parameters check `jarzynski_free_energy`
    """

    weighted_works = works * weights

    free_energy = jarzynski_free_energy(works=weighted_works,
                                        temperature=temperature,
                                        boltzmann_kappa=boltzmann_kappa)

    log_total_weight = math.log(np.sum(weights))

    log_N = math.log(works.size)

    free_energy += temperature * boltzmann_kappa * (log_total_weight - log_N)

    return free_energy


def jarzynski_exponential_average(works,
                                  temperature=298.15,
                                  boltzmann_kappa=0.001985875):
    """Calculates the Jarzynski exponential bias (not the free energy)

    Parameters
    --------------
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
    --------
    float
        The exponential bias (it is a pure number)
    """
    # To avoid overflow it is done as
    # (1/N * exp(beta * Wmin)) SUM exp(-beta(Wi - Wmin))

    w_min = np.amin(works)

    beta = 1 / (boltzmann_kappa * temperature)

    exp_avg = works - w_min
    exp_avg *= -beta
    exp_avg = np.exp(exp_avg)
    exp_avg = np.mean(exp_avg)
    exp_avg *= math.exp(beta * w_min)

    return exp_avg


def jarzynski_bias_estimation(work_std,
                              n_work_values,
                              temperature=298.15,
                              boltzmann_kappa=0.001985875,
                              n_generated_distributions=100):
    """Estimates the bias of the Jarzynski free energy value

    If the `work_std` is in Kcal and you keep the default `boltzmann_kappa`
    = 0.001985875 kcal/(mol⋅K) the result will be in Kcal/mol
    otherwise it depends on your choice

    The bias is always positive

    This estimation is valid for normal work distributions
    the less normal they are the less this estimate will be
    meaningfull

    Parameters
    -----------
    work_std : float
        the standard deviation of your work distribution
        if you don't modify `boltzmann_kappa`
        it should be Kcal/mol (1 Kcal = 1/4.148 KJ)
    n_work_values : int
        how many work values you have in your distribution
    temperature : float, optional
        the temperature in Kelvin at which the non equilibrium simulation
        has been done, default 298.15 K
    boltzmann_kappa : float
        the Boltzmann constant, the dafault is 0.001985875 kcal/(mol⋅K)
        and if you keep it the `works` shall be in Kcal
    n_generated_distributions : int, optional, default=100
        how many random normal distributions to generate

    Returns
    ----------
    bias_estimation : float
        if you used the default `boltzmann_kappa` and the
        `work_std` where in Kcal it is in Kcal/mol

    References
    --------------------
    SAMPL9 blind predictions using
    nonequilibrium alchemical approaches
    Piero Procacci and Guido Guarnieri

    https://arxiv.org/abs/2202.06720

    equation 11
    """

    exp_averages = np.empty(n_generated_distributions)

    beta = 1 / (boltzmann_kappa * temperature)

    for i in range(n_generated_distributions):
        # Mean=0 STD=work_std
        random_normal_points = norm.rvs(loc=0,
                                        scale=work_std,
                                        size=n_work_values)

        # This is Jarzynski exponential average
        # I have put it in line to try to use less memory
        # To avoid un-needed copies
        # Because I am using too much memory otherwise
        w_min = np.amin(random_normal_points)

        np.subtract(random_normal_points, w_min, out=random_normal_points)
        np.multiply(random_normal_points, -beta, out=random_normal_points)
        np.exp(random_normal_points, out=random_normal_points)
        random_normal_points = np.mean(random_normal_points)
        random_normal_points *= math.exp(beta * w_min)

        exp_averages[i] = random_normal_points

    variance = work_std * work_std

    return np.mean(exp_averages) + (variance * 0.5)


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
    bin_width : float, optional, default=0.1
        the with (more or less) that the probability histogram shall have
        don't change it if you are not 100% sure of what you are doing

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
    standard_volume = 1661.0

    #kcal/mol
    R_gas = 1.9872036e-3

    STD = np.std(distance_values)

    site_volume = (4. / 3.) * math.pi * (2 * STD)**3

    Delta_G_vol = R_gas * temperature * math.log(site_volume / standard_volume)

    return Delta_G_vol


# by using this class I can choose the temp and K inside the bootstrapping function
# had to put it outside the function to allow the use of multiprocessing
class _vDSSBJarzynskiErrorPropagationHelperClass(object):
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


def vDSSB_jarzynski_error_propagation(works_1,
                                      works_2,
                                      *,
                                      temperature=298.15,
                                      boltzmann_kappa=0.001985875,
                                      num_iterations=None):
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
    num_iterations : ins, optional, default=sys.maxsize/10
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
    if num_iterations is None:
        num_iterations = sys.maxsize // 10

    helper_obj = _vDSSBJarzynskiErrorPropagationHelperClass(
        temperature=temperature, boltzmann_kappa=boltzmann_kappa)

    out_mean, out_std = boot.mix_and_bootstrap(
        works_1,
        works_2,
        mixing_function=combine.combine_non_correlated_works,
        stat_function=helper_obj.calculate_jarzynski,
        num_iterations=num_iterations)

    return out_std, out_mean


def plain_jarzynski_error_propagation(works,
                                      *,
                                      temperature=298.15,
                                      boltzmann_kappa=0.001985875,
                                      num_iterations=10000):
    """jarzynski STD via bootstrap

    it doesn't use the vDSSB aproach, but only the normal
    one, for the rest it is 100% equal to `vDSSB_jarzynski_error_propagation`

    Parameters
    -------------
    works : numpy.array
        the work values
    temperature : float, optional
        the temperature in Kelvin at which the non equilibrium simulation
        has been done, default 298.15 K
    boltzmann_kappa : float
        the Boltzmann constant, the dafault is 0.001985875 kcal/(mol⋅K)
        and if you keep it the `works` shall be in Kcal
    num_iterations : ins, optional, default=10000
        the number of bootstrapping iterations (time and memory consuming)
    """

    helper_obj = _vDSSBJarzynskiErrorPropagationHelperClass(
        temperature=temperature, boltzmann_kappa=boltzmann_kappa)

    out_mean, out_std = boot.bootstrap_std_of_function_results(
        works,
        function=helper_obj.calculate_jarzynski,
        num_iterations=num_iterations)

    return out_std, out_mean


def crooks_free_energy_for_gaussian_distributions(mean,
                                                  std,
                                                  temperature=298.15,
                                                  boltzmann_kappa=0.001985875):
    """Unidirectional Crooks free energy for gaussian distributions

        Parameters
        -----------
        mean : float
            the mean of the gaussian distribution
        std : float
            the standard deviation of the gaussian distribution
        temperature : float, optional
            the temperature in Kelvin at which the non equilibrium simulation
            has been done, default 298.15 K
        boltzmann_kappa : float
            the Boltzmann constant, the dafault is 0.001985875 kcal/(mol⋅K)
            and if you keep it the `works` shall be in Kcal

        Returns
        ---------
        float

        Notes
        ---------
        The calculation done is mean - 0.5 * (std**2) * (1/kb*T)
        """
    kappa_T = boltzmann_kappa * temperature

    beta = 1. / kappa_T

    return mean - 0.5 * (std**2) * beta


def gaussian_mixtures_free_energy(works,
                                  temperature=298.15,
                                  boltzmann_kappa=0.001985875,
                                  n_gaussians=3,
                                  tol=1.E-6,
                                  max_iterations=None,
                                  gaussians=None):
    """Calculates the free energy with the gaussian mixtures method

    starting from the non equilibrium works obtained
    from for example alchemical transformations
    you get the free energy difference: Delta F (A, G, ...)
    if the `works` are in Kcal and you keep the default `boltzmann_kappa`
    = 0.001985875 kcal/(mol⋅K) the result will be in Kcal/mol
    otherwise it depends on your choice

    uses expectation maximization to fit the data

    DOES NOT DO A VOLUME CORRECTION NOR AN ERROR ESTIMATE!

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
        the number gaussians to use for the fit
    tol : float, optional, default=1.E-6
        the tollerance for the convergence
    max_iterations : int, optional, default=`sys.maxsize`
        maximum number of iterations for each time it tries to
        fit with a certain number of gaussians
    gaussians : list(dict), optional, default=None
        the gaussians to use to start the fitting, len(gaussians)
        must be equal to n_gaussians, if not given a defaut will be used,
        the format of the gaussian_dictionary must be {'mu': float, 'sigma': float}

    Returns
    ----------
    free_energy : float
        the value of the free energy
        if you used the default `boltzmann_kappa` and the
        `works` are in Kcal it is in Kcal/mol
    gaussians : list(dict)
        a list of dicts containing the parameters of the `n_gaussians` gaussians
        the parameters are (and keys of the dicts):
        'mu' = the mean (float)
        'sigma' = the sigma (std) of the gaussian (float)
        'lambda = the normalized coefficient of the gaussian (float)
    log_likelyhood : float
        the logarithm of the total likelihood


    Raises
    ------------
    RuntimeError
        if it is not possible to converge with the given `n_gaussians`

    Notes
    ----------
    the formula is (for more info check https://dx.doi.org/10.1021/acs.jctc.0c00634
    and http://dx.doi.org/10.1063/1.4918558)
    math :
    \Delta G = k_b T ln( \sum w_i e^( \mu - \beta \sigma^2 / 2))
    """

    kappa_T = boltzmann_kappa * temperature

    beta = 1. / kappa_T

    if gaussians is None:
        gaussians = []

    def crooks_for_gaussian(mean, var):
        """helper function

        kept in order not to modify the rest of the function
        when I extracted it from here
        """
        return crooks_free_energy_for_gaussian_distributions(
            mean=mean,
            std=var,
            temperature=temperature,
            boltzmann_kappa=boltzmann_kappa)

    try:

        em_object = em.EMGauss(works,
                               n_gaussians=n_gaussians,
                               tol=tol,
                               max_iterations=max_iterations)

        em_object.set_starting_gaussians(gaussians)

    except ValueError:

        em_object = em.EMGauss(works,
                               n_gaussians=n_gaussians,
                               tol=tol,
                               max_iterations=max_iterations)

    gaussians, log_likelyhood = em_object.fit()

    exponents = np.empty(len(gaussians))

    for j, gaussian in enumerate(gaussians):

        exponents[j] = crooks_for_gaussian(gaussian['mu'], gaussian['sigma'])

    exponents *= (-beta)

    exponents = np.exp(exponents)

    for j, gaussian in enumerate(gaussians):

        exponents[j] *= gaussian['lambda']

    delta_G = np.sum(exponents)

    del exponents

    delta_G = (-kappa_T) * math.log(delta_G)

    return delta_G, gaussians, log_likelyhood


# by using this class I can choose the temp and K inside the bootstrapping function
# had to put it outside the function to allow the use of multiprocessing
class _vDSSBGaussianMixturesErrorPropagationHelperClass(object):
    """helper class
    """

    def __init__(self,
                 temperature,
                 boltzmann_kappa,
                 n_gaussians,
                 tol,
                 max_iterations,
                 gaussians=None):

        self.temperature = temperature
        self.boltzmann_kappa = boltzmann_kappa
        self.n_gaussians = n_gaussians
        self.tol = tol
        self.max_iterations = max_iterations
        self.gaussians = gaussians

        self.log_likelyhood = []

    def calculate_gaussian_mixtures(self, values):
        """helper method
        """

        energy, self.gaussians, log_likelyhood = gaussian_mixtures_free_energy(
            values,
            temperature=self.temperature,
            boltzmann_kappa=self.boltzmann_kappa,
            n_gaussians=self.n_gaussians,
            tol=self.tol,
            max_iterations=self.max_iterations,
            gaussians=self.gaussians)

        self.log_likelyhood.append(log_likelyhood)

        return energy


def VDSSB_gaussian_mixtures_error_propagation(works_1,
                                              works_2,
                                              *,
                                              temperature=298.15,
                                              boltzmann_kappa=0.001985875,
                                              num_iterations=50,
                                              n_gaussians=3,
                                              tol=1.E-6,
                                              max_iterations=20000):
    """get STD of gaussian mixture free energy obtained by convolution of bound & unbound work

    uses vDSSB approach
    starting from the bound and unbound work values calculates the STD of the gaussian mixtures
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
    num_iterations : ins, optional, default=50
        the number of bootstrapping iterations (time and memory consuming)
    n_gaussians
        check `gaussian_mixtures_free_energy`
    tol
        check `gaussian_mixtures_free_energy`
    max_iterations
        check `gaussian_mixtures_free_energy`

    Returns
    ----------
    STD, mean, mean_log_likelyhood : float, float, float
        the STD of the Gaussian extimate estimate, it's bootstrapped mean value,
        and the mean of the log_likelyhood

    Notes
    -----------
    it uses the `gaussian_mixtures_free_energy` present in this module, it uses bootstrapping,
    will take for granted that you mixed the work values with
    `PythonFSDAM.combine_works.combine_non_correlated_works` and uses the
    `PythonFSDAM.bootstrapping.mix_and_bootstrap` function to do the bootstrapping
    """

    helper_obj = _vDSSBGaussianMixturesErrorPropagationHelperClass(
        temperature=temperature,
        boltzmann_kappa=boltzmann_kappa,
        n_gaussians=n_gaussians,
        tol=tol,
        max_iterations=max_iterations)

    out_mean, out_std = boot.mix_and_bootstrap(
        works_1,
        works_2,
        mixing_function=combine.combine_non_correlated_works,
        stat_function=helper_obj.calculate_gaussian_mixtures,
        num_iterations=num_iterations)

    # Get the mean of the log_likelyhood
    mean_log_likelyhood = np.mean(np.array(helper_obj.log_likelyhood))

    return out_std, out_mean, mean_log_likelyhood


def plain_gaussian_mixtures_error_propagation(works,
                                              *,
                                              temperature=298.15,
                                              boltzmann_kappa=0.001985875,
                                              num_iterations=10000,
                                              n_gaussians=3,
                                              tol=1.E-6,
                                              max_iterations=20000):
    """gaussian mixstures STD via bootstrap

    it doesn't use the vDSSB aproach, but only the normal
    one, for the rest it is 100% equal to `vDSSB_gaussian_mixtures_error_propagation`

    Parameters
    -------------
    works : numpy.array
        the work values
    temperature : float, optional
        the temperature in Kelvin at which the non equilibrium simulation
        has been done, default 298.15 K
    boltzmann_kappa : float
        the Boltzmann constant, the dafault is 0.001985875 kcal/(mol⋅K)
        and if you keep it the `works` shall be in Kcal
    num_iterations : ins, optional, default=10000
        the number of bootstrapping iterations (time and memory consuming)
    n_gaussians
        check `gaussian_mixtures_free_energy`
    tol
        check `gaussian_mixtures_free_energy`
    max_iterations
        check `gaussian_mixtures_free_energy`

    Returns
    ----------
    STD, mean, mean_log_likelyhood : float, float, float
        the STD of the Gaussian extimate estimate, it's bootstrapped mean value,
        and the mean of the log_likelyhood
    """

    helper_obj = _vDSSBGaussianMixturesErrorPropagationHelperClass(
        temperature=temperature,
        boltzmann_kappa=boltzmann_kappa,
        n_gaussians=n_gaussians,
        tol=tol,
        max_iterations=max_iterations)

    out_mean, out_std = boot.bootstrap_std_of_function_results(
        works,
        function=helper_obj.calculate_gaussian_mixtures,
        num_iterations=num_iterations)

    # Get the mean of the log_likelyhood
    mean_log_likelyhood = np.mean(np.array(helper_obj.log_likelyhood))

    return out_std, out_mean, mean_log_likelyhood
