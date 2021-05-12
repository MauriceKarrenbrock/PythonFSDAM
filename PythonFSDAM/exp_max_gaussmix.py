# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Use the expectation maximization algorithm for gaussian mixtures
"""

import copy
import math
import random
from sys import maxsize

import numpy as np


class EMGauss(object):
    """Fits the given values with gaussians using expectation-maximization

    works only for 1-D datasets (useful for free energy calculation with
    gaussian mixture)

    Parameters
    -----------
    values : numpy.array
        the values to fit (1-D only!!!)
    n_gaussians : int
        the number of gaussians to use for the fit
    tol : float, optional, default=1.E-6
        the tollerance for the convergence
    max_iterations : int, optional, default=`sys.maxsize`
        maximum number of iterations for each time it tries to fit with a
        certain number of gaussians

    Methods
    ----------
    fit()
        fits the data with the gaussians
    set_starting_gaussians(gaussians)
        you can set some or all the starting gaussians
        for the fitting
    """
    def __init__(self, values, n_gaussians=3, tol=1.E-6, max_iterations=None):

        self.values = values

        self.n_gaussians = n_gaussians

        self.tol = tol

        if max_iterations is None:
            self.max_iterations = maxsize
        else:
            self.max_iterations = max_iterations

        #a list of dict:
        # [{'sigma': 0, 'mu': 0, 'lambda': 0}, ...]
        # where 'lambda' is the probability of the gaussian N_gauss / N_tot
        self._gaussians = []

    def set_starting_gaussians(self, gaussians):
        """give some or all the starting gaussians for the fit

        you can set some of the starting gaussians for the fitting
        the missing ones will be added automaticly

        Parameters
        -------------
        gaussians : list(dicts)
            see constructor documentation

        Raises
        ---------
        ValueError
            for empty or for too long inputs
        """

        if len(gaussians) > self.n_gaussians:

            raise ValueError(
                'The number of given gaussians is bigger than n_gaussians')

        if not gaussians:

            raise ValueError('gaussians is empty')

        self._gaussians += gaussians

    def _create_starting_gaussians(self, mu, sigma):
        """private
        """

        if len(self._gaussians) == self.n_gaussians:

            return self._gaussians

        if not self._gaussians:

            log_lambda = self._append_gaussian(mu, sigma)

        while len(self._gaussians) < self.n_gaussians:

            log_lambda = self._append_gaussian(mu * random.random(),
                                               sigma * random.random() + 1.E-5)

        return log_lambda

    @staticmethod
    def _probability(values, gaussian):
        """Private

        Parameters
        -----------
        value : numpy.array of float
        gaussian : dict

        Returns
        ----------
        numpy.array of float
        """

        norm = 1. / (gaussian['sigma'] * (2 * math.pi)**0.5)

        _values = values - gaussian['mu']

        _values = _values**2

        _values = -_values / (2. * gaussian['sigma']**2)

        _values = np.exp(_values)

        _values *= norm * gaussian['lambda']

        return _values

    def _get_weights_matrix(self):
        """private
        """

        weights_matrix = np.empty([len(self._gaussians), len(self.values)])

        for i, gaussian in enumerate(self._gaussians):

            probability = self._probability(self.values, gaussian)

            weights_matrix[i, :] = probability[:]

        return weights_matrix

    @staticmethod
    def _normalize_weights_matrix(weights_matrix):
        """private
        """

        normalization_value = np.sum(weights_matrix, axis=0)

        normalization_value = np.where(normalization_value > 0.,
                                       normalization_value, 1.E-10)

        #normalize weights_matrix
        for i in range(weights_matrix.shape[0]):

            weights_matrix[i, :] /= normalization_value[:]

        return weights_matrix

    def _expectation_maximization(self, weights_matrix=None):
        """private
        """

        #-----------------------------------------------------
        #expectation
        #-----------------------------------------------------

        #only need to do it at the first iteration
        if weights_matrix is None:

            weights_matrix = self._get_weights_matrix()

        weights_matrix = self._normalize_weights_matrix(weights_matrix)

        number_of_values = len(self.values)

        # a vector with each element equal to np.sum(weights_matrix[i, :])
        n_per_gaussian = np.sum(weights_matrix, axis=1)

        #to avoid division by zero
        n_per_gaussian = np.where(n_per_gaussian > 0., n_per_gaussian, 1.E-10)

        for i in range(len(self._gaussians)):

            self._gaussians[i]['lambda'] = n_per_gaussian[i] / number_of_values

        #-----------------------------------------------------
        #maximization
        #-----------------------------------------------------

        for i in range(len(self._gaussians)):

            tmp_vector = weights_matrix[i, :] * self.values[:]

            self._gaussians[i]['mu'] = np.sum(tmp_vector) / n_per_gaussian[i]

            del tmp_vector

            tmp_sigma = weights_matrix[i, :] * (self.values[:] -
                                                self._gaussians[i]['mu'])**2

            tmp_sigma = np.sum(tmp_sigma)

            self._gaussians[i]['sigma'] = (tmp_sigma / n_per_gaussian[i])**0.5

            del tmp_sigma

            # to avoid sigma = 0
            if self._gaussians[i]['sigma'] < 1.E-5:

                self._gaussians[i]['sigma'] = 1.E-5

        #calculate log likelyhood

        #update weights_matrix with new gaussians (not normalized)
        weights_matrix = self._get_weights_matrix()

        log_lambda = np.sum(weights_matrix, axis=1)

        #be sure not to have zeros (for log)
        log_lambda = np.where(log_lambda > 0., log_lambda, 1.E-10)

        log_lambda = np.log(log_lambda)

        log_lambda = np.sum(log_lambda)

        return log_lambda, weights_matrix

    def _determine_convergence(self, new_log_lambda, old_log_lambda):
        """private
        """

        dlog = abs(old_log_lambda - new_log_lambda)

        if dlog < self.tol:

            return True

        return False

    def _append_gaussian(self, mu, sigma):
        """private
        """

        number_of_gaussians = len(self._gaussians) + 1

        self._gaussians.append({'mu': mu, 'sigma': sigma})

        for i in range(len(self._gaussians)):

            #be sure to have normalized coefficients
            self._gaussians[i]['lambda'] = 1.0 / number_of_gaussians

            #re-create a old_log_lambda
            old_log_lambda = float(maxsize)

        return old_log_lambda

    def fit(self):
        """The main method, fits your dataset with gaussians

        Returns
        -----------
        gaussians : list(dict)
            a list of dicts containing the parameters of the `n_gaussians` gaussians
            the parameters are (and keys of the dicts):
            'mu' = the mean (float)
            'sigma' = the sigma (std) of the gaussian (float)
            'lambda = the normalized coefficient of the gaussian (float)
        log_likelyhood : float
            the logarithm of the total likelihood

        Raises
        -----------
        RuntimeError
            if a convergence is not achieved in `max_iterations` iterations
        """

        mu = np.mean(self.values)

        sigma = np.std(self.values)

        # start with one gaussian
        old_log_lambda = self._create_starting_gaussians(mu, sigma)

        weights_matrix = None

        iteration = 0
        has_converged = False
        while not has_converged:

            if iteration == self.max_iterations - 1:

                raise RuntimeError(
                    f'Could not converge EM gaussian mixing with {self.n_gaussians} gaussians'
                )

            new_log_lambda, weights_matrix = self._expectation_maximization(
                weights_matrix)

            has_converged = self._determine_convergence(
                new_log_lambda, old_log_lambda)

            old_log_lambda = copy.deepcopy(new_log_lambda)

            iteration += 1

        return self._gaussians, new_log_lambda
