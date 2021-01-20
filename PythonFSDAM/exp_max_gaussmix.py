# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Use the expectation maximization algorithm for gaussian mixtures
"""

import math
import random
from collections import Counter
from sys import maxsize

import numpy as np
import scipy.stats as stats


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
        maximum number of iterations before raising an Exception

    Methods
    ----------
    fit()
        fits the data with the gaussians
    get_labels()
        returns the label array
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

        self._labels = np.zeros(len(values), dtype=np.int8)

    def get_labels(self):
        """returns the labels np.array created whith `fit`
        """

        return self._labels

    @staticmethod
    def _probability(value, gaussian):
        """Private

        Parameters
        -----------
        value : float
        gaussian : dict

        Returns
        ----------
        float
        """

        p = stats.norm(gaussian['mu'], gaussian['sigma']).pdf(value)

        p *= gaussian['lambda']

        return p

    def _expectation(self):
        """private
        """

        probabilities = np.empty(len(self._gaussians))

        for i, value in enumerate(self.values):

            for j, gaussian in enumerate(self._gaussians):

                probabilities[j] = self._probability(value, gaussian)

            self._labels[i] = np.argmax(probabilities)

    def _maximization(self):
        """private
        """

        number_of_values_per_gaussian = Counter(self._labels)

        tot_number_of_values = len(self.values)
        for key, number_of_values in number_of_values_per_gaussian.items():

            #get new probability (lambda for the gaussian)
            self._gaussians[key]['lambda'] = float(
                number_of_values) / tot_number_of_values

            #get new mean and sigma for the gaussian
            #might be a very slow implementation
            # TODO find something better
            tmp_array = np.empty(number_of_values)

            j = 0
            iterator = np.nditer(self._labels)
            for label in iterator:

                if label == key:

                    tmp_array[j] = self.values[iterator.index]

                    j += 1

            self._gaussians[key]['mu'] = np.mean(tmp_array)
            self._gaussians[key]['sigma'] = np.std(tmp_array)

    def _determine_convergence(self, log_new_lambdas, log_old_lambdas):
        """private
        """

        dist = 0

        for i, j in zip(log_new_lambdas, log_old_lambdas):

            dist += (i - j)**2

        dist = dist**0.5

        if dist < self.tol:

            return True

        return False

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

        Raises
        -----------
        Exception
            if a convergence is not achieved in `max_iterations` iterations
        """

        mu = np.mean(self.values)

        sigma = np.std(self.values)

        log_old_lambdas = []
        #create first guess for gaussians
        for i in range(self.n_gaussians):  # pylint: disable=unused-variable

            self._gaussians.append({
                'mu': mu * random.random(),
                'sigma': sigma * random.random(),
                'lambda': 1. / self.n_gaussians,
            })

            log_old_lambdas.append(float(maxsize))

        del mu
        del sigma

        iteration = 0
        has_converged = False
        while not has_converged:

            if iteration == self.max_iterations - 1:

                raise Exception(
                    f'EM fit did not converge in {self.max_iterations} steps with tol {self.tol}'
                )

            self._expectation()

            self._maximization()

            log_new_lambdas = []
            for gaussian in self._gaussians:

                log_new_lambdas.append(math.log(gaussian['lambda']))

            has_converged = self._determine_convergence(
                log_new_lambdas, log_old_lambdas)

            log_old_lambdas = log_new_lambdas

        return self._gaussians
