# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-self-use
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################

import numpy as np

import PythonFSDAM.bootstrapping as boot


class Teststandard_deviation():
    def test_works(self, mocker):

        m_boot = mocker.patch('bootstrapped.bootstrap.bootstrap',
                              return_value=1.)
        m_std = mocker.patch('bootstrapped.stats_functions.std')

        values = np.array([1., 2., 3.])

        alpha = 0.05
        num_iterations = 10000
        iteration_batch_size = 10
        num_threads = 1

        output = boot.standard_deviation(
            values,
            alpha=alpha,
            num_iterations=num_iterations,
            iteration_batch_size=iteration_batch_size,
            num_threads=num_threads)

        assert output == 1.

        m_boot.assert_called_once_with(
            values=values,
            stat_func=m_std,
            alpha=alpha,
            num_iterations=num_iterations,
            iteration_batch_size=iteration_batch_size,
            num_threads=num_threads)
