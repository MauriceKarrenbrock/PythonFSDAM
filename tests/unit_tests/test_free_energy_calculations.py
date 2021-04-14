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

import PythonFSDAM.free_energy_calculations as free


class Testjarzynski_free_energy():
    def test_calls(self, mocker):

        m_min = mocker.patch('numpy.amin', return_value=1.)
        m_sum = mocker.patch('numpy.sum')
        m_exp = mocker.patch('numpy.exp')
        m_log = mocker.patch('math.log')

        works = np.array([1., 2.])

        free.jarzynski_free_energy(works)

        m_min.assert_called_once_with(works)

        m_sum.assert_called_once()

        m_exp.assert_called_once()

        m_log.assert_called()


class Testweighted_jarzynski_free_energy():
    def test_works(self, mocker):

        works = np.zeros(5)
        works += 2.
        weights = np.zeros(5)
        weights += 0.5

        m_jarzynski = mocker.patch(
            'PythonFSDAM.free_energy_calculations.jarzynski_free_energy',
            return_value=2.)

        output = free.weighted_jarzynski_free_energy(works, weights, 0., 1.)

        assert output == 2.

        m_jarzynski.assert_called_once()


class Testvolume_correction():
    def test_calls(self, mocker):

        m_std = mocker.patch('numpy.std', return_value=1.)

        m_log = mocker.patch('math.log', return_value=1.)

        mocker.patch('math.pi', 1.)

        dist_values = np.array([1., 2.])

        free.volume_correction(dist_values)

        m_std.assert_called_once()

        m_log.assert_called_once()


class TestvDSSB_jarzynski_error_propagation():
    def test_works(self, mocker):

        m_boot = mocker.patch('PythonFSDAM.bootstrapping.mix_and_bootstrap',
                              return_value=(1, 2))

        mocker.patch(
            'PythonFSDAM.free_energy_calculations.jarzynski_free_energy')

        output = free.vDSSB_jarzynski_error_propagation([1], [2])

        assert output == (2, 1)

        m_boot.assert_called_once()


class Testplain_jarzynski_error_propagation():
    def test_works(self, mocker):

        m_heper = mocker.patch(
            'PythonFSDAM.free_energy_calculations._vDSSBJarzynskiErrorPropagationHelperClass'
        )

        m_std = mocker.patch(
            'PythonFSDAM.bootstrapping.bootstrap_std_of_function_results',
            return_value=(1, 2))

        assert free.plain_jarzynski_error_propagation(np.array([3.,
                                                                4.])) == (2, 1)

        m_heper.assert_called_once()

        m_std.assert_called_once()
