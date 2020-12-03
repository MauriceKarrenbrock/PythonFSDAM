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

        m_min = mocker.patch('numpy.min', return_value=1.)
        m_sum = mocker.patch('numpy.sum')
        m_exp = mocker.patch('numpy.exp')
        m_log = mocker.patch('math.log')

        works = np.array([1., 2.])

        free.jarzynski_free_energy(works)

        m_min.assert_called_once_with(works)

        m_sum.assert_called_once()

        m_exp.assert_called_once()

        m_log.assert_called()


class Testvolume_correction():
    def test_calls(self, mocker):

        fake_hist_return = (np.array([0.5, 0.5]), np.array([0.1, 0.2]))

        m_hist = mocker.patch(
            'PythonFSDAM.work_probability.make_probability_histogram',
            return_value=fake_hist_return)

        m_std = mocker.patch('numpy.std', return_value=1.)

        m_log = mocker.patch('math.log', return_value=1.)

        mocker.patch('math.pi', 1.)

        dist_values = np.array([1., 2.])

        free.volume_correction(dist_values)

        m_hist.assert_called_once_with(dist_values, 0.1)

        m_std.assert_called_once_with(fake_hist_return[0])

        m_log.assert_called_once()
