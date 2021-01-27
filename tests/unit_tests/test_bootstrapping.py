# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-self-use
# pylint: disable=protected-access
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################

from unittest import mock

import numpy as np
import pytest

import PythonFSDAM.bootstrapping as boot


class Teststandard_deviation():
    def test_works(self, mocker):
        class MockBootResult():
            def __init__(self):

                self.value = 1.

                self.lower_bound = -1.

                self.upper_bound = 2.

        boot_out = MockBootResult()

        m_boot = mocker.patch('bootstrapped.bootstrap.bootstrap',
                              return_value=boot_out)
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

        assert output == (1., (-1., 2.))

        m_boot.assert_called_once_with(
            values=values,
            stat_func=m_std,
            alpha=alpha,
            num_iterations=num_iterations,
            iteration_batch_size=iteration_batch_size,
            num_threads=num_threads)


class Testmix_and_bootstrap():
    def test_valueerror(self):

        with pytest.raises(ValueError):

            boot.mix_and_bootstrap([1], [2], num_iterations=0)

        with pytest.raises(ValueError):

            boot.mix_and_bootstrap([1], [2], num_iterations=-1)

    def test_keywordonly_arguments(self):
        def dum(x):
            return x

        with pytest.raises(TypeError):

            boot.mix_and_bootstrap([1], [2], dum, dum, 10000)  # pylint: disable=too-many-function-args

    def test_works(self, mocker):

        values_1 = np.array([1, 2, 3])

        values_2 = np.array([4, 5, 6])

        m_mixing = mock.MagicMock()
        m_stat = mock.MagicMock()

        m_pool = mocker.patch('multiprocessing.pool.Pool')

        m_pool_enter = mock.MagicMock()

        m_pool_enter.return_value.starmap_async.return_value.get.return_value = [
            1., 2., 3.
        ]

        m_pool.return_value.__enter__.return_value = m_pool_enter

        m_array = mocker.patch('numpy.array', return_value=1.)
        m_mean = mocker.patch('numpy.mean', return_value=1.)
        m_std = mocker.patch('numpy.std', return_value=2.)

        out_mean, out_std = boot.mix_and_bootstrap(values_1,
                                                   values_2,
                                                   mixing_function=m_mixing,
                                                   stat_function=m_stat,
                                                   num_iterations=1)

        assert out_mean == 1.
        assert out_std == 2.

        m_stat.assert_not_called()

        m_array.assert_called_once()

        m_mean.assert_called_once()
        m_std.assert_called_once()


class Test_mix_and_bootstrap_helper_function():
    def test_works(self, mocker):

        m_mixing = mock.MagicMock()
        m_mixing.return_value = [2, 3, 7, 8]

        m_stat = mock.MagicMock()
        m_stat.return_value = 1.

        m_random_generator = mock.MagicMock()
        m_random_generator.choice.side_effect = [[7, 8, 9], [10, 11, 12]]

        m_random_factory = mocker.patch('numpy.random.default_rng',
                                        return_value=m_random_generator)

        output = boot._mix_and_bootstrap_helper_function(
            np.array([1, 2]),
            np.array([3, 4]),
            mixing_function=m_mixing,
            stat_function=m_stat)

        m_mixing.assert_called_once_with([7, 8, 9], [10, 11, 12])

        m_stat.assert_called_once_with([2, 3, 7, 8])

        m_random_factory.assert_called_once()

        m_random_generator.choice.assert_called()

        assert output == 1.
