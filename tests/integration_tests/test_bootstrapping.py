# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-self-use
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################

import numpy as np
import pytest
from scipy.stats import norm

import PythonFSDAM.bidirectional_free_energy_calculations as bidir_free
import PythonFSDAM.bootstrapping as boot
import PythonFSDAM.combine_works as combine
import PythonFSDAM.free_energy_calculations as free

#------------------------------------------------------------
#some helper functions that I need to define in the global
#scope because I am using multiprocessing
#------------------------------------------------------------


def simple_mix(x, y):

    return x + y


def dummy_calculation(x):

    return np.mean(x)


def dummy_calculation_many_inputs(*args):
    output = []
    for arg in args:
        output.append(np.mean(arg))
    return sum(output)


class Helper():

    def __init__(self, temperature, boltzmann_kappa):

        self.temperature = temperature
        self.boltzmann_kappa = boltzmann_kappa

    def calculate_jarzynski(self, values):

        return free.jarzynski_free_energy(values,
                                          temperature=self.temperature,
                                          boltzmann_kappa=self.boltzmann_kappa)

    def calculate_bar(self, *args):
        return bidir_free.bar_free_energy(*args,
                                          temperature=self.temperature,
                                          boltzmann_kappa=self.boltzmann_kappa)


#------------------------------------------------------------
#END
#------------------------------------------------------------


class Testmix_and_bootstrap():

    def test_with_simple_function(self):

        values_1 = np.array([1, 2, 3])
        values_2 = np.array([4, 5, 6])

        expected_mean = np.mean(values_1 + values_2)
        expected_std = np.std(values_1 + values_2)

        out_mean, out_std = boot.mix_and_bootstrap(
            values_1,
            values_2,
            mixing_function=simple_mix,
            stat_function=dummy_calculation,
            num_iterations=10000)

        assert np.testing.assert_allclose(out_mean, expected_mean,
                                          rtol=0.003) is None

        #this assertion may fail
        assert np.testing.assert_allclose(out_std, expected_std,
                                          rtol=0.6) is None

    def test_non_correlated_mixing_and_jarzynski(self):

        values_1 = np.array([-1, -2, -3, 1, 2, 3])
        values_2 = np.array([-4, -5, -6, 4, 5, 6])

        #calculated in a previous run
        expected_mean = -6.463986
        #expected_std = 1.519654

        out_mean, _ = boot.mix_and_bootstrap(
            values_1,
            values_2,
            mixing_function=combine.combine_non_correlated_works,
            stat_function=free.jarzynski_free_energy,
            num_iterations=10000)

        assert np.testing.assert_allclose(out_mean, expected_mean,
                                          rtol=0.01) is None

        #to unstable does always fail (shall do more iterations but would be to time expensive)
        #assert np.testing.assert_allclose(out_std, expected_std, rtol=0.03) is None

    def test_with_helper_class(self):

        helper_obj = Helper(temperature=100., boltzmann_kappa=0.1)

        values_1 = np.array([-1, -2, -3, 1, 2, 3])
        values_2 = np.array([-4, -5, -6, 4, 5, 6])

        #calculated in a previous run
        expected_mean = -1.20363
        #expected_std = 2.190256

        out_mean, _ = boot.mix_and_bootstrap(
            values_1,
            values_2,
            mixing_function=combine.combine_non_correlated_works,
            stat_function=helper_obj.calculate_jarzynski,
            num_iterations=10000)

        assert np.testing.assert_allclose(out_mean, expected_mean,
                                          rtol=0.07) is None

        #to unstable does always fail (shall do more iterations but would be to time expensive)
        #assert np.testing.assert_allclose(out_std, expected_std, rtol=0.6) is None


class Test_mix_and_bootstrap_multiple_couples_of_values():

    def test_with_simple_function(self):

        couples = [[np.array([1, 2, 3]),
                    np.array([4, 5, 6])],
                   [np.array([1, 2, 3]),
                    np.array([4, 5, 6])]]

        expected_mean = np.mean(np.array([1, 2, 3]) + np.array([4, 5, 6])) * 2
        expected_std = 0.95

        out_mean, out_std = boot.mix_and_bootstrap_multiple_couples_of_values(
            couples,
            mixing_function=simple_mix,
            stat_function=dummy_calculation_many_inputs,
            num_iterations=10000)

        assert out_mean == pytest.approx(expected_mean, abs=0.1)

        #this assertion may fail
        assert out_std == pytest.approx(expected_std, abs=0.1)

    def test_non_correlated_mixing_and_bar(self):

        random_normal_points_1 = norm.rvs(loc=0, scale=1, size=100)

        random_normal_points_2 = norm.rvs(loc=0, scale=1, size=100)

        couples = [[
            random_normal_points_1,
            np.zeros(random_normal_points_1.shape)
        ], [random_normal_points_2,
            np.zeros(random_normal_points_2.shape)]]

        #calculated in a previous run
        expected_mean = -0.1
        #expected_std = 1.519654

        out_mean, _ = boot.mix_and_bootstrap_multiple_couples_of_values(
            couples,
            mixing_function=combine.combine_non_correlated_works,
            stat_function=bidir_free.bar_free_energy,
            num_iterations=10000)

        assert out_mean == pytest.approx(expected_mean, abs=0.5)

        #to unstable does always fail (shall do more iterations but would be to time expensive)
        #assert np.testing.assert_allclose(out_std, expected_std, rtol=0.03) is None

    def test_with_helper_class(self):

        helper_obj = Helper(temperature=100., boltzmann_kappa=0.1)

        random_normal_points_1 = norm.rvs(loc=0, scale=1, size=100)

        random_normal_points_2 = norm.rvs(loc=0, scale=1, size=100)

        couples = [[
            random_normal_points_1,
            np.zeros(random_normal_points_1.shape)
        ], [random_normal_points_2,
            np.zeros(random_normal_points_2.shape)]]

        #calculated in a previous run
        expected_mean = -0.1
        #expected_std = 2.190256

        out_mean, _ = boot.mix_and_bootstrap_multiple_couples_of_values(
            couples,
            mixing_function=combine.combine_non_correlated_works,
            stat_function=helper_obj.calculate_bar,
            num_iterations=10000)

        assert out_mean == pytest.approx(expected_mean, abs=0.5)


class Test_bootstrap_std_of_function_results():

    def test_one_set_of_values(self):

        values = np.array([1, 2, 3])

        out_mean, _ = boot.bootstrap_std_of_function_results(
            values,
            function=dummy_calculation_many_inputs,
            num_iterations=10000,
            n_threads=None,
            number_of_sets_of_values=1)

        assert out_mean == pytest.approx(np.mean(values), abs=0.1)

    def test_more_sets_of_values(self):

        values = [np.array([1, 2, 3])] * 3

        out_mean, _ = boot.bootstrap_std_of_function_results(
            values,
            function=dummy_calculation_many_inputs,
            num_iterations=10000,
            n_threads=None,
            number_of_sets_of_values=3)

        assert out_mean == pytest.approx(np.mean(values) * 3, abs=0.1)
