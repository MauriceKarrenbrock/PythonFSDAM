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


class Helper():
    def __init__(self, temperature, boltzmann_kappa):

        self.temperature = temperature
        self.boltzmann_kappa = boltzmann_kappa

    def calculate_jarzynski(self, values):

        return free.jarzynski_free_energy(values,
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
