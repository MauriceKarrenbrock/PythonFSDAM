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

import pytest
from scipy.stats import norm

import PythonFSDAM.bidirectional_free_energy_calculations as free


class Testbar_free_energy():

    def test_works(self):
        random_normal_points_1 = norm.rvs(loc=-1, scale=1, size=100000)

        random_normal_points_2 = norm.rvs(loc=3, scale=1, size=100000)

        output = free.bar_free_energy(random_normal_points_1,
                                      random_normal_points_2)

        assert output == pytest.approx(2, abs=0.01)


class Test_plain_bar_error_propagation():

    def test_works(self):
        random_normal_points_1 = norm.rvs(loc=-1, scale=1, size=100000)

        random_normal_points_2 = norm.rvs(loc=3, scale=1, size=100000)

        _, out_mean = free.plain_bar_error_propagation(random_normal_points_1,
                                                       random_normal_points_2,
                                                       num_iterations=500)

        assert out_mean == pytest.approx(2, abs=0.01)


class Testcrossing_of_2_gaussians():

    def test_works_1_crossing(self):
        output = free.crossing_of_2_gaussians(mu1=1, mu2=3, sigma1=1, sigma2=1)

        assert output[0] == pytest.approx(2, abs=0.01)

    def test_works_2_crossings(self):
        output = free.crossing_of_2_gaussians(mu1=1, mu2=3, sigma1=1, sigma2=5)

        assert min(output) == pytest.approx(-0.9612595198527841, abs=0.01)
        assert max(output) == pytest.approx(2.794592853186117, abs=0.01)


class Testcrooks_gaussian_crossing_free_energy():

    def test_works_1_crossing(self):
        random_normal_points_1 = norm.rvs(loc=-1, scale=1, size=100000)

        random_normal_points_2 = norm.rvs(loc=3, scale=1, size=100000)

        output = free.crooks_gaussian_crossing_free_energy(
            random_normal_points_1, random_normal_points_2)

        assert output == pytest.approx(-2, abs=0.01)

    def test_works_2_crossings(self):
        random_normal_points_1 = norm.rvs(loc=-1, scale=1, size=100000)

        random_normal_points_2 = norm.rvs(loc=3, scale=5, size=100000)

        output = free.crooks_gaussian_crossing_free_energy(
            random_normal_points_1, random_normal_points_2)

        assert output == pytest.approx(-2.794592853186117, abs=0.01)


class Testplain_crooks_gaussian_crossing_error_propagation():

    def test_works_1_crossing(self):
        random_normal_points_1 = norm.rvs(loc=-1, scale=1, size=100000)

        random_normal_points_2 = norm.rvs(loc=3, scale=1, size=100000)

        _, out_mean = free.plain_crooks_gaussian_crossing_error_propagation(
            random_normal_points_1, random_normal_points_2, num_iterations=500)

        assert out_mean == pytest.approx(-2, abs=0.01)
