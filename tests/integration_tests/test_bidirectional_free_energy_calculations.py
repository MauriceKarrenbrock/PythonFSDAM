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

        assert output == pytest.approx(1, abs=0.01)


class Test_plain_bar_error_propagation():

    def test_works(self):
        random_normal_points_1 = norm.rvs(loc=-1, scale=1, size=100000)

        random_normal_points_2 = norm.rvs(loc=3, scale=1, size=100000)

        _, out_mean = free.plain_bar_error_propagation(random_normal_points_1,
                                                       random_normal_points_2,
                                                       num_iterations=500)

        assert out_mean == pytest.approx(1, abs=0.01)
