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

from PythonFSDAM import exp_max_gaussmix


class TestEMGauss():

    def test_fit_1_gaussian(self):

        normal_distribution = norm.rvs(loc=0.,
                                       scale=1.,
                                       size=10000,
                                       random_state=1)

        em = exp_max_gaussmix.EMGauss(normal_distribution, n_gaussians=1)

        gaussians, log_likelyhood = em.fit()

        assert pytest.approx(gaussians[0]['mu'], abs=0.01) == 0.
        assert pytest.approx(gaussians[0]['sigma'], abs=0.01) == 1.

        assert isinstance(log_likelyhood, float)
