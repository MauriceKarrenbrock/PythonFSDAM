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

import PythonFSDAM.purge_outliers as purge_outliers


class Testpurge_outliers_zscore():
    def test_works(self):

        input_values = np.array([1, 2, 3, 10000])

        expected_output = np.array([1, 2, 3])

        output = purge_outliers.purge_outliers_zscore(values=input_values,
                                                      z_score=1.0)

        assert np.array_equal(output, expected_output)

    def test_empty_input(self):

        input_values = np.array([])

        with pytest.raises(ValueError):

            purge_outliers.purge_outliers_zscore(values=input_values,
                                                 z_score=1.0)

    def test_empty_output(self):

        input_values = np.array([-10000, 10000])

        expected_output = np.array([])

        output = purge_outliers.purge_outliers_zscore(values=input_values,
                                                      z_score=0.00001)

        assert np.array_equal(output, expected_output)
