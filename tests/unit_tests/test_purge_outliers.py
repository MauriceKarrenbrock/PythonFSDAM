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
    def test_raises(self):

        input_values = np.array([])

        with pytest.raises(ValueError):

            purge_outliers.purge_outliers_zscore(values=input_values,
                                                 z_score=1.0)

    @pytest.mark.parametrize(
        'test_type, z_values, expected_output',
        [('low_z', np.array([0, 0, 0]), np.array([1, 2, 3])),
         ('high_z', np.array([10, -10, -9.9]), np.array([]))])
    def test_works(self, mocker, test_type, z_values, expected_output):

        print('Logging test type for visibility: ' + test_type)

        input_values = np.array([1, 2, 3])

        m_zscore = mocker.patch('scipy.stats.zscore', return_value=z_values)

        output = purge_outliers.purge_outliers_zscore(values=input_values,
                                                      z_score=3.0)

        m_zscore.assert_called_once_with(input_values, nan_policy='raise')

        assert np.array_equal(output, expected_output)
