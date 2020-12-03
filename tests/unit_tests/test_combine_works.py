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

import PythonFSDAM.combine_works as cw


class Testcombine_non_correlated_works():
    def test_works(self):

        work_1 = np.array([1., 2., 3.])

        work_2 = np.array([4., 5.])

        expected = np.array([5., 6., 6., 7., 7., 8.])

        output = cw.combine_non_correlated_works(work_1, work_2)

        assert np.testing.assert_allclose(output, expected) is None
