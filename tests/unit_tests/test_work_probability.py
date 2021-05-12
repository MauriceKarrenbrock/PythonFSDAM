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

import PythonFSDAM.work_probability as prob


class Testmake_probability_histogram():
    def test_works(self, mocker):

        m_max = mocker.patch('numpy.max', return_value=1.)
        m_min = mocker.patch('numpy.min', return_value=0.)
        m_hist = mocker.patch('numpy.histogram', return_value='output')
        m_ceil = mocker.patch('math.ceil')

        values = [[1, 2, 3], [4, 5, 6]]

        output = prob.make_probability_histogram(values, 0.1)

        assert output == 'output'

        m_max.assert_called_once()
        m_min.assert_called_once()
        m_hist.assert_called_once()
        m_ceil.assert_called_once_with(10.)
