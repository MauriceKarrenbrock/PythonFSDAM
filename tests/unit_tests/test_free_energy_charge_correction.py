# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-self-use
#############################################################
# Copyright (c) 2020-2022 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################

import math

import pytest

from PythonFSDAM import free_energy_charge_correction as char_corr


class Testhomogeneus_charge_correction_alchemical_leg():

    def test_zero(self):
        output = char_corr.homogeneus_charge_correction_alchemical_leg(
            0., 0., 1.)
        expected = 0.

        assert output == pytest.approx(expected)

    def test_minus_one(self):

        output = char_corr.homogeneus_charge_correction_alchemical_leg(
            0., 1., 1., (math.pi**0.5) / (2**0.5))
        expected = -1.

        assert output == pytest.approx(expected)


class Testhomogeneus_charge_correction_vDSSB():

    def test_works(self, mocker):
        m_corr = mocker.patch(
            'PythonFSDAM.free_energy_charge_correction.homogeneus_charge_correction_alchemical_leg',
            return_value=1)

        output = char_corr.homogeneus_charge_correction_vDSSB(
            host_charge=1.,
            guest_charge=1.,
            host_guest_box_volume=1.,
            only_guest_box_volume=1.)
        expected = 2

        assert output == expected

        m_corr.assert_called()
