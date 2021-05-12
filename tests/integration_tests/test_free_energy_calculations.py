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

import PythonFSDAM.free_energy_calculations as free


class Testjarzynski_free_energy():
    def test_works(self):

        works = np.array([1., 2.])

        output = free.jarzynski_free_energy(works,
                                            temperature=298.15,
                                            boltzmann_kappa=0.001985875)

        #previously calculated, only needed to see if nothing changes in the future
        expected = 1.3100437685851793

        assert output == expected


class Testvolume_correction():
    def test_works(self):

        distance_values = np.array([1., 2.])

        temperature = 298.15

        output = free.volume_correction(distance_values, temperature)

        #previously calculated, only needed to see if nothing changes in the future
        expected = -3.544695949996799

        assert output == expected
