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

from PythonFSDAM import free_energy_charge_correction as charge_correction


class Testglobular_protein_correction():

    def test_works(self, get_data_dir):
        input_file = get_data_dir / 'ligand_in_pocket.gro'

        output = charge_correction.globular_protein_correction(
            input_file,
            host_charge=1,
            guest_charge=1,
            ligand='resname LIG',
        )

        assert pytest.approx(
            output
        ) == -0.00011226578741158404  # a old result to check consistency
