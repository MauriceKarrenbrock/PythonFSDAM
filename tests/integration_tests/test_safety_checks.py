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

from PythonFSDAM import safety_checks


class Testcheck_ligand_in_pocket():
    def test_multiple_files(self, get_data_dir):
        in_pocket = get_data_dir / 'ligand_in_pocket.gro'
        not_in_pocket = get_data_dir / 'ligand_out_of_pocket.gro'

        input_files = [in_pocket, not_in_pocket, in_pocket]

        output = safety_checks.check_ligand_in_pocket(
            ligand='resname LIG',
            pocket='resSeq 92 81',  # completely random values
            pdb_file=input_files)

        assert output == [True, False, True]

    def test_high_tollerance(self, get_data_dir):
        in_pocket = get_data_dir / 'ligand_in_pocket.gro'
        not_in_pocket = get_data_dir / 'ligand_out_of_pocket.gro'

        input_files = [in_pocket, not_in_pocket, in_pocket]

        output = safety_checks.check_ligand_in_pocket(
            ligand='resname LIG',
            pocket='all',  # completely random values
            pdb_file=input_files,
            tollerance=1000)

        assert output == [True, True, True]
