# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-self-use
# pylint: disable=protected-access
#############################################################
# Copyright (c) 2021-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################

import numpy as np
import pandas as pd

from PythonFSDAM import integrate_works


class Testmake_work_vs_lambda_csv():
    # pylint: disable=unsubscriptable-object
    # pylint: disable=no-member
    def test_creation_gromacs_files(self, get_data_dir, tmp_path):

        q_0 = get_data_dir / 'q_creation_0.xvg'
        q_1 = get_data_dir / 'q_creation_1.xvg'

        vdw_0 = get_data_dir / 'vdw_creation_0.xvg'
        vdw_1 = get_data_dir / 'vdw_creation_1.xvg'

        work_files = [[vdw_0, vdw_1], [q_0, q_1]]

        csv_file = integrate_works.make_work_vs_lambda_csv(
            work_files=work_files,
            md_program='gromacs',
            creation=True,
            csv_name=str(tmp_path / 'TEST.csv'))

        created_df = pd.read_csv(csv_file)
        expected_df = pd.read_csv(
            str(get_data_dir / 'creation_work_vs_lambda.csv'))

        assert list(created_df.columns) == list(expected_df.columns)

        for column in created_df.columns:
            assert np.testing.assert_allclose(
                created_df[column].values, expected_df[column].values) is None

    def test_annihilation_gromacs_files(self, get_data_dir, tmp_path):

        q_0 = get_data_dir / 'q_annihilation_0.xvg'
        q_1 = get_data_dir / 'q_annihilation_1.xvg'

        vdw_0 = get_data_dir / 'vdw_annihilation_0.xvg'
        vdw_1 = get_data_dir / 'vdw_annihilation_1.xvg'

        work_files = [[vdw_0, vdw_1], [q_0, q_1]]

        csv_file = integrate_works.make_work_vs_lambda_csv(
            work_files=work_files,
            md_program='gromacs',
            creation=False,
            csv_name=str(tmp_path / 'TEST.csv'))

        created_df = pd.read_csv(csv_file)
        expected_df = pd.read_csv(
            str(get_data_dir / 'annihilation_work_vs_lambda.csv'))

        assert list(created_df.columns) == list(expected_df.columns)

        for column in created_df.columns:
            assert np.testing.assert_allclose(
                created_df[column].values, expected_df[column].values) is None
