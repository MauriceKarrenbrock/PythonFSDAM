# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=no-self-use
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################

import numpy as np

import PythonFSDAM.integrate_works as wrk


class Testintegrate_multiple_work_files():
    def test_works_create_true(self, mocker):

        m_parser = mocker.patch('PythonFSDAM.parse.parse.ParseWorkProfile')

        m_works = mocker.patch('PythonFSDAM.integrate_works.WorkResults')

        m_works.return_value.get_work_values.return_value = 1

        file_names = ['1', '2', '3']

        md_program = 'some_md_program'

        output = wrk.integrate_multiple_work_files(file_names,
                                                   md_program=md_program,
                                                   creation=True)

        assert output == 1

        m_works.assert_called_with(len(file_names))

        m_parser.assert_called_with(md_program)

        m_parser.return_value.parse.assert_called_with('3',
                                                       starting_lambda=0,
                                                       ending_lambda=1)

        m_works.return_value.integrate_and_add.assert_called()

        m_works.return_value.get_work_values.assert_called_once()

    def test_works_create_false(self, mocker):

        m_parser = mocker.patch('PythonFSDAM.parse.parse.ParseWorkProfile')

        m_works = mocker.patch('PythonFSDAM.integrate_works.WorkResults')

        m_works.return_value.get_work_values.return_value = 1

        file_names = ['1', '2', '3']

        md_program = 'some_md_program'

        output = wrk.integrate_multiple_work_files(file_names,
                                                   md_program=md_program,
                                                   creation=False)

        assert output == 1

        m_works.assert_called_with(len(file_names))

        m_parser.assert_called_with(md_program)

        m_parser.return_value.parse.assert_called_with('3',
                                                       starting_lambda=1,
                                                       ending_lambda=0)

        m_works.return_value.integrate_and_add.assert_called()

        m_works.return_value.get_work_values.assert_called_once()

    def test_works_create_TypeError(self, mocker):

        m_parser = mocker.patch('PythonFSDAM.parse.parse.ParseWorkProfile')

        m_parser.return_value.parse.side_effect = [TypeError, None]

        m_works = mocker.patch('PythonFSDAM.integrate_works.WorkResults')

        m_works.return_value.get_work_values.return_value = 1

        file_names = ['1']

        md_program = 'some_md_program'

        output = wrk.integrate_multiple_work_files(file_names,
                                                   md_program=md_program)

        assert output == 1

        m_works.assert_called_with(len(file_names))

        m_parser.assert_called_with(md_program)

        m_parser.return_value.parse.assert_called_with('1')

        m_works.return_value.integrate_and_add.assert_called()

        m_works.return_value.get_work_values.assert_called_once()


class Testintegrate_work_profiles():
    def test_works(self, mocker):

        m_trapz = mocker.patch('numpy.trapz', return_value=1.)

        input_array = np.array([[1, 2, 3], [4, 5, 6]])

        output = wrk.integrate_work_profiles(input_array)

        assert output == 1.

        m_trapz.assert_called_once()


class TestWorkResults():
    def test__init__(self, mocker):

        m_empty = mocker.patch('numpy.empty')

        instance = wrk.WorkResults(3)

        assert instance.number_of_works == 3
        assert instance._index == 0

        m_empty.assert_called_once_with([3])

    def test_add(self):

        instance = wrk.WorkResults(3)

        instance.add(33.)

        assert instance._index == 1
        assert instance._works[0] == 33.

    def test_integrate_and_add(self, mocker):

        m_add = mocker.patch.object(wrk.WorkResults, 'add')

        m_integrate = mocker.patch(
            'PythonFSDAM.integrate_works.integrate_work_profiles',
            return_value=1.)

        input_array = np.array([[1, 2, 3], [4, 5, 6]])

        instance = wrk.WorkResults(3)

        instance.integrate_and_add(input_array)

        m_add.assert_called_once_with(1.)
        m_integrate.assert_called_once()

    def test_get_work_values(self):

        instance = wrk.WorkResults(3)

        assert instance.get_work_values() is instance._works
