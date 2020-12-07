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

import unittest.mock as mock

import PythonFSDAM.parse.parse as parse


class TestParseWorkProfile():
    def test__new__(self, mocker):

        class_A = mock.MagicMock()

        class_B = mock.MagicMock()

        mocker.patch.object(parse.ParseWorkProfile, '_classes', {
            'A': class_A,
            'B': class_B
        })

        parse.ParseWorkProfile('A')

        class_A.assert_called_once()

        class_B.assert_not_called()

    def test_implemented(self, mocker):

        mocker.patch.object(parse.ParseWorkProfile, '_classes', {
            'A': 'a',
            'B': 'b'
        })

        output = parse.ParseWorkProfile.implemented()

        assert tuple(output) == ('A', 'B')
