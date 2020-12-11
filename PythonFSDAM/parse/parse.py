# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Functions to parse output files of MD programs
"""

import FSDAMGromacs.parse

import PythonFSDAM.parse.parse_superclasses as super_classes


class ParseWorkProfile(super_classes.ParseWorkProfileSuperclass):
    """Parse the work profile output file

    This is a factory that instantiates the right class to
    parse the work profile file

    it inherits from `PythonFSDAM.parse.parse_superclasses.ParseWorkProfileSuperclass`
    class so check out it's documentation for more info

    Parameters
    -----------
    md_program : str
        the md program that created the file,
        check `cls.implemented()` to check the
        implemented md programs and the
        relative keywords, and check `self.md_program` to
        check which kind of md progam files an instance parses

    Attributes
    -------------
    _classes : dict
        a dictionary for class instantiation (PRIVATE)

    Returns
    ----------
    The instance of the right class to parse the
    file

    Raises
    ----------
    KeyError
        if you give a not valid `md_program`
        value
    """

    _classes = {'gromacs': FSDAMGromacs.parse.GromacsParseWorkProfile}

    def __new__(cls, md_program):

        if cls is ParseWorkProfile:

            return cls._classes[md_program](md_program)

        return super(ParseWorkProfile, cls).__new__(cls, md_program)

    @classmethod
    def implemented(cls):
        """returns a list of implemented keywords
        """

        return cls._classes.keys()

    @staticmethod
    def parse(file_name):

        raise NotImplementedError


class ParseCOMCOMDistanceFile(super_classes.Parser):
    """Parse the COM-COM distance output file

    This is a factory that instantiates the right class to
    parse the COM-COM distance output file

    it inherits from `PythonFSDAM.parse.parse_superclasses.Parser`
    class so check out it's documentation for more info

    Parameters
    -----------
    md_program : str
        the md program that created the file,
        check `cls.implemented()` to check the
        implemented md programs and the
        relative keywords, and check `self.md_program` to
        check which kind of md progam files an instance parses

    Attributes
    -------------
    _classes : dict
        a dictionary for class instantiation (PRIVATE)

    Returns
    ----------
    The instance of the right class to parse the
    file

    Raises
    ----------
    KeyError
        if you give a not valid `md_program`
        value
    """

    _classes = {'gromacs': FSDAMGromacs.parse.GromacsParsePullDistances}

    def __new__(cls, md_program):

        if cls is ParseCOMCOMDistanceFile:

            return cls._classes[md_program](md_program)

        return super(ParseCOMCOMDistanceFile, cls).__new__(cls, md_program)

    @classmethod
    def implemented(cls):
        """returns a list of implemented keywords
        """

        return cls._classes.keys()

    @staticmethod
    def parse(file_name):

        raise NotImplementedError
