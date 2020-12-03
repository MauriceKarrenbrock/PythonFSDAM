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


class ParseWorkProfile(object):
    """Parse the work profile output file

    This is a factory that instantiates the right class to
    parse the work profile file

    all returned classes will implement the `parse(file_name)` method
    and return a 2-D numpy.array containing [[lambda, ...], [dH/dL, ...]]
    for more details check the class you are interested in

    Parameters
    -----------
    md_program : str
        the md program that created the file,
        check cls.implemented() to check the
        implemented md programs and the
        relative keywords

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

    classes = {'gromacs': FSDAMGromacs.parse.GromacsParseWorkProfile}

    @classmethod
    def implemented(cls):
        """returns a list of implemented keywords
        """

        return cls.classes.keys()

    def __new__(cls, md_program):

        return cls.classes[md_program]()

    def parse(self, file_name):
        """all classes should implement this method

        it should return a 2-D numpy.array containing [[lambda, ...], [dH/dL, ...]]
        """

        #parse file_name

        #return a 2-D numpy.array
