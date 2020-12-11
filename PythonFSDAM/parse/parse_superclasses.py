# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""This are the superclasses of all the parsing classes

This superclasses are both inherited by the factories in .parse
and in the instances the factories instantiate
"""


class Parser(object):
    """The generic parser superclass

    Methods
    --------------
    parse(file_name)
        parses the given file
    """
    def __init__(self, md_program):

        self.md_program = md_program

    @staticmethod
    def parse(file_name):
        """parses a file

        all classes should implement this method
        """

        raise NotImplementedError

        #parse file_name


class ParseWorkProfileSuperclass(Parser):
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
        check `self.md_program` to
        check which kind of md progam files an instance parses
    """
    @staticmethod
    def parse(file_name):
        """all classes should implement this method

        it should return a 2-D numpy.array containing [[lambda, ...], [dH/dL, ...]]
        """

        raise NotImplementedError

        #parse file_name

        #return a 2-D numpy.array
