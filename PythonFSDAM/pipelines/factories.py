# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Contains the pipeline factories
"""

import FSDAMGromacs.pipelines.preprocessing

import PythonFSDAM.pipelines.superclasses as superclasses


class PreProcessing(superclasses.PreProcessingPipeline):
    """Pre processing for FSDAM factory

    This is a factory that instantiates the right class to
    preprocess the needed files for FSDAM

    it inherits from `PythonFSDAM.pipelines.superclasses.PreProcessingPipeline`
    class so check out it's documentation for more info

    Parameters
    -----------
    md_program : str
        the md program that you want to use,
        check `cls.implemented()` to check the
        implemented md programs and the
        relative keywords
    kwargs : keyword arguments
        any input argument that the class to instantiate
        requires (no checks are done)

    Attributes
    -------------
    _classes : dict
        a dictionary for class instantiation (PRIVATE)

    Returns
    ----------
    The instance of the right class

    Raises
    ----------
    KeyError
        if you give a not valid `md_program`
        value
    """

    _classes = {
        'gromacs': FSDAMGromacs.pipelines.preprocessing.PreprocessGromacsFSDAM
    }

    def __new__(cls, md_program, **kwargs):

        if cls is PreProcessing:

            return cls._classes[md_program](**kwargs)

        return super(PreProcessing, cls).__new__(cls, md_program, **kwargs)

    @classmethod
    def implemented(cls):
        """returns a list of implemented keywords
        """

        return cls._classes.keys()

    def execute(self):

        raise NotImplementedError
