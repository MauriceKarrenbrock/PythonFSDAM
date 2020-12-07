# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Contains the pipeline superclasses
"""

import pathlib

import PythonPDBStructures.trajectories.extract_frames as extract_frames


class Pipeline(object):
    """The generic superclass for pipelines

    Methods
    --------
    execute
    """
    def execute(self):
        """Executes the pipeline, it is implemented by the subclasses
        """

        raise NotImplementedError


class PreProcessingPipeline(Pipeline):
    """This is the superclass that deals with the preprocessing before FSDAM

    some subclass will then implement the methods for the given md program

    Parameters
    ------------
    topology_files : list of pathlib.Path
        the topology files needed (ex: gromacs: .top, .itp)
    md_program_path : pathlib.Path
        the path to the executable of the md program
    alchemical_residue : str
        the residue name of the alchemical residue (must be different from other
        residue names to distinguish it)
    structure_files : list of pathlib.Path, optional
        the input pdb files to start the FSDAM from
        normally obtained with HREM, if not given
        the `extract_from_trajectory` method must be called
        before `execute`
    """
    def __init__(self,
                 topology_files,
                 md_program_path,
                 alchemical_residue,
                 structure_files=None):

        self.topology_files = topology_files

        self.md_program_path = md_program_path

        self.alchemical_residue = alchemical_residue

        self.structure_files = structure_files

    def extract_from_trajectory(self,
                                delta_steps,
                                trajectory,
                                topology,
                                output_name='fsdam_input_',
                                output_format='pdb',
                                starting=0,
                                end=None):
        """this method extracts structure files from a trajectory

        it is a wrapper to
        `PythonPDBStructures.trajectories.extract_frames.extract_frames`
        function https://github.com/MauriceKarrenbrock/PythonPDBStructures.git

        it updates `self.structure_files`
        """

        nr_of_files = extract_frames.extract_frames(
            delta_steps=delta_steps,
            trajectory=trajectory,
            topology=topology,
            output_name=output_name,
            output_format=output_format,
            starting=starting,
            end=end)

        list_of_files = []

        for i in range(nr_of_files):

            list_of_files.append(
                pathlib.Path(f'{output_name}{i}.{output_format}'))

        self.structure_files = list_of_files

    def execute(self):
        """Executes the pipeline

        Returns
        ----------
        dict
            this dictionary is strictly implementation dependent
            and will contain all the needed files to do the molecular
            dynamics run
        """


class PostProcessingPipeline(Pipeline):
    """WORK IN PROGRESS
    """
    def execute(self):
        """wip
        """
