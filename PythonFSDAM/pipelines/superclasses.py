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

import numpy as np
import PythonAuxiliaryFunctions.files_IO.write_file as _write
import PythonPDBStructures.trajectories.extract_frames as extract_frames

import PythonFSDAM.bootstrapping as bootstrapping
import PythonFSDAM.combine_works as combine_works
import PythonFSDAM.free_energy_calculations as free_energy_calculations
import PythonFSDAM.integrate_works as integrate_works
import PythonFSDAM.parse.parse as parse
import PythonFSDAM.purge_outliers as purge_outliers

# pylint: disable=too-many-statements


class Pipeline(object):
    """The generic superclass for pipelines

    Methods
    --------
    execute
        executes the pipeline
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
    temperature : float
    """
    def __init__(self,
                 topology_files,
                 md_program_path,
                 alchemical_residue,
                 structure_files=None,
                 temperature=298.15):

        self.topology_files = topology_files

        self.md_program_path = md_program_path

        self.alchemical_residue = alchemical_residue

        self.structure_files = structure_files

        self.temperature = temperature

        #For developers: this is should be a string that will be used to correctly
        #instantiate any kind of classes that are usually instantiated with a factory
        #it might be something like 'gromacs'
        self.md_program = None

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
    """The superclass of all the postprocessing classes

    in the end you will obtain a free energy and a confidence intrvall
    other details might be subclass dependent

    Parameters
    -------------
    bound_state_dhdl : list of pathlib.Path
        the files with the lambda and dH/dL values for the
        bound state, if the md program you are using forces
        you to annihilate in 2 different runs (q and vdw) like gromacs
        use a nested list, in any case check out the specific subclass
        documentation
    unbound_state_dhdl : list of pathlib.Path
        the files with the lambda and dH/dL values for the
        unbound state, if the md program you are using forces
        you to create in 2 different runs (q and vdw) like gromacs
        use a nested list, in any case check out the specific subclass
        documentation
    vol_correction_distances_bound_state : pathlib.Path, optional
        if you have put a COM-COM restrain between the protein and the ligand
        you will have to make a volume correction to the free energy, and to do
        it a distance file obtained from the previous enhanced sampling run might be
        needed. If you leave it blank no volume correction will be done (can always do it
        later on your own)
    vol_correction_distances_unbound_state : pathlib.Path, optional
        if you have put a COM-COM restrain between the ligand and something else
        you will have to make a volume correction to the free energy, and to do
        it a distance file obtained from the previous enhanced sampling run might be
        needed. If you leave it blank no volume correction will be done (can always do it
        later on your own but you won't probably need it in the unbound state)
    """
    def __init__(self,
                 bound_state_dhdl,
                 unbound_state_dhdl,
                 vol_correction_distances_bound_state=None,
                 vol_correction_distances_unbound_state=None,
                 temperature=298.15):

        self.bound_state_dhdl = bound_state_dhdl

        self.unbound_state_dhdl = unbound_state_dhdl

        self.vol_correction_distances_bound_state = vol_correction_distances_bound_state

        self.vol_correction_distances_unbound_state = vol_correction_distances_unbound_state

        self.temperature = temperature

        #For developers: this is should be a string that will be used to correctly
        #instantiate any kind of classes that are usually instantiated with a factory
        #it might be something like 'gromacs'
        self.md_program = None

        self._free_energy_value = 0

    def _calculate_free_energy(self, works):
        """Calculates the free energy somehow

        it is a hook for subclasses and it must
        update `self._free_energy_value`

        Parameters
        -------------
        works : numpy.array
            the work values
        """

        raise NotImplementedError

    def _propagate_error(self, error, works=None):
        """propagates the error

        according to `calculate_free_energy` or better to what is
        implemented inside it in the subclass the error will be
        propagated accordingly in this method

        Parameters
        -------------
        error : float
            the value of the error before the free energy calculation
        works : numpy.array, optional, default=None
            some error propagations might need the work values

        Returns
        -----------
        float
            the new error
        """

        raise NotImplementedError

    def _calculate_free_energy_volume_correction(self):
        """Calculates the free energy volume correction

        it is a hook for subclasses and it must
        update `self._free_energy_value`

        Parameters
        ------------
        distance_file : pathlib.Path
            the file containing the distance values
        """

        raise NotImplementedError

    def execute(self):
        """Calculates the free energy

        Returns
        -----------
        float, float
            free energy, 2sigma_cofidence_itervall
        """

        bound_multiple_runs = False
        unbound_multiple_runs = False
        number_of_bound_subruns = 1
        number_of_unbound_subruns = 1
        #understand if the bound and unbound runs where made in one or multiple runs
        if hasattr(self.bound_state_dhdl[0], '__iter__') and not \
            isinstance(self.bound_state_dhdl[0], str):

            bound_multiple_runs = True

            number_of_bound_subruns = len(self.bound_state_dhdl[0])


        if hasattr(self.unbound_state_dhdl[0], '__iter__') and not \
            isinstance(self.unbound_state_dhdl[0], str):

            unbound_multiple_runs = True

            number_of_unbound_subruns = len(self.unbound_state_dhdl[0])

        dhdl_parser = parse.ParseWorkProfile(self.md_program)

        bound_works_calculate = []
        for i in range(number_of_bound_subruns):  # pylint: disable=unused-variable

            bound_works_calculate.append(
                integrate_works.WorkResults(len(self.bound_state_dhdl)))

        #parse and save all the bound run work values
        for file_name in self.bound_state_dhdl:

            for i in range(number_of_bound_subruns):

                if bound_multiple_runs:

                    lambda_work_value = dhdl_parser.parse(file_name[i])

                else:

                    lambda_work_value = dhdl_parser.parse(file_name)

                bound_works_calculate[i].integrate_and_add(lambda_work_value)

        bound_work_values = bound_works_calculate[0].get_work_values()

        for i in range(1, len(bound_works_calculate)):

            bound_work_values += bound_works_calculate[i].get_work_values()

        #free memory, this arrays can be quite big
        for i in range(len(bound_works_calculate)):

            bound_works_calculate[i] = None

        del bound_works_calculate

        bound_work_values = purge_outliers.purge_outliers_zscore(
            bound_work_values, z_score=3.0)

        # print a backup to file
        np.savetxt('bound_work_values.dat',
                   bound_work_values,
                   header='bound work values after z score purging')

        #I am only interested in the vale and not in the confidence intervall
        STD_bound = bootstrapping.standard_deviation(bound_work_values)[0]

        ############################################################

        ############################################################

        #do everything again for unbound works
        unbound_works_calculate = []
        for i in range(number_of_unbound_subruns):  # pylint: disable=unused-variable

            unbound_works_calculate.append(
                integrate_works.WorkResults(len(self.unbound_state_dhdl)))

        #parse and save all the unbound run work values
        for file_name in self.unbound_state_dhdl:

            for i in range(number_of_unbound_subruns):

                if unbound_multiple_runs:

                    lambda_work_value = dhdl_parser.parse(file_name[i])

                else:

                    lambda_work_value = dhdl_parser.parse(file_name)

                unbound_works_calculate[i].integrate_and_add(lambda_work_value)

        unbound_work_values = unbound_works_calculate[0].get_work_values()

        for i in range(1, len(unbound_works_calculate)):

            unbound_work_values += unbound_works_calculate[i].get_work_values()

        #free memory, this arrays can be quite big
        for i in range(len(unbound_works_calculate)):

            unbound_works_calculate[i] = None

        del unbound_works_calculate

        unbound_work_values = purge_outliers.purge_outliers_zscore(
            unbound_work_values, z_score=3.0)

        # print a backup to file
        np.savetxt('unbound_work_values.dat',
                   unbound_work_values,
                   header='unbound work values after z score purging')

        #I am only interested in the vale and not in the confidence intervall
        STD_unbound = bootstrapping.standard_deviation(unbound_work_values)[0]

        #combine the bound and unbound work values and get a 2sigma (95%) confidence intarvall
        confidence_intervall = 2. * STD_bound + 2. * STD_unbound

        del STD_bound
        del STD_unbound

        combined_work_values = \
            combine_works.combine_non_correlated_works(bound_work_values, unbound_work_values)

        # print a backup to file
        np.savetxt('combined_work_values.dat',
                   combined_work_values,
                   header='combined_work_values work values')

        del bound_work_values
        del unbound_work_values

        #subclass dependent
        self._calculate_free_energy(combined_work_values)

        self._calculate_free_energy_volume_correction()

        confidence_intervall = self._propagate_error(confidence_intervall,
                                                     combined_work_values)

        # print the values of delta G and the confidence intervall (2 sigma)
        lines = [
            '# Delta_G  Confidence_intervall(2 sigma)\n',
            f'{self._free_energy_value:.18e} {confidence_intervall:.18e}'
        ]
        _write.write_file(lines, 'free_energy.dat')

        return self._free_energy_value, confidence_intervall


class JarzynskiPostProcessingPipeline(PostProcessingPipeline):
    """subclass of `PostProcessingPipeline` that calculates free energy with  Jarzynski

    The distances for this class must be in Angstrom and the energies will always be in
    Kcal/mol
    """
    def _calculate_free_energy(self, works):
        """Calculates the free energy with Jarzinky

        Parameters
        -------------
        works : numpy.array
            the work values
        """

        energy = free_energy_calculations.jarzynski_free_energy(
            works, self.temperature)

        self._free_energy_value += energy

    def _propagate_error(self, error, works=None):
        """propagates the error for Jarzynski

        Parameters
        -------------
        error : float
            the value of the error before the free energy calculation

        Returns
        -----------
        float
            the new error
        """

        if not works:
            raise ValueError(
                f'works must be a non empty numpy.array, not {works}')

        return free_energy_calculations.jarzynski_error_propagation(
            error, works, self.temperature)

    def _calculate_free_energy_volume_correction(self):
        """Calculates the free energy volume correction

        it takes for granted you have a COM COM restraint
        and uses the COM COM distance to make a correction

        Parameters
        ------------
        distance_file : pathlib.Path
            the file containing the distance values
            they must be in Angstrom and will return in Kcal/mol
        """

        files = [
            self.vol_correction_distances_bound_state,
            self.vol_correction_distances_unbound_state
        ]

        distances = []

        parser = parse.ParseCOMCOMDistanceFile(self.md_program)

        for i in files:

            if i is not None:

                distances.append(parser.parse(i))

        for dist in distances:

            energy = free_energy_calculations.volume_correction(
                dist, self.temperature)

            self._free_energy_value += energy
