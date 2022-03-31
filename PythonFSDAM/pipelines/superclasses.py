# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
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
import scipy.stats

import PythonFSDAM.combine_works as combine_works
import PythonFSDAM.free_energy_calculations as free_energy_calculations
import PythonFSDAM.integrate_works as integrate_works
import PythonFSDAM.parse.parse as parse
import PythonFSDAM.purge_outliers as purge_outliers

# pylint: disable=too-many-statements

################################################
# MixIn classes
################################################


class IntegrateWorksMixIn(object):
    """MixIn class to integrate work values and files
    """

    @staticmethod
    def integrate_work_files(file_names, md_program, creation):
        """integrate many work files

        it is a wrapper of `PythonFSDAM.integrate_works.integrate_multiple_work_files`
        function, check it's documentation for more
        """

        return integrate_works.integrate_multiple_work_files(
            work_files=file_names, md_program=md_program, creation=creation)

    def get_purged_work_values(self, dhdl_files, md_program, creation,
                               z_score):
        """convenience method to get work values

        the work values are already purged with z score

        Parameters
        --------------
        dhdl_files : list of pathlib.Path or nested list of pathlib.Path
        md_program : str
            the md program that created the dhdl files
        creation : bool
            check `PythonFSDAM.integrate_works.integrate_multiple_work_files`
        z_score : float
            check `PythonFSDAM.purge_outliers.purge_outliers_zscore`

        Returns
        -----------
        numpy.array
        """

        number_of_subruns = 1
        #understand if the bound and unbound runs where made in one or multiple runs
        if hasattr(dhdl_files[0], '__iter__') and not \
            isinstance(dhdl_files[0], str):

            number_of_subruns = len(dhdl_files)

        #will become a numpy array
        work_values = 0.
        for i in range(number_of_subruns):

            if number_of_subruns > 1:

                dhdl_tmp = dhdl_files[i]

            else:

                dhdl_tmp = dhdl_files

            work_values += self.integrate_work_files(file_names=dhdl_tmp,
                                                     creation=creation,
                                                     md_program=md_program)

        work_values = purge_outliers.purge_outliers_zscore(work_values,
                                                           z_score=z_score)

        return work_values


class FreeEnergyCorrectionsMixIn(object):
    """MixIn for volume, orientational, ... corrections to free energy
    """

    @staticmethod
    def volume_com_com_correction(distance_file, temperature, md_program):
        """make volume correction for COM-COM weak restraints (COM=center of mass)

        it is a wrapper of the `PythonFSDAM.parse.parse.ParseCOMCOMDistanceFile`
        and `PythonFSDAM.free_energy_calculations.volume_correction` functions

        it takes for granted you have a COM COM restraint
        and uses the COM COM distance to make a correction

        Parameters
        ------------
        distance_file : pathlib.Path
            the file containing the com-com distances
        temperature : float
            temperature in kelvin
        md_program : str
            the md program that created the files

        Returns
        ---------------
        free_energy_correction : float
            in Kcal/mol
        """

        parser = parse.ParseCOMCOMDistanceFile(md_program)

        distances = parser.parse(distance_file)

        free_energy_correction = free_energy_calculations.volume_correction(
            distances, temperature)

        return free_energy_correction


class FreeEnergyMixInSuperclass(object):
    """superclass for free energy mixins

    useful for abstract pipeline superclasses
    """

    def calculate_free_energy(self, works, temperature):
        """calculate the Jarzynski free energy

        Parameters
        ------------
        works : numpy.array
            the work values in Kcal/mol
        temperature : float
            kelvin

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        #Avoid problems with possible ill defined multiple inheritances
        try:

            return super().calculate_free_energy(works, temperature)

        except AttributeError:

            raise NotImplementedError(
                'Need to define this abstract method') from None

    def vdssb_calculate_free_energy(self, works_1, works_2, temperature):
        """calculate the Jarzynski free energy vDSSB

        Parameters
        ------------
        works_1 : numpy.array
            the work values in Kcal/mol
        works_2 : numpy.array
            the work values in Kcal/mol
        temperature : float
            kelvin

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        #Avoid problems with possible ill defined multiple inheritances
        try:

            return super().vdssb_calculate_free_energy(works_1, works_2,
                                                       temperature)

        except AttributeError:

            raise NotImplementedError(
                'Need to define this abstract method') from None

    def calculate_standard_deviation(self, works, temperature):
        """calculates STD of the free energy with bootstrapping

        it is a wrapper of `PythonFSDAM.free_energy_calculations.plain_jarzynski_error_propagation`
        """

        #Avoid problems with possible ill defined multiple inheritances
        try:

            return super().calculate_standard_deviation(works, temperature)

        except AttributeError:

            raise NotImplementedError(
                'Need to define this abstract method') from None

    def vdssb_calculate_standard_deviation(self, works_1, works_2,
                                           temperature):
        """calculates STD of the free energy with bootstrapping vDSSB

        it is a wrapper of `PythonFSDAM.free_energy_calculations.vDSSB_jarzynski_error_propagation`
        """

        #Avoid problems with possible ill defined multiple inheritances
        try:

            return super().vdssb_calculate_standard_deviation(
                works_1, works_2, temperature)

        except AttributeError:

            raise NotImplementedError(
                'Need to define this abstract method') from None


class JarzynskiFreeEnergyMixIn(FreeEnergyMixInSuperclass):
    """MixIn that implements the Jarzynski theorem

    contains the methods needed both for the standard and vDSSB
    approach

    the vDSSB methods will contain vdssb in the method name

    Notes
    ------------
    implements the abstract class `FreeEnergyMixInSuperclass`
    """

    @staticmethod
    def calculate_free_energy(works, temperature):
        """calculate the Jarzynski free energy

        Parameters
        ------------
        works : numpy.array
            the work values in Kcal/mol
        temperature : float
            kelvin

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        free_energy = free_energy_calculations.jarzynski_free_energy(
            works, temperature)

        return free_energy

    def vdssb_calculate_free_energy(self, works_1, works_2, temperature):
        """calculate the Jarzynski free energy vDSSB

        Parameters
        ------------
        works_1 : numpy.array
            the work values in Kcal/mol
        works_2 : numpy.array
            the work values in Kcal/mol
        temperature : float
            kelvin

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        works = combine_works.combine_non_correlated_works(works_1, works_2)

        free_energy = self.calculate_free_energy(works, temperature)

        return free_energy

    @staticmethod
    def calculate_standard_deviation(works, temperature):
        """calculates STD of the free energy with bootstrapping

        it is a wrapper of `PythonFSDAM.free_energy_calculations.plain_jarzynski_error_propagation`
        """

        STD, mean = free_energy_calculations.plain_jarzynski_error_propagation(
            works, temperature=temperature)

        return STD, mean

    @staticmethod
    def vdssb_calculate_standard_deviation(works_1, works_2, temperature):
        """calculates STD of the free energy with bootstrapping vDSSB

        it is a wrapper of `PythonFSDAM.free_energy_calculations.vDSSB_jarzynski_error_propagation`
        """

        STD, mean = free_energy_calculations.vDSSB_jarzynski_error_propagation(
            works_1, works_2, temperature=temperature)

        return STD, mean

    @staticmethod
    def calculate_jarzynski_bias(works, temperature):
        """Estimates the bias in the Jarzynski free energy for normal distributions

        it is a wrapper of `PythonFSDAM.free_energy_calculations.jarzynski_bias_estimation`
        """

        STD = np.std(works)

        return free_energy_calculations.jarzynski_bias_estimation(
            STD, works.size, temperature)

    def vdssb_calculate_jarzynski_bias(self, works_1, works_2, temperature):
        """Estimates the bias in the Jarzynski free energy for normal distributions

        it is a wrapper of `PythonFSDAM.free_energy_calculations.jarzynski_bias_estimation`
        """

        works = combine_works.combine_non_correlated_works(works_1, works_2)

        return self.calculate_jarzynski_bias(works, temperature)


class GaussianMixtureFreeEnergyMixIn(FreeEnergyMixInSuperclass):
    """MixIn for gaussian mixtures free energy calcutations

    for both vDSSB and standard
    """

    def _write_gaussians(self, gaussians, log_likelyhood):
        """private"""

        lines = []

        lines.append(
            f'#each line is a gaussian, log likelyhood = {log_likelyhood}\n')
        lines.append('mean,sigma,coefficient\n')

        for gaussian in gaussians:

            lines.append(
                f'{gaussian["mu"]:.18e},{gaussian["sigma"]:.18e},{gaussian["lambda"]:.18e}\n'
            )

        _write.write_file(lines, f'{str(self)}_gaussians.csv')

    def calculate_free_energy(self, works, temperature):
        """calculate the gaussian mixtures free energy

        Parameters
        ------------
        works : numpy.array
            the work values in Kcal/mol
        temperature : float
            kelvin

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        energy, gaussians, log_likelyhood = free_energy_calculations.gaussian_mixtures_free_energy(
            works, temperature, n_gaussians=self.n_gaussians)  # pylint: disable=no-member)

        #in order to be able to check the quality of the fit
        self._write_gaussians(gaussians, log_likelyhood)

        return energy

    def vdssb_calculate_free_energy(self, works_1, works_2, temperature):
        """calculate the gaussian mixtures free energy vDSSB

        Parameters
        ------------
        works_1 : numpy.array
            the work values in Kcal/mol
        works_2 : numpy.array
            the work values in Kcal/mol
        temperature : float
            kelvin

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        works = combine_works.combine_non_correlated_works(works_1, works_2)

        energy = self.calculate_free_energy(works, temperature)

        return energy

    def calculate_standard_deviation(self, works, temperature):
        """calculates STD of the free energy with bootstrapping

        it is a wrapper of
        `PythonFSDAM.free_energy_calculations.plain_gaussian_mixtures_error_propagation`
        """

        STD, mean, mean_log_likelyhood = free_energy_calculations.plain_gaussian_mixtures_error_propagation(
            works, temperature=temperature, n_gaussians=self.n_gaussians)  # pylint: disable=no-member

        self.mean_log_likelyhood = mean_log_likelyhood

        return STD, mean

    def vdssb_calculate_standard_deviation(self, works_1, works_2,
                                           temperature):
        """calculates STD of the free energy with bootstrapping vDSSB

        it is a wrapper of
        `PythonFSDAM.free_energy_calculations.VDSSB_gaussian_mixtures_error_propagation`
        """

        STD, mean, mean_log_likelyhood = free_energy_calculations.VDSSB_gaussian_mixtures_error_propagation(
            works_1,
            works_2,
            temperature=temperature,
            n_gaussians=self.n_gaussians)  # pylint: disable=no-member

        self.mean_log_likelyhood = mean_log_likelyhood

        return STD, mean


################################################
# END MixIn classes
################################################


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
    constrains : str, default=all-bonds
        possible values none, h-bonds, all-bonds
    """

    def __init__(self,
                 topology_files,
                 md_program_path,
                 alchemical_residue,
                 structure_files=None,
                 temperature=298.15,
                 constrains=None):

        if isinstance(topology_files, str):
            topology_files = pathlib.Path(topology_files)
        if isinstance(topology_files, pathlib.Path):
            topology_files = [topology_files]
        self.topology_files = topology_files

        self.md_program_path = md_program_path

        self.alchemical_residue = alchemical_residue

        self.structure_files = structure_files

        self.temperature = temperature

        if constrains is None:
            constrains = 'all-bonds'
        if constrains not in ('none', 'h-bonds', 'all-bonds'):
            raise ValueError(
                f'constrains must be none, h-bonds, all-bonds not {constrains}'
            )
        self.constrains = constrains

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


class PostProcessingSuperclass(Pipeline, FreeEnergyMixInSuperclass,
                               FreeEnergyCorrectionsMixIn,
                               IntegrateWorksMixIn):
    """the abstract superclass for posr processing without vDSSB

    Inherits from `Pipeline` `FreeEnergyMixInSuperclass`
    `FreeEnergyCorrectionsMixIn` `IntegrateWorksMixIn`

    subclasses must define the methods of the abstract `FreeEnergyMixInSuperclass`
    superclass

    Parameters
    -------------
    dhdl_files : list of pathlib.Path
        the files with the lambda and dH/dL values,
        if the md program you are using forces
        you to annihilate in 2 different runs (q and vdw) like gromacs
        use a nested list, in any case check out the specific subclass
        documentation
    vol_correction_distances : pathlib.Path, optional
        if you have put a COM-COM restrain between the protein and the ligand
        you will have to make a volume correction to the free energy, and to do
        it a distance file obtained from the previous enhanced sampling run might be
        needed. If you leave it blank no volume correction will be done (can always do it
        later on your own)
    temperature : float, default=298.15
        the temperature of the MD runs in Kelvin
    md_program : str, default=gromacs
        the md program that created the output to post-process
        in some cases you can use a superclass directly by setting the right
        `md_program` in other cases you will need to use some subclass
    creation : bool, optional, default=True
        it is only used for the kind of md programs that print the
        work values in function of time and not of lambda
        if creation=True the class will take for granted that
        lambda went linearly from 0 to 1, viceversa for creation=False

    Methods
    -----------
    execute()
        the only public method
    """

    def __init__(self,
                 *,
                 dhdl_files,
                 vol_correction_distances=None,
                 temperature=298.15,
                 md_program='gromacs',
                 creation=True):

        self.dhdl_files = dhdl_files

        self.vol_correction_distances = vol_correction_distances

        self.temperature = temperature

        self.md_program = md_program

        self.creation = creation

        self._free_energy_value = 0.

    def __str__(self):

        return 'not_defined_pipeline'

    def _write_free_energy_file(self, STD):
        """Private
        """

        # print the values of delta G and the confidence intervall (95%)
        lines = [
            f'# {str(self)}\n',
            '# Delta_G  STD  confidence_intervall_95%(1.96STD)  unit=Kcal/mol\n',
            f'{self._free_energy_value:.18e} {STD:.18e} {1.96*STD:.18e}\n'
        ]
        _write.write_file(lines, f'{str(self)}_free_energy.dat')

    def _hook(self, works):
        """Virtual method"""

    def execute(self):
        """calculate free energy

        Returns
        ---------
        free_energy, STD : float, float
            the free energy and the standard deviation
        """

        z_score = 3.0
        work_values = self.get_purged_work_values(self.dhdl_files,
                                                  md_program=self.md_program,
                                                  creation=self.creation,
                                                  z_score=z_score)

        np.savetxt(
            'work_values.dat',
            work_values,
            header=
            ('work_values work values Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(work_values).statistic}'
             ))

        #volume correction
        if self.vol_correction_distances is not None:

            for file_name in self.vol_correction_distances:

                self._free_energy_value += self.volume_com_com_correction(
                    distance_file=file_name,
                    temperature=self.temperature,
                    md_program=self.md_program)

        STD, free_energy = self.calculate_standard_deviation(
            work_values, temperature=self.temperature)

        self._free_energy_value += free_energy

        self._write_free_energy_file(STD)

        self._hook(work_values)

        return self._free_energy_value, STD


class VDSSBPostProcessingPipeline(Pipeline, FreeEnergyMixInSuperclass,
                                  FreeEnergyCorrectionsMixIn,
                                  IntegrateWorksMixIn):
    """The class for vDSDB postprocessing

    in the end you will obtain a free energy and a confidence intervall

    Inherits from `Pipeline` `FreeEnergyMixInSuperclass`
    `FreeEnergyCorrectionsMixIn` `IntegrateWorksMixIn`

    Parameters
    -------------
    bound_state_dhdl : list of pathlib.Path
        the files with the lambda and dH/dL values for the
        bound state, if the md program you are using forces
        you to annihilate in 2 different runs (q and vdw) like gromacs
        use a nested list
    unbound_state_dhdl : list of pathlib.Path
        the files with the lambda and dH/dL values for the
        unbound state, if the md program you are using forces
        you to create in 2 different runs (q and vdw) like gromacs
        use a nested list
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
    md_program : str, default=gromacs
        the md program that created the output to post-process
        in some cases you can use a superclass directly by setting the right
        `md_program` in other cases you will need to use some subclass
    """

    def __init__(self,
                 *,
                 bound_state_dhdl,
                 unbound_state_dhdl,
                 vol_correction_distances_bound_state=None,
                 vol_correction_distances_unbound_state=None,
                 temperature=298.15,
                 md_program='gromacs'):

        self.bound_state_dhdl = bound_state_dhdl

        self.unbound_state_dhdl = unbound_state_dhdl

        self.vol_correction_distances_bound_state = vol_correction_distances_bound_state

        self.vol_correction_distances_unbound_state = vol_correction_distances_unbound_state

        self.temperature = temperature

        self.md_program = md_program

        self._free_energy_value = 0.

    def __str__(self):

        return 'vDSSB_not_defined_pipeline'

    def _write_free_energy_file(self, STD):
        """Private
        """

        # print the values of delta G and the confidence intervall (95%)
        lines = [
            f'# {str(self)}\n',
            '# Delta_G  STD  confidence_intervall_95%(1.96STD)  unit=Kcal/mol\n',
            f'{self._free_energy_value:.18e} {STD:.18e} {1.96*STD:.18e}\n'
        ]
        _write.write_file(lines, f'{str(self)}_free_energy.dat')

    def _hook(self, works_1, works_2):
        """Virtual method"""

    def execute(self):
        """Calculates the free energy

        Returns
        -----------
        float, float
            free energy, STD
        """

        #calculate and purge outliers (zscore > z_score)
        #for bound and unbound work

        z_score = 3.0

        #numpy array
        bound_work_values = self.get_purged_work_values(
            self.bound_state_dhdl,
            md_program=self.md_program,
            creation=False,
            z_score=z_score)

        #numpy array
        unbound_work_values = self.get_purged_work_values(
            self.unbound_state_dhdl,
            md_program=self.md_program,
            creation=True,
            z_score=z_score)

        # print a backup to file
        np.savetxt(
            'bound_work_values.dat',
            bound_work_values,
            header=
            ('bound work values after z score purging Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(bound_work_values).statistic}'
             ))

        np.savetxt(
            'unbound_work_values.dat',
            unbound_work_values,
            header=
            ('unbound work values after z score purging Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(unbound_work_values).statistic}'
             ))

        #get STD
        STD, free_energy = self.vdssb_calculate_standard_deviation(
            bound_work_values,
            unbound_work_values,
            temperature=self.temperature)

        #get free energy
        self._free_energy_value += free_energy

        #make a backup of the combined work values
        combined_work_values = combine_works.combine_non_correlated_works(
            bound_work_values, unbound_work_values)

        self._hook(bound_work_values, unbound_work_values)

        del bound_work_values
        del unbound_work_values

        # print a backup to file
        np.savetxt(
            'combined_work_values.dat',
            combined_work_values,
            header=
            ('combined_work_values work values Kcal/mol '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(combined_work_values).statistic}'
             ))

        del combined_work_values

        #volume correction
        if self.vol_correction_distances_bound_state is not None:

            for file_name in self.vol_correction_distances_bound_state:

                self._free_energy_value += self.volume_com_com_correction(
                    distance_file=file_name,
                    temperature=self.temperature,
                    md_program=self.md_program)

        if self.vol_correction_distances_unbound_state is not None:

            for file_name in self.vol_correction_distances_unbound_state:

                self._free_energy_value += self.volume_com_com_correction(
                    distance_file=file_name,
                    temperature=self.temperature,
                    md_program=self.md_program)

        self._write_free_energy_file(STD)

        return self._free_energy_value, STD


class JarzynskiVDSSBPostProcessingPipeline(VDSSBPostProcessingPipeline,
                                           JarzynskiFreeEnergyMixIn):
    """calculates free energy with  Jarzynski and the vDSSB method

    subclass of `VDSSBPostProcessingPipeline` and `JarzynskiFreeEnergyMixIn` that

    The distances for this class must be in Angstrom and the energies will always be in
    Kcal/mol

    This class elaborates bound and unbound state toghether, that is very different from
    the not vDSSB one
    """

    def __str__(self):

        return 'vDSSB_jarzynski_pipeline'

    def _hook(self, works_1, works_2):
        """private
        """
        jarzynski_bias = self.vdssb_calculate_jarzynski_bias(
            works_1, works_2, self.temperature)

        lines = ('# Estimation of the bias of the jarzanski free energy '
                 'for the convoluted work values\n'
                 '# This estimate is correct for normal work distributions\n'
                 '# You can check the anderson-darling test for normality in '
                 'the first line of the convoluted work values file\n'
                 '# Check equation 11 of ref:\n'
                 '# SAMPL9 blind predictions using\n'
                 '# nonequilibrium alchemical approaches\n'
                 '# Piero Procacci and Guido Guarnieri\n'
                 '# https://arxiv.org/abs/2202.06720\n\n\n'
                 f'Jarzynski_bias_kcalmol={jarzynski_bias:.18e}\n')

        with open(f'{str(self)}_jarzanski_bias.dat', 'w') as f:
            f.write(lines)


class GaussianMixturesVDSSBPostProcessingPipeline(
        VDSSBPostProcessingPipeline, GaussianMixtureFreeEnergyMixIn):
    """subclass of `VDSSBPostProcessingPipeline` that calculates free energy with  gaussian mixtures

    The distances for this class must be in Angstrom and the energies will always be in
    Kcal/mol

    Parameters
    ------------
    n_gaussians : int, default=3
        the number of gaussians to use for the fitting
    **kwargs
        all the needed arguments of the superclass

    Notes
    ----------
    for more info check https://dx.doi.org/10.1021/acs.jctc.0c00634
    and http://dx.doi.org/10.1063/1.4918558
    """

    def __init__(self, *, n_gaussians=3, **kwargs):

        super().__init__(**kwargs)

        self.n_gaussians = n_gaussians
        self.mean_log_likelyhood = None

    def __str__(self):

        return f'vDSSB_gaussian_mixtures_pipeline_N_gaussians_{self.n_gaussians}'

    def _write_free_energy_file(self, STD):
        """Private
        """

        # print the values of delta G and the confidence intervall (95%)
        lines = [
            f'# {str(self)}\n',
            #f'# Mean log likelyhood = {self.mean_log_likelyhood}\n', # TODO solve nan problem
            '# Delta_G  STD  confidence_intervall_95%(1.96STD)  unit=Kcal/mol\n',
            f'{self._free_energy_value:.18e} {STD:.18e} {1.96*STD:.18e}\n'
        ]
        _write.write_file(lines, f'{str(self)}_free_energy.dat')


class JarzynskiPostProcessingAlchemicalLeg(PostProcessingSuperclass,
                                           JarzynskiFreeEnergyMixIn):
    """post processing class for plain jarzynski (no vDSSB)

    this class post processes only a single alchemical leg (creation or annihilation)
    note that this behaviour is very different from his vDSSB counterpart that
    post processes everything together

    see documentation for `PostProcessingSuperclass` and `JarzynskiFreeEnergyMixIn`
    """

    def __str__(self):

        return 'standard_jarzynski_pipeline'

    def _hook(self, works):
        """private
        """
        jarzynski_bias = self.calculate_jarzynski_bias(works, self.temperature)

        lines = ('# Estimation of the bias of the jarzanski free energy\n'
                 '# This estimate is correct for normal work distributions\n'
                 '# You can check the anderson-darling test for normality in '
                 'the first line of the work values file\n'
                 '# Check equation 11 of ref:\n'
                 '# SAMPL9 blind predictions using\n'
                 '# nonequilibrium alchemical approaches\n'
                 '# Piero Procacci and Guido Guarnieri\n'
                 '# https://arxiv.org/abs/2202.06720\n\n\n'
                 f'Jarzynski_bias_kcalmol={jarzynski_bias:.18e}\n')

        with open(f'{str(self)}_jarzanski_bias.dat', 'w') as f:
            f.write(lines)


class GaussianMixturesPostProcessingAlchemicalLeg(
        PostProcessingSuperclass, GaussianMixtureFreeEnergyMixIn):
    """post processing class for plain jarzynski (no vDSSB)

    this class post processes only a single alchemical leg (creation or annihilation)
    note that this behaviour is very different from his vDSSB counterpart that
    post processes everything together

    see documentation for `PostProcessingSuperclass` and `GaussianMixtureFreeEnergyMixIn`

    Parameters
    ------------
    n_gaussians : int, default=3
        the number of gaussians to use for the fitting
    **kwargs
        all the needed arguments of the superclass
    """

    def __init__(self, *, n_gaussians=3, **kwargs):

        super().__init__(**kwargs)

        self.n_gaussians = n_gaussians
        self.mean_log_likelyhood = None

    def __str__(self):

        return f'standard_gaussian_mixtures_pipeline_N_gaussians_{self.n_gaussians}'

    def _write_free_energy_file(self, STD):
        """Private
        """

        # print the values of delta G and the confidence intervall (95%)
        lines = [
            f'# {str(self)}\n',
            #f'# Mean log likelyhood = {self.mean_log_likelyhood}\n', # TODO solve nan problem
            '# Delta_G  STD  confidence_intervall_95%(1.96STD)  unit=Kcal/mol\n',
            f'{self._free_energy_value:.18e} {STD:.18e} {1.96*STD:.18e}\n'
        ]
        _write.write_file(lines, f'{str(self)}_free_energy.dat')
