# -*- coding: utf-8 -*-
# pylint: disable=too-many-statements
# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Contains the pipeline superclasses for the post processing of bidirectional processes
"""

import numpy as np
import scipy.stats

import PythonFSDAM.bidirectional_free_energy_calculations as _free
import PythonFSDAM.combine_works as combine_works
import PythonFSDAM.pipelines.superclasses as _super

################################################
# MixIn classes
################################################


class FreeEnergyMixInSuperclass(object):
    """superclass for free energy mixins

    useful for abstract pipeline superclasses
    """

    def calculate_free_energy(self, works_1, works_2, temperature):
        """calculate the bidirectional free energy

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

            return super().calculate_free_energy(works_1, works_2, temperature)

        except AttributeError:

            raise NotImplementedError(
                'Need to define this abstract method') from None

    def vdssb_calculate_free_energy(self, bound_works_1, unbound_works_1,
                                    bound_works_2, unbound_works_2,
                                    temperature):
        """calculate the Jarzynski free energy vDSSB

        Parameters
        ------------
        bound_works_1 : numpy.array
            numpy array of the work values
            in Kcal/mol
        unbound_works_1 : numpy.array
            numpy array of the work values
            in Kcal/mol
        bound_works_2 : numpy.array
            numpy array of the work values
            in Kcal/mol
        unbound_works_2 : numpy.array
            numpy array of the work values
            in Kcal/mol
        temperature : float
            kelvin

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        #Avoid problems with possible ill defined multiple inheritances
        try:

            return super().vdssb_calculate_free_energy(bound_works_1,
                                                       unbound_works_1,
                                                       bound_works_2,
                                                       unbound_works_2,
                                                       temperature)

        except AttributeError:

            raise NotImplementedError(
                'Need to define this abstract method') from None

    def calculate_standard_deviation(self, works_1, works_2, temperature):
        """calculates STD of the free energy with bootstrapping
        """

        #Avoid problems with possible ill defined multiple inheritances
        try:

            return super().calculate_standard_deviation(
                works_1, works_2, temperature)

        except AttributeError:

            raise NotImplementedError(
                'Need to define this abstract method') from None

    def vdssb_calculate_standard_deviation(self, bound_works_1,
                                           unbound_works_1, bound_works_2,
                                           unbound_works_2, temperature):
        """calculates STD of the free energy with bootstrapping vDSSB

        it is a wrapper of `PythonFSDAM.free_energy_calculations.vDSSB_jarzynski_error_propagation`
        """

        #Avoid problems with possible ill defined multiple inheritances
        try:

            return super().vdssb_calculate_standard_deviation(
                bound_works_1, unbound_works_1, bound_works_2, unbound_works_2,
                temperature)

        except AttributeError:

            raise NotImplementedError(
                'Need to define this abstract method') from None


class BarFreeEnergyMixIn(FreeEnergyMixInSuperclass):
    """MixIn that implements the Jarzynski theorem

    contains the methods needed both for the standard and vDSSB
    approach

    the vDSSB methods will contain vdssb in the method name

    Notes
    ------------
    implements the abstract class `FreeEnergyMixInSuperclass`
    """

    @staticmethod
    def calculate_free_energy(works_1, works_2, temperature):
        """calculate the bidirectional free energy

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

        return _free.bar_free_energy(works_1, works_2, temperature)

    def vdssb_calculate_free_energy(self, bound_works_1, unbound_works_1,
                                    bound_works_2, unbound_works_2,
                                    temperature):
        """calculate the BAR free energy vDSSB

        Parameters
        ------------
        bound_works_1 : numpy.array
            numpy array of the work values
            in Kcal/mol
        unbound_works_1 : numpy.array
            numpy array of the work values
            in Kcal/mol
        bound_works_2 : numpy.array
            numpy array of the work values
            in Kcal/mol
        unbound_works_2 : numpy.array
            numpy array of the work values
            in Kcal/mol
        temperature : float
            kelvin

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        works_1 = combine_works.combine_non_correlated_works(
            bound_works_1, unbound_works_1)

        works_2 = combine_works.combine_non_correlated_works(
            bound_works_2, unbound_works_2)

        free_energy = self.calculate_free_energy(works_1, works_2, temperature)

        return free_energy

    @staticmethod
    def calculate_standard_deviation(works_1, works_2, temperature):
        """calculates STD of the free energy with bootstrapping
        """

        STD, mean = _free.plain_bar_error_propagation(works_1,
                                                      works_2,
                                                      temperature=temperature)

        return STD, mean

    @staticmethod
    def vdssb_calculate_standard_deviation(bound_works_1, unbound_works_1,
                                           bound_works_2, unbound_works_2,
                                           temperature):
        """calculates STD of the free energy with bootstrapping vDSSB
        """

        STD, mean = _free.vDSSB_bar_error_propagation(bound_works_1,
                                                      unbound_works_1,
                                                      bound_works_2,
                                                      unbound_works_2,
                                                      temperature=temperature)

        return STD, mean


class CrooksGaussianCrossingFreeEnergyMixIn(FreeEnergyMixInSuperclass):
    """MixIn that implements the Crooks gaussian crossing theorem

    contains the methods needed both for the standard and vDSSB
    approach

    the vDSSB methods will contain vdssb in the method name

    Notes
    ------------
    implements the abstract class `FreeEnergyMixInSuperclass`
    """

    @staticmethod
    def calculate_free_energy(works_1, works_2, temperature):
        """calculate the bidirectional free energy

        Parameters
        ------------
        works_1 : numpy.array
            the work values in Kcal/mol
        works_2 : numpy.array
            the work values in Kcal/mol
        temperature : float
            is there only for interface homogeneity

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        temperature = None  # useless

        return _free.crooks_gaussian_crossing_free_energy(works_1, works_2)

    def vdssb_calculate_free_energy(self, bound_works_1, unbound_works_1,
                                    bound_works_2, unbound_works_2,
                                    temperature):
        """calculate the Crooks gaussian crossing free energy vDSSB

        Parameters
        ------------
        bound_works_1 : numpy.array
            numpy array of the work values
            in Kcal/mol
        unbound_works_1 : numpy.array
            numpy array of the work values
            in Kcal/mol
        bound_works_2 : numpy.array
            numpy array of the work values
            in Kcal/mol
        unbound_works_2 : numpy.array
            numpy array of the work values
            in Kcal/mol
        temperature : float
            is there only for interface homogeneity

        Returns
        --------------
        free_energy : float
            Kcal/mol
        """

        temperature = None  # useless

        works_1 = combine_works.combine_non_correlated_works(
            bound_works_1, unbound_works_1)

        works_2 = combine_works.combine_non_correlated_works(
            bound_works_2, unbound_works_2)

        free_energy = self.calculate_free_energy(works_1, works_2, temperature)

        return free_energy

    @staticmethod
    def calculate_standard_deviation(works_1, works_2, temperature):
        """calculates STD of the free energy with bootstrapping
        """

        temperature = None  # useless

        STD, mean = _free.plain_crooks_gaussian_crossing_error_propagation(
            works_1, works_2)

        return STD, mean

    @staticmethod
    def vdssb_calculate_standard_deviation(bound_works_1, unbound_works_1,
                                           bound_works_2, unbound_works_2,
                                           temperature):
        """calculates STD of the free energy with bootstrapping vDSSB
        """

        temperature = None  # useless

        STD, mean = _free.vDSSB_crooks_gaussian_crossing_error_propagation(
            bound_works_1, unbound_works_1, bound_works_2, unbound_works_2)

        return STD, mean


############################################################
# End MixIns
############################################################


class PostProcessingSuperclass(_super.Pipeline, FreeEnergyMixInSuperclass,
                               _super.FreeEnergyCorrectionsMixIn,
                               _super.IntegrateWorksMixIn):
    """the abstract superclass for posr processing without vDSSB
    for bidirectional processes (Crooks)

    subclasses must define the methods of the abstract `FreeEnergyMixInSuperclass`
    superclass

    Parameters
    -------------
    dhdl_files_1 : list of pathlib.Path
        the files with the lambda and dH/dL values,
        if the md program you are using forces
        you to annihilate in 2 different runs (q and vdw) like gromacs
        use a nested list, in any case check out the specific subclass
        documentation
    dhdl_files_2 : list of pathlib.Path
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
    creation_1 : bool, optional, default=True
        it is only used for the kind of md programs that print the
        work values in function of time and not of lambda
        if creation=True the class will take for granted that
        lambda went linearly from 0 to 1, viceversa for creation=False
    creation_2 : bool, optional, default=False
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
                 dhdl_files_1,
                 dhdl_files_2,
                 vol_correction_distances=None,
                 temperature=298.15,
                 md_program='gromacs',
                 creation_1=True,
                 creation_2=False):

        self.dhdl_files_1 = dhdl_files_1

        self.dhdl_files_2 = dhdl_files_2

        self.vol_correction_distances = vol_correction_distances

        self.temperature = temperature

        self.md_program = md_program

        self.creation_1 = creation_1

        self.creation_2 = creation_2

        self._free_energy_value = 0.

        self._check_creation()

    def __str__(self):

        return 'not_defined_pipeline'

    def _check_creation(self):
        """Private
        """
        # 1
        if not (isinstance(self.creation_1, bool)
                and isinstance(self.creation_2, bool)):
            raise TypeError(
                'creation_1 and creation_2 must be boolean, '
                f'not {type(self.creation_1)} and {type(self.creation_2)}')

        if self.creation_1 == self.creation_2:
            raise ValueError('creation_1 and creation_2 must be different, '
                             'you are not describing a bidirectional process')

    def _write_free_energy_file(self, STD):
        """Private
        """

        # print the values of delta G and the confidence intervall (95%)
        lines = [
            f'# {str(self)}\n',
            '# Delta_G  STD  confidence_intervall_95%(1.96STD)  unit=Kcal/mol\n',
            f'{self._free_energy_value:.18e} {STD:.18e} {1.96*STD:.18e}\n'
        ]

        with open(f'{str(self)}_free_energy.dat', 'w') as f:
            f.write(''.join(lines))

    def _hook(self, works_1, works_2):
        """Virtual method"""

    def execute(self):
        """calculate free energy

        Returns
        ---------
        free_energy, STD : float, float
            the free energy and the standard deviation
        """

        z_score = 100000
        work_values_1 = self.get_purged_work_values(self.dhdl_files_1,
                                                    md_program=self.md_program,
                                                    creation=self.creation_1,
                                                    z_score=z_score)

        work_values_2 = self.get_purged_work_values(self.dhdl_files_2,
                                                    md_program=self.md_program,
                                                    creation=self.creation_2,
                                                    z_score=z_score)

        np.savetxt(
            'work_values_1.dat',
            work_values_1,
            header=
            ('work_values work values Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(work_values_1).statistic}'
             ))

        np.savetxt(
            'work_values_2.dat',
            work_values_2,
            header=
            ('work_values work values Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(work_values_2).statistic}'
             ))

        #volume correction
        if self.vol_correction_distances is not None:

            for file_name in self.vol_correction_distances:

                self._free_energy_value += self.volume_com_com_correction(
                    distance_file=file_name,
                    temperature=self.temperature,
                    md_program=self.md_program)

        STD, free_energy = self.calculate_standard_deviation(
            work_values_1, work_values_2, temperature=self.temperature)

        self._free_energy_value += free_energy

        self._write_free_energy_file(STD)

        self._hook(work_values_1, work_values_2)

        return self._free_energy_value, STD


class VDSSBPostProcessingPipeline(_super.Pipeline, FreeEnergyMixInSuperclass,
                                  _super.FreeEnergyCorrectionsMixIn,
                                  _super.IntegrateWorksMixIn):
    """The class for vDSDB postprocessing

    in the end you will obtain a free energy and a confidence intervall

    Inherits from `Pipeline` `FreeEnergyMixInSuperclass`
    `FreeEnergyCorrectionsMixIn` `IntegrateWorksMixIn`

    Parameters
    -------------
    bound_state_dhdl_1 : list of pathlib.Path
        the files with the lambda and dH/dL values for the
        bound state, if the md program you are using forces
        you to annihilate in 2 different runs (q and vdw) like gromacs
        use a nested list
    unbound_state_dhdl_1 : list of pathlib.Path
        the files with the lambda and dH/dL values for the
        unbound state, if the md program you are using forces
        you to create in 2 different runs (q and vdw) like gromacs
        use a nested list
    bound_state_dhdl_2 : list of pathlib.Path
        the files with the lambda and dH/dL values for the
        bound state, if the md program you are using forces
        you to annihilate in 2 different runs (q and vdw) like gromacs
        use a nested list
    unbound_state_dhdl_2 : list of pathlib.Path
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
    bound_creation_1 : bool, default=False
        True if the ligand was generate in the protein-ligand system, false if it was annihilated,
        must be different from `unbound_creation` otherwise the obtained convolution
        will be meaningless
    unbound_creation_1 : bool, default=True
        True if the ligand was generate in a box of water, false if it was annihilated,
        must be different from `unbound_creation` otherwise the obtained convolution
        will be meaningless
    bound_creation_2 : bool, default=False
        True if the ligand was generate in the protein-ligand system, false if it was annihilated,
        must be different from `unbound_creation` otherwise the obtained convolution
        will be meaningless
    unbound_creation_2 : bool, default=True
        True if the ligand was generate in a box of water, false if it was annihilated,
        must be different from `unbound_creation` otherwise the obtained convolution
        will be meaningless
    """

    def __init__(self,
                 *,
                 bound_state_dhdl_1,
                 unbound_state_dhdl_1,
                 bound_state_dhdl_2,
                 unbound_state_dhdl_2,
                 vol_correction_distances_bound_state=None,
                 vol_correction_distances_unbound_state=None,
                 temperature=298.15,
                 md_program='gromacs',
                 bound_creation_1=False,
                 unbound_creation_1=True,
                 bound_creation_2=True,
                 unbound_creation_2=False):

        self.bound_state_dhdl_1 = bound_state_dhdl_1

        self.unbound_state_dhdl_1 = unbound_state_dhdl_1

        self.bound_state_dhdl_2 = bound_state_dhdl_2

        self.unbound_state_dhdl_2 = unbound_state_dhdl_2

        self.vol_correction_distances_bound_state = vol_correction_distances_bound_state

        self.vol_correction_distances_unbound_state = vol_correction_distances_unbound_state

        self.temperature = temperature

        self.md_program = md_program

        self.bound_creation_1 = bound_creation_1

        self.unbound_creation_1 = unbound_creation_1

        self.bound_creation_2 = bound_creation_2

        self.unbound_creation_2 = unbound_creation_2

        self._free_energy_value = 0.

        # Checking if everything is ok
        self._check_bound_unbound_creation()

    def __str__(self):

        return 'vDSSB_not_defined_pipeline'

    def _check_bound_unbound_creation(self):
        """Private
        """
        # 1
        if not (isinstance(self.bound_creation_1, bool)
                and isinstance(self.unbound_creation_1, bool)):
            raise TypeError(
                'bound_creation_1 and unbound_creation_1 must be boolean, '
                f'not {type(self.bound_creation_1)} and {type(self.unbound_creation_1)}'
            )

        if self.bound_creation_1 == self.unbound_creation_1:
            raise ValueError(
                'bound_creation_1 and unbound_creation_1 must be different, '
                'otherwise the obtained convolution will be meaningless')

        # 2
        if not (isinstance(self.bound_creation_2, bool)
                and isinstance(self.unbound_creation_2, bool)):
            raise TypeError(
                'bound_creation_2 and unbound_creation_2 must be boolean, '
                f'not {type(self.bound_creation_2)} and {type(self.unbound_creation_2)}'
            )

        if self.bound_creation_2 == self.unbound_creation_2:
            raise ValueError(
                'bound_creation_2 and unbound_creation_2 must be different, '
                'otherwise the obtained convolution will be meaningless')

        # Check if the two processes are actually different
        if self.bound_creation_1 == self.bound_creation_2:
            raise ValueError(
                '(un)bound_creation_1 and (un)bound_creation_2 must be different, '
                'you are not describing a bidirectional process')

    def _write_free_energy_file(self, STD):
        """Private
        """

        # print the values of delta G and the confidence intervall (95%)
        lines = [
            f'# {str(self)}\n',
            '# Delta_G  STD  confidence_intervall_95%(1.96STD)  unit=Kcal/mol\n',
            f'{self._free_energy_value:.18e} {STD:.18e} {1.96*STD:.18e}\n'
        ]

        with open(f'{str(self)}_free_energy.dat', 'w') as f:
            f.write(''.join(lines))

    def _hook(self):
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

        z_score = 100000

        # 1
        #numpy array
        bound_work_values_1 = self.get_purged_work_values(
            self.bound_state_dhdl_1,
            md_program=self.md_program,
            creation=self.bound_creation_1,
            z_score=z_score)

        #numpy array
        unbound_work_values_1 = self.get_purged_work_values(
            self.unbound_state_dhdl_1,
            md_program=self.md_program,
            creation=self.unbound_creation_1,
            z_score=z_score)

        # print a backup to file
        np.savetxt(
            'bound_work_values_1.dat',
            bound_work_values_1,
            header=
            ('bound work values after z score purging Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(bound_work_values_1).statistic}'
             ))

        np.savetxt(
            'unbound_work_values_1.dat',
            unbound_work_values_1,
            header=
            ('unbound work values after z score purging Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(unbound_work_values_1).statistic}'
             ))

        # 2
        #numpy array
        bound_work_values_2 = self.get_purged_work_values(
            self.bound_state_dhdl_2,
            md_program=self.md_program,
            creation=self.bound_creation_2,
            z_score=z_score)

        #numpy array
        unbound_work_values_2 = self.get_purged_work_values(
            self.unbound_state_dhdl_2,
            md_program=self.md_program,
            creation=self.unbound_creation_2,
            z_score=z_score)

        # print a backup to file
        np.savetxt(
            'bound_work_values_2.dat',
            bound_work_values_2,
            header=
            ('bound work values after z score purging Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(bound_work_values_2).statistic}'
             ))

        np.savetxt(
            'unbound_work_values_2.dat',
            unbound_work_values_2,
            header=
            ('unbound work values after z score purging Kcal/mol '
             f'(outliers with z score > {z_score} were purged) '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(unbound_work_values_2).statistic}'
             ))

        #get STD
        STD, free_energy = self.vdssb_calculate_standard_deviation(
            bound_work_values_1,
            unbound_work_values_1,
            bound_work_values_2,
            unbound_work_values_2,
            temperature=self.temperature)

        #get free energy
        self._free_energy_value += free_energy

        #make a backup of the combined work values
        combined_work_values_1 = combine_works.combine_non_correlated_works(
            bound_work_values_1, unbound_work_values_1)

        del bound_work_values_1
        del unbound_work_values_1

        # print a backup to file
        np.savetxt(
            'combined_work_values_1.dat',
            combined_work_values_1,
            header=
            ('combined_work_values work values Kcal/mol '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(combined_work_values_1).statistic}'
             ))

        del combined_work_values_1

        combined_work_values_2 = combine_works.combine_non_correlated_works(
            bound_work_values_2, unbound_work_values_2)

        del bound_work_values_2
        del unbound_work_values_2

        # print a backup to file
        np.savetxt(
            'combined_work_values_2.dat',
            combined_work_values_2,
            header=
            ('combined_work_values work values Kcal/mol '
             f'ANDERSON_TEST_VALUE={scipy.stats.anderson(combined_work_values_2).statistic}'
             ))

        del combined_work_values_2

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

        self._hook()

        return self._free_energy_value, STD


class BarVDSSBPostProcessingPipeline(VDSSBPostProcessingPipeline,
                                     BarFreeEnergyMixIn):
    """calculates free energy with  BAR and the vDSSB method

    subclass of `VDSSBPostProcessingPipeline` and `BarFreeEnergyMixIn` that

    The distances for this class must be in Angstrom and the energies will always be in
    Kcal/mol

    This class elaborates bound and unbound state toghether, that is very different from
    the not vDSSB one
    """

    def __str__(self):

        return 'vDSSB_bidirectional_bar_pipeline'


class BarPostProcessingAlchemicalLeg(PostProcessingSuperclass,
                                     BarFreeEnergyMixIn):
    """post processing class for plain BAR (no vDSSB)

    this class post processes only a single alchemical leg (bound or unbound)
    note that this behaviour is very different from his vDSSB counterpart that
    post processes everything together

    see documentation for `PostProcessingSuperclass` and `BARFreeEnergyMixIn`
    """

    def __str__(self):

        return 'standard_bidirectional_bar_pipeline'


class CrooksGaussianCrossingVDSSBPostProcessingPipeline(
        VDSSBPostProcessingPipeline, CrooksGaussianCrossingFreeEnergyMixIn):
    """calculates free energy with  CrooksGaussianCrossing and the vDSSB method

    subclass of `VDSSBPostProcessingPipeline` and `CrooksGaussianCrossingFreeEnergyMixIn` that

    The distances for this class must be in Angstrom and the energies will always be in
    Kcal/mol

    This class elaborates bound and unbound state toghether, that is very different from
    the not vDSSB one
    """

    def __str__(self):

        return 'vDSSB_bidirectional_crooks_gaussian_crossing_pipeline'


class CrooksGaussianCrossingPostProcessingAlchemicalLeg(
        PostProcessingSuperclass, CrooksGaussianCrossingFreeEnergyMixIn):
    """post processing class for plain CrooksGaussianCrossing (no vDSSB)

    this class post processes only a single alchemical leg (bound or unbound)
    note that this behaviour is very different from his vDSSB counterpart that
    post processes everything together

    see documentation for `PostProcessingSuperclass` and `CrooksGaussianCrossingFreeEnergyMixIn`
    """

    def __str__(self):

        return 'standard_bidirectional_crooks_gaussian_crossing_pipeline'
