# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-self-use
# pylint: disable=protected-access
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################

import unittest.mock as mock

import pytest

import PythonFSDAM.pipelines.superclasses as _super

#############################################################
# TEST MixIn
#############################################################


class TestIntegrateWorksMixIn():
    def test_integrate_work_files(self, mocker):

        m_wrk = mocker.patch(
            'PythonFSDAM.integrate_works.integrate_multiple_work_files',
            return_value=-1)

        tmp_class = _super.IntegrateWorksMixIn()

        output = tmp_class.integrate_work_files(['1', '2'],
                                                md_program='gromacs',
                                                creation=True)

        assert output == -1

        m_wrk.assert_called_once_with(work_files=['1', '2'],
                                      md_program='gromacs',
                                      creation=True)

    def test_get_purged_work_values_nested(self, mocker):

        m_integrate = mocker.patch.object(_super.IntegrateWorksMixIn,
                                          'integrate_work_files',
                                          side_effect=[-1, -2])

        m_purge = mocker.patch(
            'PythonFSDAM.purge_outliers.purge_outliers_zscore',
            return_value=-2)

        tmp_class = _super.IntegrateWorksMixIn()

        output = tmp_class.get_purged_work_values([['file_1'], ['file_2']],
                                                  md_program='gromacs',
                                                  creation=True,
                                                  z_score=3.0)

        assert output == -2

        m_integrate.assert_called_with(file_names=['file_2'],
                                       creation=True,
                                       md_program='gromacs')

        m_purge.assert_called_once_with(-3.0, z_score=3.0)

    def test_get_purged_work_values_not_nested(self, mocker):

        m_integrate = mocker.patch.object(_super.IntegrateWorksMixIn,
                                          'integrate_work_files',
                                          return_value=-1)

        m_purge = mocker.patch(
            'PythonFSDAM.purge_outliers.purge_outliers_zscore',
            return_value=-2)

        tmp_class = _super.IntegrateWorksMixIn()

        output = tmp_class.get_purged_work_values(['file'],
                                                  md_program='gromacs',
                                                  creation=True,
                                                  z_score=3.0)

        assert output == -2

        m_integrate.assert_called_once_with(file_names=['file'],
                                            creation=True,
                                            md_program='gromacs')

        m_purge.assert_called_once_with(-1.0, z_score=3.0)


class TestFreeEnergyCorrectionsMixIn():
    def test_volume_com_com_correction(self, mocker):

        m_parse = mocker.patch(
            'PythonFSDAM.parse.parse.ParseCOMCOMDistanceFile')

        m_parse.return_value.parse.return_value = 'parsed_stuff'

        m_vol = mocker.patch(
            'PythonFSDAM.free_energy_calculations.volume_correction',
            return_value=-1)

        tmp_class = _super.FreeEnergyCorrectionsMixIn()

        output = tmp_class.volume_com_com_correction('dist_file',
                                                     temperature=1000.0,
                                                     md_program='gromacs')

        assert output == -1

        m_parse.assert_called_once_with('gromacs')

        m_parse.return_value.parse.assert_called_once_with('dist_file')

        m_vol.assert_called_once_with('parsed_stuff', 1000.0)


class TestFreeEnergyMixInSuperclass():
    def test_all_notimplementederror(self):

        tmp_class = _super.FreeEnergyMixInSuperclass()

        with pytest.raises(NotImplementedError):

            tmp_class.calculate_free_energy(1, 2)

        with pytest.raises(NotImplementedError):

            tmp_class.vdssb_calculate_free_energy(1, 2, 3)

        with pytest.raises(NotImplementedError):

            tmp_class.calculate_standard_deviation(1, 2)

        with pytest.raises(NotImplementedError):

            tmp_class.vdssb_calculate_standard_deviation(1, 2, 3)


class TestJarzynskiFreeEnergyMixIn():
    def test_calculate_free_energy(self, mocker):

        m_free = mocker.patch(
            'PythonFSDAM.free_energy_calculations.jarzynski_free_energy',
            return_value=-1)

        tmp_class = _super.JarzynskiFreeEnergyMixIn()

        output = tmp_class.calculate_free_energy(1, 2)

        assert output == -1

        m_free.assert_called_once_with(1, 2)

    def test_vdssb_calculate_free_energy(self, mocker):

        m_combine = mocker.patch(
            'PythonFSDAM.combine_works.combine_non_correlated_works',
            return_value=55)

        m_free = mocker.patch.object(_super.JarzynskiFreeEnergyMixIn,
                                     'calculate_free_energy',
                                     return_value=-1)

        tmp_class = _super.JarzynskiFreeEnergyMixIn()

        output = tmp_class.vdssb_calculate_free_energy(1, 2, 3)

        assert output == -1

        m_combine.assert_called_once_with(1, 2)

        m_free.assert_called_once_with(55, 3)

    def test_calculate_standard_deviation(self, mocker):

        m_std = mocker.patch(
            'PythonFSDAM.free_energy_calculations.plain_jarzynski_error_propagation',
            return_value=(-1, -2))

        tmp_class = _super.JarzynskiFreeEnergyMixIn()

        output = tmp_class.calculate_standard_deviation(1, 2)

        assert output == -1

        m_std.assert_called_once_with(1, temperature=2)

    def test_vdssb_calculate_standard_deviation(self, mocker):

        m_std = mocker.patch(
            'PythonFSDAM.free_energy_calculations.vDSSB_jarzynski_error_propagation',
            return_value=(-1, -2))

        tmp_class = _super.JarzynskiFreeEnergyMixIn()

        output = tmp_class.vdssb_calculate_standard_deviation(1, 2, 3)

        assert output == -1

        m_std.assert_called_once_with(1, 2, temperature=3)


class TestGaussianMixtureFreeEnergyMixIn():
    def test__write_gaussians(self, mocker):

        m_write = mocker.patch(
            'PythonAuxiliaryFunctions.files_IO.write_file.write_file')

        mocker.patch.object(_super.GaussianMixtureFreeEnergyMixIn,
                            '__str__',
                            return_value='test')

        tmp_class = _super.GaussianMixtureFreeEnergyMixIn()

        gaussians = [{
            'mu': -1,
            'sigma': -2,
            'lambda': -3
        }, {
            'mu': -4,
            'sigma': -5,
            'lambda': -6
        }]

        log_likelyhood = 1

        tmp_class._write_gaussians(gaussians, log_likelyhood)

        expected_list = [
            f'#each line is a gaussian, log likelyhood = {log_likelyhood}\n',
            'mean,sigma,coefficient\n',
            f'{gaussians[0]["mu"]:.18e},{gaussians[0]["sigma"]:.18e},'
            f'{gaussians[0]["lambda"]:.18e}\n',
            f'{gaussians[1]["mu"]:.18e},{gaussians[1]["sigma"]:.18e},'
            f'{gaussians[1]["lambda"]:.18e}\n'
        ]

        m_write.assert_called_once_with(expected_list, 'test_gaussians.csv')

    def test_calculate_free_energy(self, mocker):

        m_free = mocker.patch(
            'PythonFSDAM.free_energy_calculations.gaussian_mixtures_free_energy',
            return_value=(-1, -2, -3))

        m_write = mocker.patch.object(_super.GaussianMixtureFreeEnergyMixIn,
                                      '_write_gaussians')

        tmp_class = _super.GaussianMixtureFreeEnergyMixIn()

        output = tmp_class.calculate_free_energy(1, 2)

        assert output == -1

        m_free.assert_called_once_with(1, 2)

        m_write.assert_called_once_with(-2, -3)

    def test_vdssb_calculate_free_energy(self, mocker):

        m_combine = mocker.patch(
            'PythonFSDAM.combine_works.combine_non_correlated_works',
            return_value=55)

        m_free = mocker.patch.object(_super.GaussianMixtureFreeEnergyMixIn,
                                     'calculate_free_energy',
                                     return_value=-1)

        tmp_class = _super.GaussianMixtureFreeEnergyMixIn()

        output = tmp_class.vdssb_calculate_free_energy(1, 2, 3)

        assert output == -1

        m_combine.assert_called_once_with(1, 2)

        m_free.assert_called_once_with(55, 3)

    def test_calculate_standard_deviation(self, mocker):

        m_std = mocker.patch(
            'PythonFSDAM.free_energy_calculations.plain_gaussian_mixtures_error_propagation',
            return_value=(-1, -2))

        tmp_class = _super.GaussianMixtureFreeEnergyMixIn()

        output = tmp_class.calculate_standard_deviation(1, 2)

        assert output == -1

        m_std.assert_called_once_with(1, temperature=2)

    def test_vdssb_calculate_standard_deviation(self, mocker):

        m_std = mocker.patch(
            'PythonFSDAM.free_energy_calculations.VDSSB_gaussian_mixtures_error_propagation',
            return_value=(-1, -2))

        tmp_class = _super.GaussianMixtureFreeEnergyMixIn()

        output = tmp_class.vdssb_calculate_standard_deviation(1, 2, 3)

        assert output == -1

        m_std.assert_called_once_with(1, 2, temperature=3)


#############################################################
# END TEST MixIn
#############################################################


class TestPostProcessingSuperclass():
    def test__init__(self):

        test_class = _super.PostProcessingSuperclass(
            ['files'],
            vol_correction_distances=['dist'],
            temperature=1,
            md_program='MD',
            creation=False)

        assert test_class.dhdl_files == ['files']
        assert test_class.vol_correction_distances == ['dist']
        assert test_class.temperature == 1
        assert test_class.md_program == 'MD'
        assert not test_class.creation
        assert test_class._free_energy_value == 0.

    def test_execute(self, mocker):

        m_work_purge = mocker.patch.object(_super.PostProcessingSuperclass,
                                           'get_purged_work_values',
                                           return_value=-1)

        m_free = mocker.patch.object(_super.PostProcessingSuperclass,
                                     'calculate_free_energy',
                                     return_value=-2)

        m_vol = mocker.patch.object(_super.PostProcessingSuperclass,
                                    'volume_com_com_correction',
                                    return_value=-3)

        m_std = mocker.patch.object(_super.PostProcessingSuperclass,
                                    'calculate_standard_deviation',
                                    return_value=-4)

        m_savetxt = mocker.patch('numpy.savetxt')

        m_write = mocker.patch(
            'PythonAuxiliaryFunctions.files_IO.write_file.write_file')

        test_class = _super.PostProcessingSuperclass(
            ['files'],
            vol_correction_distances=['dist'],
            temperature=1,
            md_program='MD',
            creation=False)

        out_free, out_std = test_class.execute()

        assert out_free == -5
        assert out_std == -4

        m_work_purge.assert_called_once_with(['files'],
                                             md_program='MD',
                                             creation=False,
                                             z_score=3.0)

        m_free.assert_called_once_with(-1, temperature=1)

        m_vol.assert_called_once_with(distance_file='dist',
                                      temperature=1,
                                      md_program='MD')

        m_std.assert_called_once_with(-1, temperature=1)

        m_savetxt.assert_called_once_with(
            'work_values.dat',
            -1,
            header=('work_values work values Kcal/mol '
                    f'(outliers with z score > {3.0} were purged)'))

        expected_lines = [
            '# not_defined_pipeline\n',
            '# Delta_G  STD  confidence_intervall_95%(1.96STD)  unit=Kcal/mol\n',
            f'{out_free:.18e} {out_std:.18e} {1.96*out_std:.18e}\n'
        ]

        m_write.assert_called_once_with(
            expected_lines, 'not_defined_pipeline_free_energy.dat')


class VDSSBTestPostProcessingSuperclass():
    def test__init__(self):

        test_class = _super.VDSSBPostProcessingPipeline(
            bound_state_dhdl=['bound_files'],
            unbound_state_dhdl=['unbound_files'],
            vol_correction_distances_bound_state=['bound_dist'],
            vol_correction_distances_unbound_state=['unbound_dist'],
            temperature=1,
            md_program='MD')

        assert test_class.bound_state_dhdl == ['bound_files']
        assert test_class.unbound_state_dhdl == ['unbound_files']
        assert test_class.vol_correction_distances_bound_state == [
            'bound_dist'
        ]
        assert test_class.vol_correction_distances_unbound_state == [
            'unbound_dist'
        ]
        assert test_class.temperature == 1
        assert test_class.md_program == 'MD'
        assert test_class._free_energy_value == 0.

    def test_execute(self, mocker):

        m_work_purge = mocker.patch.object(_super.VDSSBPostProcessingPipeline,
                                           'get_purged_work_values',
                                           return_value=-1)

        m_free = mocker.patch.object(_super.VDSSBPostProcessingPipeline,
                                     'vdssb_calculate_free_energy',
                                     return_value=-2)

        m_vol = mocker.patch.object(_super.VDSSBPostProcessingPipeline,
                                    'volume_com_com_correction',
                                    return_value=-3)

        m_std = mocker.patch.object(_super.VDSSBPostProcessingPipeline,
                                    'vdssb_calculate_standard_deviation',
                                    return_value=-4)

        m_savetxt = mocker.patch('numpy.savetxt')

        m_combine = mocker.patch(
            'PythonFSDAM.combine_works.combine_non_correlated_works',
            return_value=55)

        m_write = mocker.patch(
            'PythonAuxiliaryFunctions.files_IO.write_file.write_file')

        test_class = _super.VDSSBPostProcessingPipeline(
            bound_state_dhdl=['bound_files'],
            unbound_state_dhdl=['unbound_files'],
            vol_correction_distances_bound_state=None,
            vol_correction_distances_unbound_state=None,
            temperature=1,
            md_program='MD')

        out_free, out_std = test_class.execute()

        assert out_free == -5
        assert out_std == -4

        calls = [
            mock.call(['bound_files'],
                      md_program='MD',
                      creation=False,
                      z_score=3.0),
            mock.call(['unbound_files'],
                      md_program='MD',
                      creation=True,
                      z_score=3.0)
        ]

        m_work_purge.assert_has_calls(calls)

        m_free.assert_called_once_with(-1, temperature=1)

        m_vol.assert_not_called()

        m_std.assert_called_once_with(-1, temperature=1)

        calls = [
            mock.call(
                'bound_work_values.dat',
                -1,
                header=('bound work values after z score purging Kcal/mol '
                        f'(outliers with z score > {3.0} were purged)')),
            mock.call(
                'unbound_work_values.dat',
                -1,
                header=('unbound work values after z score purging Kcal/mol '
                        f'(outliers with z score > {3.0} were purged)')),
            mock.call('combined_work_values.dat',
                      55,
                      header=('combined_work_values work values Kcal/mol'))
        ]

        m_savetxt.assert_has_calls(calls)

        m_combine.assert_called_once_with(-1, -1)

        expected_lines = [
            '# vDSSB_not_defined_pipeline\n',
            '# Delta_G  STD  confidence_intervall_95%(1.96STD)  unit=Kcal/mol\n',
            f'{out_free:.18e} {out_std:.18e} {1.96*out_std:.18e}\n'
        ]

        m_write.assert_called_once_with(
            expected_lines, 'vDSSB_not_defined_pipeline_free_energy.dat')
