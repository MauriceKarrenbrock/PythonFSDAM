# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-self-use
# pylint: disable=protected-access
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################

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

        pass
        # m_free = mocker.patch('PythonFSDAM.free_energy_calculations.jarzynski_free_energy',
        # return_value=-1)

        # tmp_class = _super.GaussianMixtureFreeEnergyMixIn()

        # output = tmp_class.calculate_free_energy(1, 2)

        # assert output == -1

        # m_free.assert_called_once_with(1, 2)

    def test_vdssb_calculate_free_energy(self, mocker):

        m_combine = mocker.patch(
            'PythonFSDAM.combine_works.combine_non_correlated_works',
            return_value=55)

        m_free = mocker.patch(
            'PythonFSDAM.free_energy_calculations.gaussian_mixtures_free_energy',
            return_value=(-1, -2, -3))

        m_write = mocker.patch.object(_super.GaussianMixtureFreeEnergyMixIn,
                                      '_write_gaussians')

        tmp_class = _super.GaussianMixtureFreeEnergyMixIn()

        output = tmp_class.vdssb_calculate_free_energy(1, 2, 3)

        assert output == -1

        m_combine.assert_called_once_with(1, 2)

        m_free.assert_called_once_with(55, 3)

        m_write.assert_called_once_with(-2, -3)

    def test_calculate_standard_deviation(self, mocker):

        pass
        # m_std = mocker.patch(
        #     'PythonFSDAM.free_energy_calculations.plain_jarzynski_error_propagation',
        #     return_value=(-1, -2))

        # tmp_class = _super.GaussianMixtureFreeEnergyMixIn()

        # output = tmp_class.calculate_standard_deviation(1, 2)

        # assert output == -1

        # m_std.assert_called_once_with(1, temperature=2)

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
