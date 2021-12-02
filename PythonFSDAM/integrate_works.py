# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""functions for the integration of the work profiles
"""

import numpy as np
import pandas as pd

import PythonFSDAM.parse.parse as _parse


def get_work_profile_vs_lambda(lambda_work):
    """Shows how the work value evolves in time (lambda is time dependent)

    It's useful to plot during data analysis

    Parameters
    -----------
    lambda_work : numpy.array
        a 2-D array with 1 st line is lambda and the 2nd dH/dL
        [
            [lambda, lambda, lambda],
            [dH/dL, dH/dL, dH/dL]
        ]

    Returns
    ---------
    np.array
        an array with the same length of lambda that shows the time evolution of the work
    """

    output_work = np.empty(len(lambda_work[0]))

    output_work[0] = 0.
    output_work[1] = 0.

    for i in range(2, len(lambda_work[0])):

        output_work[i] = integrate_work_profiles(
            [lambda_work[0][:i], lambda_work[1][:i]])

    return output_work


def integrate_multiple_work_files(work_files,
                                  md_program='gromacs',
                                  creation=True,
                                  csv_name=None):
    """convenience function to parse and integrate many work files

    this is a convenient and quick high level function, it depends on the
    other functions and classes in this module and in the `parse` module

    It also creates a csv file called creation_work_vs_lambda.csv or
    creation_work_vs_lambda.csv (default)
    that contains the time evolution of work along lambda (useful for plotting in post processing)

    Parameters
    -------------
    work_files : iterable of str or path
        the files containing the work values
    md_program : str, default=gromacs
        the md program that created the file, in order to know which parser to
        use, check the `parse` module to see which ones are supported
    creation : bool, optional, default=True
        it is only used for the kind of md programs that print the
        work values in function of time and not of lambda
        if creation=True the function will take for granted that
        lambda went linearly from 0 to 1, viceversa for creation=False
    csv_name : str, default=creation_work_vs_lambda.csv or creation_work_vs_lambda.csv
        a csv file called that contains the time evolution of work along lambda
        (useful for plotting in post processing)

    Returns
    ------------
    numpy.array
        the values of the integrated works in the same order as in input
    """

    if creation:

        starting_lambda = 0.
        ending_lambda = 1.

    else:

        starting_lambda = 1.
        ending_lambda = 0.

    dhdl_parser = _parse.ParseWorkProfile(md_program)

    work_integrator = WorkResults(len(work_files))

    work_vs_lambda_profiles = pd.DataFrame()

    for i, file_name in enumerate(work_files):

        #it is needed for some programs like gromacs
        #that don't print the values of lambda but print the time
        #and therefore the value of lambda must be calculated
        #by the parser
        try:

            lambda_work_value = dhdl_parser.parse(  # pylint: disable=unexpected-keyword-arg
                file_name,
                starting_lambda=starting_lambda,
                ending_lambda=ending_lambda)

        except TypeError:

            lambda_work_value = dhdl_parser.parse(file_name)

        work_vs_lambda_profiles[f'w{i}'] = get_work_profile_vs_lambda(
            lambda_work_value)

        work_integrator.integrate_and_add(lambda_work_value)

    work_vs_lambda_profiles.insert(0, 'lambda', lambda_work_value[0])

    if csv_name is None:
        if creation:
            csv_name = 'creation_work_vs_lambda.csv'
        else:
            csv_name = 'annihilation_work_vs_lambda.csv'

    work_vs_lambda_profiles.to_csv(csv_name, index=False)

    return work_integrator.get_work_values()


def integrate_work_profiles(lambda_work):
    """Integrates the work profiles with numpy.trapz

    it takes the results of a non equilibium simulation
    and integrates it returning an array of work values

    Parameters
    ------------
    lambda_work : numpy.array
        a 2-D array with 1 st line is lambda and the 2nd dH/dL
        [
            [lambda, lambda, lambda],
            [dH/dL, dH/dL, dH/dL]
        ]

    Returns
    ------------
    float
        the value of the integral of the work profile
    """

    work_result = np.trapz(lambda_work[1], lambda_work[0])

    return work_result


class WorkResults(object):
    """Calculate and keep the results of integrated works in a numpy array

    If you are doing a FSDAM (or a vDSSB) you will have to calculate
    plenty of work integrals, this class will calculate and keep them
    for you in a numpy array that you can get in the end

    It is usually handy to instantiate one for the bound and one for the unbound
    works

    Once you don't need the instance of this class consider to free the memory with
    ```instance = None``` or ```del instance``` (the second one is perhaps safer)

    Parameters
    -----------
    number_of_works : int
        the number of work values that you will create
        an numpy.empty array with this lenght will be
        instantiated

    Methods
    -------------
    add(work_value)
        if you want to add a work value that you integrated independently
        with some function of your choice
    integrate_and_add(lambda_work)
        it takes a 2-D numpy.array containing lambda and dh/dl
        integrates the work profile with the default funcion
        (see `integrate_work_profiles` function) and add it
        to the work values
    get_work_values()
        returns the numpy.array with the work values, remember that if
        you have not filled all the values given with `number_of_works`
        some values of the array will be nonsense
    get_standard_deviation()
        returns the bootstrapped standard deviation (STD) of the work values
        run this method only when you have already filled in all the
        work values otherwise you will get some nonsense result
        For confidence intervall 2*std (95%) is usually a good value
        check `bootstrapping.standard_deviation` for more info
    """
    def __init__(self, number_of_works):

        self.number_of_works = number_of_works

        self._index = 0

        self._works = np.empty([self.number_of_works])

    def add(self, work_value):
        """add an already integrated work value

        if you want to add a work value that you integrated independently
        with some function of your choice

        Parameters
        ------------
        work_value : float
        """

        self._works[self._index] = work_value

        self._index += 1

    def integrate_and_add(self, lambda_work):
        """integrate and add a work value

        it takes a 2-D numpy.array containing lambda and dh/dl
        integrates the work profile with the default funcion
        (see `integrate_work_profiles` function) and add it
        to the work values

        Parameters
        ------------
        lambda_work : numpy.array
            2-D array lambda dh/dl
        """

        work_value = integrate_work_profiles(lambda_work)

        self.add(work_value)

    def get_work_values(self):
        """returns the work values

        returns the numpy.array with the work values, remember that if
        you have not filled all the values given with `number_of_works`
        some values of the array will be nonsense

        Returns
        -----------
        numpy.array
        """

        return self._works
