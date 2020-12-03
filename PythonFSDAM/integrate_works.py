# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""functions for the integration of the work profiles
"""

import numpy as np

from . import bootstrapping


def integrate_work_profiles(lambda_work):
    """Integrates the work profiles with numpy.trapz

    it takes the results of a non equilibium simulation
    and integrates it returning an array of work values

    Parameters
    ------------
    lambda_work : numpy.array
        a 2-D array with 1 st dimention is lambda and the 2nd dH/dL
        [
            [lambda, lambda, lambda],
            [dH/dL, dH/dL, dH/dL]
        ]

    Returns
    ------------
    float
        the value of the integral of the work profile
    """

    work_result = np.trapz(lambda_work[1, :], lambda_work[0, :])

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

    def get_standard_deviation(self):
        """Bootstraps the standard deviation

        returns the bootstrapped standard deviation (STD) of the work values
        run this method only when you have already filled in all the
        work values otherwise you will get some nonsense result
        For confidence intervall 2*std (95%) is usually a good value

        check `bootstrapping.standard_deviation` for more info

        Returns
        ---------
        float
        """

        STD = bootstrapping.standard_deviation(self._works)

        STD = STD[0]

        return STD
