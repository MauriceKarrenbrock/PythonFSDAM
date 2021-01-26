# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2020 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""functions to combine bound and unbound works
"""

import numpy as np


def combine_non_correlated_works(works_1, works_2):
    """combines 2 non correlated sets of work values

    If you have 2 set of work values (for example
    bound and unbound in the case of vDSSB) that are
    un correlated you can combine them in order to
    get N * M resulting works. It is equivalent to
    convoluting the 2 probability distributions.

    Parameters
    ------------
    works_1 : numpy.array
        the first set of works values to combine
    works_2 : numpy.array
        the second set of works values to combine

    Returns
    -----------
    numpy.array :
        a 1-D array N * M long containing the combined
        work values

    Notes
    ---------
    for more information check out this paper:

    Virtual Double-System Single-Box: A Nonequilibrium
    Alchemical Technique for Absolute Binding Free Energy
    Calculations: Application to Ligands of the SARS-CoV-2 Main Protease
    Marina Macchiagodena, Marco Pagliai, Maurice Karrenbrock,
    Guido Guarnieri, Francesco Iannone, and Piero Procacci
    Journal of Chemical Theory and Computation 2020 16 (11), 7160-7172
    DOI: 10.1021/acs.jctc.0c00634

    section 2 "THEORETICAL BACKGROUND"
    """

    #empty array N*M long
    output_array = np.empty([works_1.size * works_2.size])

    len_works_2 = len(works_2)

    i = 0
    iterator_1 = np.nditer(works_1)
    for value_1 in iterator_1:

        cutoff = i * len_works_2

        output_array[cutoff:cutoff + len_works_2] = value_1 + works_2[:]

        i += 1

    return output_array
