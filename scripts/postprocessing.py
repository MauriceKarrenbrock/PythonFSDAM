# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
# pylint: disable=duplicate-code
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""script to do post processing of a FSDAM or a vDSSB

use --help for usage info
"""

import argparse
import shutil
from pathlib import Path

from PythonFSDAM.pipelines import superclasses

parser = argparse.ArgumentParser(
    description='This script will post process everything needed after '
    'FSDAM or vDSSB with the wanted md program use --help for usage info '
    'Parallelism may be used, use OMP_NUM_THREADS environment variable to limit '
    'the number of used GPUs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--md-program',
                    action='store',
                    type=str,
                    default='gromacs',
                    help='The MD program to use')

parser.add_argument('--kind-of-process',
                    action='store',
                    type=str,
                    default='standard',
                    choices=['standard', 'vdssb'],
                    help='if you are doing a standard FSDAM or a vDSSB')

parser.add_argument(
    '--unbound-file',
    action='store',
    type=str,
    help=
    'the path to the file that contains the names of the energy files resulting from the unbound (ligand) process, something like unbound/file.dat\n'
    'IF THEY ARE 2 LIKE IN GROMACS USE A COMMA SEPARATED LIST!')

parser.add_argument(
    '--bound-file',
    action='store',
    type=str,
    help=
    'the path to the file that contains the names of the energy files resulting from the bound (protein-ligand) process, something like bound/file.dat\n'
    'IF THEY ARE 2 LIKE IN GROMACS USE A COMMA SEPARATED LIST!')

parser.add_argument('--temperature',
                    action='store',
                    type=float,
                    default=298.15,
                    help='temperature in Kelvin (K)')

parsed_input = parser.parse_args()

#Deal with unbound files
unbound_dir = parsed_input.unbound_file.split(',')
for i, directory in enumerate(unbound_dir):

    unbound_dir[i] = Path(directory.rsplit('/', 1)[0].strip()).resolve()

unbound_files = []
for n, i in enumerate(parsed_input.unbound_file.split(',')):

    tmp_list = []

    with open(i, 'r') as f:

        for line in f:

            if line.strip():

                tmp_list.append(unbound_dir[n] / Path(line.strip()))

    unbound_files.append(tmp_list)

if len(unbound_files) == 1:

    unbound_files = unbound_files[0]

#Deal with bound files
bound_dir = parsed_input.bound_file.split(',')
for i, directory in enumerate(bound_dir):

    bound_dir[i] = Path(directory.rsplit('/', 1)[0].strip()).resolve()

bound_files = []
for n, i in enumerate(parsed_input.bound_file.split(',')):

    tmp_list = []

    with open(i, 'r') as f:

        for line in f:

            if line.strip():

                tmp_list.append(bound_dir[n] / Path(line.strip()))

    bound_files.append(tmp_list)

if len(bound_files) == 1:

    bound_files = bound_files[0]

if parsed_input.kind_of_process == 'standard':

    unbound_obj = superclasses.JarzynskiPostProcessingAlchemicalLeg(
        unbound_files,
        temperature=parsed_input.temperature,
        md_program=parsed_input.md_program,
        creation=False)

    unbound_free_energy, unbound_std = unbound_obj.execute()

    shutil.move(f'{str(unbound_obj)}_free_energy.dat',
                'unbound_' + f'{str(unbound_obj)}_free_energy.dat')

    bound_obj = superclasses.JarzynskiPostProcessingAlchemicalLeg(
        bound_files,
        temperature=parsed_input.temperature,
        md_program=parsed_input.md_program,
        creation=False)

    bound_free_energy, bound_std = bound_obj.execute()

    shutil.move(f'{str(bound_obj)}_free_energy.dat',
                'bound_' + f'{str(bound_obj)}_free_energy.dat')

    total_free_energy = bound_free_energy - unbound_free_energy

    total_std = bound_std + unbound_std

    with open('total_free_energy.dat', 'w') as f:

        f.write(
            '#total unbinding free energy (no volume correction done) in Kcal mol\n'
            '#Dg   CI95%  STD\n'
            f'{total_free_energy:.18e} {1.96*(total_std):.18e} {total_std:.18e}\n'
        )

        print(f'free energy {total_free_energy}\n' f'CI95 {1.96*(total_std)}')

elif parsed_input.kind_of_process == 'vdssb':

    obj = superclasses.JarzynskiVDSSBPostProcessingPipeline(
        bound_files,
        unbound_files,
        temperature=parsed_input.temperature,
        md_program=parsed_input.md_program)

    free_energy, std = obj.execute()

    print(f'free energy {free_energy:.18e}\nCI95 {std*1.96:.18e}')

else:
    ValueError('Unknown input')
