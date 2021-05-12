# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""This script will pre process everything needed to do FSDAM with the wanted md program

use --help for usage info
"""

import argparse
import random
import shutil
from pathlib import Path

import FSDAMGromacs.add_dummy_atom as _dummy
import mdtraj
import numpy as np

from PythonFSDAM.free_energy_calculations import volume_correction
from PythonFSDAM.pipelines.factories import PreProcessing

parser = argparse.ArgumentParser(
    description='his script will pre process everything needed to do '
    'FSDAM with the wanted md program use --help for usage info')

#parser = argparse.ArgumentParser(description="This script creates the input for FS-DAM both for the boded and unbonded system. For Gromacs\n You must run this script in the HREM root directory")

parser.add_argument('--md-program',
                    action='store',
                    default='gromacs',
                    help='The MD program to use')

parser.add_argument('--md-program-path',
                    action='store',
                    default='gmx',
                    help='The absolute path to the chosen program executable')

parser.add_argument(
    '--creation',
    action='store',
    default=True,
    choices=[True, False],
    help=
    'If the alchemical transformation shall be an annihilation or a creation')

parser.add_argument(
    '--number-of-frames-to-use',
    action='store',
    default=200,
    help=
    'The number of alchemical transformations to do, a good default is 200 protein ligand and 400 ligand only, default=200'
)

parser.add_argument(
    '--kind-of-system',
    action='store',
    default='only-ligand',
    choices=['only-ligand', 'solvated-ligand', 'protein-ligand'],
    help=
    'if you set it to protein-ligand instead of the default (ligand) an harmonic COM-COM restaint will be set between the ligand and the protein'
)

parser.add_argument(
    '--alchemical-region',
    action='store',
    help=
    'The syntax depends on the MD program, in gromacs you shall write the residue name (resname) of what you want to create/annihilate'
)

parser.add_argument(
    '--harmonic-kappa',
    action='store',
    default=120,
    help=
    'the harmonic constant in case an harmonic restraint has to be introduced in (KJ/mol)/A**2'
)

parser.add_argument(
    '--structure-files-type',
    action='store',
    default='pdb',
    choices=['pdb', 'gro'],
    help=
    'if the structure files are pdb, gro,... all the files in the current directory with this suffix will be used'
)

parser.add_argument(
    '--topology-file',
    action='store',
    default='system.top',
    help=
    'the topology file of the system (es gromacs top file or openMM xml file)')

parser.add_argument(
    '--water-box',
    action='store',
    default=None,
    help=
    'in case --kinf-of-system is only-ligand you must give the structure file (pdb, gro, ...) of the needed water box (must be equilibrated)'
)

parser.add_argument('--temperature',
                    action='store',
                    default=298.15,
                    help='temperature in Kelvin (K)')

parser.add_argument(
    '--protein-select-tring',
    action='store',
    default='protein',
    help=
    'a mdtraj select string that identifies the protein, it is useful when protein (default) does not behave as expected'
)

parser.add_argument(
    '--add-dummy-atom',
    action='store',
    default='no',
    choices=['yes', 'no'],
    help=
    'if yes a heavy dummy atom (DUM) will be added on the cemter of mass of the protein and a COM-COM restrain added between the protein and the ligand'
)

parsed_input = parser.parse_args()

# get the structure files
input_files = parsed_input.structure_files_type
input_files = Path('.').glob(f'*.{input_files.strip()}')
input_files = list(input_files)

#keep only the requested number of structures
not_used_dir = Path('./not_used_frames')
not_used_dir.mkdir(parents=True, exist_ok=True)

#randomly shuffle the structure files
random.shuffle(input_files)

for i in range(len(input_files) - int(parsed_input.number_of_frames_to_use)):

    shutil.move(str(input_files.pop(-1)),
                str(not_used_dir),
                copy_function=shutil.copy)

COM_pull_groups = None

harmonic_kappa = None

if parsed_input.kind_of_system == 'protein-ligand':

    COM_pull_groups = ['Protein', parsed_input.alchemical_region]

    harmonic_kappa = [[
        'Protein', parsed_input.alchemical_region, parsed_input.harmonic_kappa
    ]]

    tmp_select = parsed_input.alchemical_region

    if parsed_input.add_dummy_atom == 'yes':
        #gromacs doesn't deal well with PBC when there are restraints
        #therefore I will add a dummy atom on the center of mass
        #of the center of mass of the protein

        _dummy.add_dummy_atom_to_topology(parsed_input.topology_file)

        for i in input_files:
            _dummy.add_dummy_atom_to_center_of_mass(
                str(i), select=parsed_input.protein_select_string)

        COM_pull_groups.append('DUM')

        harmonic_kappa.append(['DUM', 'Protein', 120])
        harmonic_kappa.append(['DUM', parsed_input.alchemical_region, 0])

        tmp_select = f'resname {parsed_input.alchemical_region}'

if parsed_input.kind_of_system == 'only-ligand':

    water = mdtraj.load(str(parsed_input.water_box))
    water.center_coordinates()
    #water.make_molecules_whole()

    for i in input_files:

        ligand = mdtraj.load(str(i))

        ligand.center_coordinates()

        #ligand.make_molecules_whole()

        joined = water.stack(ligand)  #box info are taken from left operand

        joined.save(str(i), force_overwrite=True)

pipeline_obj = PreProcessing(md_program=parsed_input.md_program,
                             topology_files=parsed_input.topology_file,
                             md_program_path=parsed_input.md_program_path,
                             alchemical_residue=parsed_input.alchemical_region,
                             structure_files=input_files,
                             COM_pull_groups=COM_pull_groups,
                             harmonic_kappa=harmonic_kappa,
                             temperature=parsed_input.temperature,
                             pbc_atoms=None,
                             constraints=None,
                             creation=parsed_input.creation)

output = pipeline_obj.execute()

if parsed_input.md_program == 'gromacs':
    with open('MAKE_VDW_TPR.sh', 'w') as f:

        for line in output['make_vdw_tpr']:
            f.write(line + '\n')

    with open('MAKE_Q_TPR.sh', 'w') as f:

        for line in output['make_q_tpr']:
            f.write(line + '\n')

    with open('RUN_Q.sh', 'w') as f:

        number_of_runs = len(output['run_q'])
        f.write(f'# there are {number_of_runs} runs to do\n')
        f.write('# I suggest to use one GPU per run\n\n\n')

        f.write('\n'.join(output['run_q']))

    with open('RUN_VDW.sh', 'w') as f:

        number_of_runs = len(output['run_vdw'])
        f.write(f'# there are {number_of_runs} runs to do\n')
        f.write('# I suggest to use one GPU per run\n\n\n')

        f.write('\n'.join(output['run_vdw']))

    with open('Q_DHDL_FILES.dat', 'w') as f:

        f.write('\n'.join(output['q_dhdl']))

    with open('VDW_DHDL_FILES.dat', 'w') as f:

        f.write('\n'.join(output['vdw_dhdl']))

    print(
        'I created some MAKE_*_TPR.sh files that must be run like bash filename.sh\n'
        'in order to make the needed .tpr files\n'
        'and some RUN*.sh files that are some examples of '
        'how you may run the simulations (on a cluster I would suggest to use job arrays though)'
    )

#calculate volume correction
if parsed_input.kind_of_system == 'protein-ligand':

    print(
        '\n\n\nI am calculating the volume correction, it will take a while\n'
        'But all the needed files to run the MD simulations are ready therefore you can go on on '
        'a different terminal window\n'
        'In the end com_com_dist_Angstrom.dat and volume_correction.dat file will be created'
    )

    str_input_files = [str(i) for i in input_files]
    traj = mdtraj.load(str_input_files, top=str_input_files[0])

    dist = mdtraj.geometry.distance.compute_center_of_mass(
        traj,
        select=tmp_select) - mdtraj.geometry.distance.compute_center_of_mass(
            traj, select='protein')

    dist = dist**2

    dist = np.sum(dist, axis=1)

    dist = dist**0.5

    dist *= 10.  #nm to A

    with open('com_com_dist_Angstrom.dat', 'w') as f:

        for i in dist:
            f.write(f'{i:.18e}\n')

    vol_correction = volume_correction(dist,
                                       temperature=parsed_input.temperature)

    del dist

    with open('volume_correction.dat', 'w') as f:

        f.write('#volume correction in Kcal/mol\n' 'f{vol_correction:.18e}\n')

print('Done')
