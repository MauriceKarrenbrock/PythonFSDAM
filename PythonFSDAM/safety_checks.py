# -*- coding: utf-8 -*-
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""functions to check if everything is going OK"""

import mdtraj
import numpy as np


def check_ligand_in_pocket(ligand,
                           pocket,
                           pdb_file,
                           tollerance=0,
                           top=None,
                           make_molecules_whole=False):
    """Check if the ligand is in the pocket

    If you used some kind of enhanced sampling before the FSDAM
    or sometimes a simple equilibration the ligand may exit the
    binding pocket and therefore you want to discard the frames relative
    to this unwanted exits

    Parameters
    -----------
    ligand : str or list(int)
        a mdtraj selection string or a list of atom indexes (0 indexed)
    pocket : str or list(int)
        a mdtraj selection string or a list of atom indexes (0 indexed)
    pdb_file : str or path or list(str or path)
        the path to any structure file supported by mdtraj.load (pdb, gro, ...)
    tollerance : float, optional, default=0
        how much a ligand atom can exit the pocket in Angtrom
    top : str or path, optional
        this is the top keywor argument in mdtraj.load
        it is only needed if the structure file `pdb_file`
        doesn't contain topology information
    make_molecules_whole : bool, optional, default=False
        if True make_molecules_whole() method will be called on the
        mdtraj trajectory, I suggest not to use this option and to
        give whole molecules as input

    Notes
    ----------
    This function uses mdtraj to parse the files
    This implementation should work well for small ligands
    that are not too flexible. I don't know how well it may
    perform on more complex systems.
    What it does is checking if at leas one atom of the ligand
    is still in the pocket defined as input, with tollerance `tollerance`
    in Angstrom

    Returns
    -----------
    bool or list(bool)
        True if the ligand is in the pocket
        False if the ligand is outside the pocket
        If you gave a list of structures as input you
        it will return a list of bool
    """
    if isinstance(pdb_file, str) or not hasattr(pdb_file, '__iter__'):
        pdb_file = [pdb_file]

    #mdtraj can't manage Path objects
    pdb_file = [str(i) for i in pdb_file]

    #angstrom to nm
    tollerance /= 10

    if top is None:
        # For a more omogeneus mdtraj.load function call
        top = pdb_file[0]
    else:
        top = str(top)
    traj = mdtraj.load(pdb_file, top=top)

    #want only positive coordinates

    if make_molecules_whole:
        traj.make_molecules_whole(inplace=True)

    if isinstance(ligand, str):
        ligand = traj.top.select(ligand)
    if isinstance(pocket, str):
        pocket = traj.top.select(pocket)

    ligand_coord = traj.atom_slice(ligand).xyz
    pocket_coord = traj.atom_slice(pocket).xyz

    #free memory
    del traj
    del ligand
    del pocket

    is_in_pocket = []

    for ligand_frame, pocket_frame in zip(ligand_coord, pocket_coord):
        tmp_in_pocket = True

        # [min_x, min_y, min_z] and [max_x, max_y, max_z]
        pocket_min = np.amin(pocket_frame, axis=0) - tollerance
        pocket_max = np.amax(pocket_frame, axis=0) + tollerance

        for i in range(3):
            # If any ligand atom is outside the [min, max] intervall
            # The ligand is not in the pocket
            if np.all(np.less(ligand_frame[i, :], pocket_min[i])):
                tmp_in_pocket = False
            if np.all(np.greater(ligand_frame[i, :], pocket_max[i])):
                tmp_in_pocket = False

        is_in_pocket.append(tmp_in_pocket)

    if len(is_in_pocket) == 1:
        return is_in_pocket[0]
    return is_in_pocket
