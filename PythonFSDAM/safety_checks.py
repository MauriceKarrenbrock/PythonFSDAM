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
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module


def _in_ellipsoid(X, center, rotation_matrix, radii):
    """private"""
    X = X.copy()
    X -= center

    X = rotation_matrix @ X

    x = X[0]
    y = X[1]
    z = X[2]

    result = (x / radii[0])**2 + (y / radii[1])**2 + (z / radii[2])**2

    if result >= -1 and result <= 1:  # pylint: disable=chained-comparison
        return True
    return False


def get_atoms_in_pocket(ligand,
                        pocket,
                        pdb_file,
                        top=None,
                        make_molecules_whole=False):
    """Get the number of ligand atoms in the given pocket

    Parameters
    -----------
    ligand : str or list(int)
        a mdtraj selection string or a list of atom indexes (0 indexed)
    pocket : str or list(int)
        a mdtraj selection string or a list of atom indexes (0 indexed)
    pdb_file : str or path or list(str or path)
        the path to any structure file supported by mdtraj.load (pdb, gro, ...)
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
    Then creates a hollow hull with ```scipy.spatial.ConvexHull```
    Then fits is with an arbitrary ellipsoid
    If at least `n_atoms_inside` atoms are inside the ellipsoid
    the ligand is still in the pocket

    Returns
    -----------
    int or list(int)
        the number of atoms in the pocket
        if more than a frame was given it will be a list
    """

    if isinstance(pdb_file, str) or not hasattr(pdb_file, '__iter__'):
        pdb_file = [pdb_file]

    #mdtraj can't manage Path objects
    pdb_file = [str(i) for i in pdb_file]

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

    atoms_in_pocket_list = []

    for ligand_frame, pocket_frame in zip(ligand_coord, pocket_coord):
        atoms_in_pocket = 0

        convex_hull = ConvexHull(pocket_frame)
        convex_hull = convex_hull.points[convex_hull.vertices]

        center, rotation_matrix, radii, _ = ellipsoid_fit(convex_hull)

        for atom in ligand_frame:
            if _in_ellipsoid(atom, center, rotation_matrix, radii):
                atoms_in_pocket += 1

        atoms_in_pocket_list.append(atoms_in_pocket)

    if len(atoms_in_pocket_list) == 1:
        return atoms_in_pocket_list[0]
    return atoms_in_pocket_list


def check_ligand_in_pocket(ligand,
                           pocket,
                           pdb_file,
                           n_atoms_inside=1,
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
    n_atoms_inside : int, optional, default=1
        how many atoms of the ligand shall be inside the pocket to be considered
        in the pocket. With the default 1 if at leas one atom of the ligand is in the defined pocket
        the ligand is considered inside
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
    Then creates a hollow hull with ```scipy.spatial.ConvexHull```
    Then fits is with an arbitrary ellipsoid
    If at least `n_atoms_inside` atoms are inside the ellipsoid
    the ligand is still in the pocket

    Returns
    -----------
    bool or list(bool)
        True if the ligand is in the pocket
        False if the ligand is outside the pocket
        If you gave a list of structures as input you
        it will return a list of bool
    """

    atoms_in_pocket_list = get_atoms_in_pocket(
        ligand=ligand,
        pocket=pocket,
        pdb_file=pdb_file,
        top=top,
        make_molecules_whole=make_molecules_whole)

    if not hasattr(atoms_in_pocket_list, '__iter__'):
        atoms_in_pocket_list = [atoms_in_pocket_list]

    is_in_pocket = []

    for atoms_in_pocket in atoms_in_pocket_list:
        if atoms_in_pocket < n_atoms_inside:
            is_in_pocket.append(False)
        else:
            is_in_pocket.append(True)

    if len(is_in_pocket) == 1:
        return is_in_pocket[0]
    return is_in_pocket


# https://github.com/aleksandrbazhin/ellipsoid_fit_python/blob/master/ellipsoid_fit.py
# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
# (Under MIT license)
def ellipsoid_fit(X):
    """fits an arbitrary ellipsoid to a set of points
    """
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([
        x * x + y * y - 2 * z * z, x * x + z * z - 2 * y * y, 2 * x * y,
        2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, 1 - 0 * x
    ])
    d2 = np.array(x * x + y * y + z * z).T  # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]], [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]], [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(-A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v
