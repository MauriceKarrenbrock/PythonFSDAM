# -*- coding: utf-8 -*-
# pylint : disable=anomalous-backslash-in-string
# pylint : ignore=docstrings
#############################################################
# Copyright (c) 2020-2021 Maurice Karrenbrock               #
#                                                           #
# This software is open-source and is distributed under the #
# BSD 3-Clause "New" or "Revised" License                   #
#############################################################
"""Formulas to calculate the free energy correction associated charged ligands
when no neutralizing counter ions are indroduced to neutralize
this methods are valid for the Ewald family (PME, SPME, ...)

References
------------
check

Quantifying Artifacts in Ewald Simulations of Inhomogeneous Systems with a Net Charge
Jochen S. Hub, Bert L. de Groot, Helmut Grubmüller, and Gerrit Groenhof
Journal of Chemical Theory and Computation 2014 10 (1), 381-390
DOI: 10.1021/ct400626b
https://pubs.acs.org/doi/abs/10.1021/ct400626b

and check

Procacci, P., Guarrasi, M. & Guarnieri, G.
SAMPL6 host–guest blind predictions using a non equilibrium alchemical approach.
J Comput Aided Mol Des 32, 965–982 (2018).
https://doi.org/10.1007/s10822-018-0151-9
"""

import math

import mdtraj
import numpy as np
import parmed
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module
from simtk import unit


def homogeneus_charge_correction_alchemical_leg(starting_charge,
                                                final_charge,
                                                volume,
                                                ewald_alpha=0.37):
    r"""Homogeneus charge correction for one alchemical leg

    Parameters
    ----------------
    starting_charge : float
        the charge of the system at the beginning of the process
    final_charge : float
        the charge of the system at the end of the process
    volume : float
        the volume of the simulation box
    ewald_alpha : float, default=0.37(1/Angstrom)
        the value of the ewald alpha in your MD run. units must be coherent with the volumes
        given. default=0.37(1/Angstrom)

    Returns
    ----------
    float
        the energy correction, if `ewald_alpha` and the volumes are in angstrom and
        the charges in atomic units the output will be in Hartree (atomic unit of energy)
        1 hartree = 627.5 Kcal/mol

    Notes
    --------------
    This correction must be added to the dissociacion free energy or removed from
    the binding free energy

    In gromacs the homogeneus correction is done under the hood by gromacs itself
    see the function ewald_charge_correction in the source file
    src/gromacs/ewald/ewald.cpp of the gromacs MD program

    The result will be

    .. math::

        \Delta G_{correction} = - \frac{\pi}{2 \alpha ^2} {\frac{Q_{F}^{2} - (Q_{S})^2}{VOL}}}
    """

    correction = (final_charge**2 - starting_charge**2) / volume
    correction *= -(math.pi / (2 * ewald_alpha * ewald_alpha))

    return correction


def homogeneus_charge_correction_vDSSB(
        host_charge,
        guest_charge,
        host_guest_box_volume,
        only_guest_box_volume,
        ewald_alpha=0.37):  # pylint : disable=line-too-long
    r"""Calculate the free energy correction of an homogeneus host guest system for vDSSB

    With homogeneus I mean that the charged ligand is very near to the surface
    of the protein. In fact if the ligand is burried in the ligand you will also
    need to add another correction too

    This correction must be added to the dissociacion free energy or removed from
    the binding free energy

    It is for the virtual double system single box vDSSB method aka the ligand gets annihilated
    from the protein and created in a box of solvent.
    If you need something else use the lower level function:
    `homogeneus_charge_correction_alchemical_leg`

    Parameters
    -----------------
    host_charge : float or iterable(float)
        the total charge of the host (protein and solvent) or the charge of all
        its atoms (will be summed)
    guest_charge : float or iterable(float)
        the total charge of the ligand or the charge of all
        its atoms (will be summed)
    host_guest_box_volume : float
        the volume of the MD box of the host guest alchemical transformation
        it must be in units that are coherent with `ewald_alpha` (default Angstrom^3)
    only_guest_box_volume : float
        the volume of the MD box of the guest in solvent alchemical transformation
        it must be in units that are coherent with `ewald_alpha` (default Angstrom^3)
    ewald_alpha : float, default=0.37(1/Angstrom)
        the value of the ewald alpha in your MD run. units must be coherent with the volumes
        given. default=0.37(1/Angstrom)

    Returns
    ----------
    float
        the energy correction, if `ewald_alpha` and the volumes are in angstrom and
        the charges in atomic units the output will be in Hartree (atomic unit of energy)
        1 hartree = 627.5 Kcal/mol

    Notes
    ----------
    The result will be

    .. math::

        \Delta G_{correction} = - \frac{\pi}{2 \alpha ^2} {\frac{Q_{H}^{2} - (Q_H + Q_G)^2}{V_{BOX}^(b)} + \frac{Q^{2}_{G}}{V_{BOX}^{(u)}}

    In gromacs the homogeneus correction is done under the hood by gromacs itself
    see the function ewald_charge_correction in the source file
    src/gromacs/ewald/ewald.cpp of the gromacs MD program

    References
    ------------
    Piero Procacci and Guido Guarnieri,
    "SAMPL9 blind predictions using nonequilibrium alchemical approaches",
    J. Chem. Phys. 156, 164104 (2022)
    https://doi.org/10.1063/5.0086640
    """ # pylint : disable=line-too-long

    bound = homogeneus_charge_correction_alchemical_leg(
        starting_charge=(host_charge + guest_charge),
        final_charge=host_charge,
        volume=host_guest_box_volume,
        ewald_alpha=ewald_alpha)

    unbound = homogeneus_charge_correction_alchemical_leg(
        starting_charge=0.,
        final_charge=guest_charge,
        volume=only_guest_box_volume,
        ewald_alpha=ewald_alpha)

    return bound + unbound


def get_charges_with_parmed(*args, **kwargs):
    """Get the charges of a protein (or ligand) through parmed

    This function uses parmed.load_file to parse wathever parameter
    file / force field file supported and gets the charges

    Parameters
    ------------
    *args
    **kwargs
        see parmed.load_file documentation

    Returns
    ----------
    numpy.array(float)
        an ordered array of charges in atomic units
    """

    parm = parmed.load_file(*args, **kwargs)

    return np.array([atom.charge for atom in parm.atoms])


def get_charges_with_openff(mol):
    """Starting from a openff molecule returns atomic charges

    If the charges are already defined will return them without
    change
    I not will calculate am1bcc charges

    Parameters
    ------------
    mol : openff.toolkit.topology.Molecule

    Examples
    ---------
    from openff.toolkit.topology import Molecule

    mol = Molecule.from_file(SOME_FILE)
    # mol = Molecule.from_smiles(SMILES)

    get_charges_with_openff(mol)

    Returns
    ------------
    np.array(float)
        charges in atomic units (elementary charge)

    Notes
    ----------
    Some extra conformers may be generated because of
    https://github.com/openforcefield/openff-toolkit/issues/492
    """

    if (mol.partial_charges is None) or (np.allclose(
            mol.partial_charges / unit.elementary_charge,
            np.zeros([mol.n_particles]))):
        # NOTE: generate_conformers seems to be required for some molecules
        # https://github.com/openforcefield/openff-toolkit/issues/492
        mol.generate_conformers(n_conformers=10)
        mol.compute_partial_charges_am1bcc()

    return mol.partial_charges.value_in_unit(unit.elementary_charge)


def globular_protein_correction(pdb_file,
                                host_charge,
                                guest_charge,
                                ligand,
                                protein='protein',
                                protein_dielectric_constant=4,
                                top=None):
    """correction for charged ligands in globular proteins

    this correction must be added to
    `correction_homogeneus_host_guest_system`
    and treats the protein as a perfect sphere
    therefore the less globular the protein is the
    less correct the result is

    Paremeters
    --------------
    host_charge : float or iterable(float)
        the total charge of the host (protein) or the charge of all
        its atoms (will be summed) in atomic units
    guest_charge : float or iterable(float)
        the total charge of the host (protein) or the charge of all
        its atoms (will be summed) in atomic units
    ligand : str or list(int)
        a mdtraj selection string or a list of atom indexes (0 indexed)
    protein : str or list(int), default=protein
        a mdtraj selection string or a list of atom indexes (0 indexed)
    protein_dielectric_constant : float, optional, default=4
        the dielectric constant of the protein, usually between 4 and 7
    top : str or path, optional, default=None
        this is the top keywor argument in mdtraj.load
        it is only needed if the structure file `pdb_file`
        doesn't contain topology information

    Returns
    ---------
    float
        energy in Hartree (atomic unit of energy)
        1 hartree = 627.5 Kcal/mol

    References
    ------------
    Quantifying Artifacts in Ewald Simulations of Inhomogeneous Systems with a Net Charge
    Jochen S. Hub, Bert L. de Groot, Helmut Grubmüller, and Gerrit Groenhof
    Journal of Chemical Theory and Computation 2014 10 (1), 381-390
    DOI: 10.1021/ct400626b
    https://pubs.acs.org/doi/abs/10.1021/ct400626b

    This function implements function n 24 of the paper
    """

    if hasattr(host_charge, '__iter__'):
        host_charge = sum(host_charge)
    if hasattr(guest_charge, '__iter__'):
        guest_charge = sum(guest_charge)

    pdb_file = str(pdb_file)

    if top is None:
        # For a more omogeneus mdtraj.load function call
        top = pdb_file
    else:
        top = str(top)

    traj = mdtraj.load(pdb_file, top=top)

    unit_cell_volume = traj.unitcell_volumes[0] * 1000  # nm^3 to Angstrom^3

    background_charge_density = -(host_charge +
                                  guest_charge) / unit_cell_volume

    result = 2 * math.pi * guest_charge * background_charge_density
    result /= 3 * protein_dielectric_constant

    if isinstance(ligand, str):
        ligand = traj.top.select(ligand)
    if isinstance(protein, str):
        protein = traj.top.select(protein)

    ligand_center_of_mass = mdtraj.geometry.compute_center_of_mass(
        traj.atom_slice(ligand))[0] * 10  # nm to Angstrom
    protein_center_of_mass = mdtraj.geometry.compute_center_of_mass(
        traj.atom_slice(protein))[0] * 10  # nm to Angstrom

    protein_coord = traj.atom_slice(protein).xyz[0] * 10  # nm to Angstrom

    protein_volume = ConvexHull(protein_coord).volume

    # the radius of a sphere with the same volume
    protein_radius = ((3 * protein_volume) / (4 * math.pi))**(1 / 3)

    # protein ligand center of mass distance
    r_ligand = ligand_center_of_mass - protein_center_of_mass
    r_ligand = r_ligand**2
    r_ligand = sum(r_ligand)**0.5

    result *= (protein_radius**2 - r_ligand**2)

    return result
