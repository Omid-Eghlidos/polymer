"""
polymer_system.py
=================

This module defines the `PolymerSystem` class, which is used to generate and
manipulate a system of polymer chains for molecular dynamics simulations.
The system includes chains of monomers, each consisting of carbon and hydrogen atoms,
and it can be constructed with specified bond lengths, angles, and dihedral angles.
The class supports both the creation and manipulation of the polymer structure,
including handling periodic boundary conditions (PBC).

Classes
-------
PolymerSystem
    A class for generating and managing a polymer system for simulations.

Dependencies
------------
- numpy
- scipy.constants
- scipy.spatial.transform
- math
- itertools
"""


import numpy
from numpy import linalg,  dot


class PolymerSystem():
    """
    A class to represent a polymer system for molecular dynamics simulations.

    The `PolymerSystem` class generates a polymer system containing a specified
    number of polymer chains and monomers per chain. It sets up the simulation
    box, adds carbon and hydrogen atoms, and defines the bonding structure,
    including bond lengths, angles, dihedrals, and improper torsions.

    Attributes
    ----------
    box : numpy.ndarray
        The simulation box dimensions.
    atom_types : numpy.ndarray
        Array of atom types for each atom in the system.
    atom_chains : numpy.ndarray
        Array of chain indices for each atom.
    atom_charges : numpy.ndarray
        Array of atomic charges for each atom in the system.
    atom_coords : numpy.ndarray
        Array of atomic coordinates for each atom in the system.
    bonds : list
        List of pairs of bonded atoms.
    unique_bond_types : list
        List of unique bond types in the system.
    bond_types : list
        List of bond types for each bond.
    bond_table : list
        Table of atoms bonded to each atom.
    angles : list
        List of angles (three bonded atoms) in the system.
    angle_types : list
        List of angle types for each angle.
    dihedrals : list
        List of dihedral torsion angles (four bonded atoms) in the system.
    dihedral_types : list
        List of dihedral types for each dihedral.
    impropers : list
        List of improper angles (four bonded atoms) in the system.
    improper_types : list
        List of improper angle types for each improper.

    Methods
    -------
    _build_system():
        Constructs the polymer system by setting up atoms, bonds, angles, and dihedrals.
    _add_atom(atom_id, num_chain, atom_type, atom_coords):
        Adds an atom to the system.
    _add_bond(i, j):
        Adds a bond between two atoms.
    _construct_bond_table():
        Creates a table of atoms bonded to each atom in the system.
    _normalized(r):
        Normalizes a given vector.
    _bond_vector(i, j):
        Calculates the bond vector between two atoms considering periodic boundary conditions.
    """
    def __init__(self):
        """
        Initialize a PolymerSystem instance.

        This method initializes the polymer system with the given settings,
        which include the number of chains, monomers per chain, atom types,
        bond lengths, angles, and the force field. It also initializes the
        necessary data structures for storing atom coordinates, bonds, angles,
        dihedrals, and impropers.

        Parameters
        ----------
        settings : object
            An object containing the configuration settings for the polymer system,
            such as the number of chains (Nc), number of monomers per chain (Nm),
            atom types, bond lengths, angles, and force field parameters.
        """
        # Simulation box
        self.box = numpy.zeros((3, 3))
        # Each atom's type, chain number, charge, and coordinates
        self.atoms = []
        self.atom_types = []
        self.atom_charges = []
        # Pair bonds and their types
        self.bonds = []
        self.bond_types = []
        self.bond_table = []
        # Angles (three bonded atoms) and their types
        self.angles = []
        self.angle_types = []
        # Dihedrals (four bonded atoms) and their types
        self.dihedrals = []
        self.dihedral_types = []
        # Impropers (four bonded atoms) and their types
        self.impropers = []
        self.improper_types = []

        # Forcefield unique types for atoms, bonds, angles, torsion, and impropers
        self._ff_types = dict(atoms=dict(), bonds=dict(), angles=(),
                              torsions=dict(), impropers=())
        # Forcefield atom weights, charges and equilibrium sizes (if given) for the structure
        self._ff_sizes = dict(masses=dict(), charges=dict(),
                              bonds=dict(), angles=dict(), torsions=dict())

    def _add_atom(self, atom_id, chain_id, atom_type, atom_coords):
        """
        Add the current atom to the system.

        Parameters
        ----------
        atom_id : int
            The ID of the atom.
        atom_chains : list
            List to store the chain number of each atom.
        atom_types : list
            List to store the type of each atom.
        atom_charges : list
            List to store the charge of each atom.
        atom_coords : list
            List to store the coordinates of each atom.
        num_chain : int
            Chain number.
        atom_type : str
            Type of the atom.
        atom_coords_new : numpy.ndarray
            The coordinates of the new atom.
        atom_types_dict : dict
            Dictionary of atom types.
        charge_types : dict
            Dictionary of atom charges.
        """
        if len(self._ff_sizes['charges']) != 0:
            if atom_type in self._ff_sizes['charges']:
                self.atom_charges.append(self._ff_sizes['charges'][atom_type])
            else:
                self.atom_charges.append(0)
        atom_type = self._ff_types['atoms'][atom_type]
        self.atom_types.append(atom_type)
        self.atoms.append([chain_id, atom_type, atom_coords])

    def _add_bond(self, i, j, covalent=None):
        """
        Add a bond and its bond type.

        Parameters
        ----------
        i : int
            Index of the first atom.
        j : int
            Index of the second atom.
        """
        bond_type = self._get_bond_type(i, j, covalent)
        self.bond_types.append(bond_type)
        self.bonds.append((bond_type, sorted([i, j])))

    def _construct_bond_table(self):
        """
        Create a table where each row contains atoms bonded to an atom with ID
        as the row number.

        Parameters
        ----------
        Nt : int
            Total number of atoms.
        bonds : list
            List of bonds.

        Returns
        -------
        list
            Bond table.
        """
        self.bonds = sorted(self.bonds, key = lambda x: x[1][0])
        self.bond_table = [[] for _ in range(self.Nt)]
        for b in self.bonds:
            i, j = b[1]
            self.bond_table[i].append(j)
            self.bond_table[j].append(i)
        for bond in self.bond_table:
            bond.sort()

    def _add_angles(self):
        """
        Traverses the bond list and defines all bond angles.

        Parameters
        ----------
        angles : list
            List to store the angles.
        angle_types : list
            List to store the angle types.
        bonds : list
            List of bonds.
        atom_types : list
            List of atom types.
        """
        for j, bonded_to_j in enumerate(self.bond_table):
            for i in bonded_to_j:
                for k in bonded_to_j:
                    # Don't add angle twice.
                    if i < k:
                        self.angles.append(self._get_angle_type([i, j, k]))
        self.angles = sorted(self.angles, key = lambda x: x[1][1])

    def _add_dihedrals(self):
        """
        Traverses the bond list and defines all the dihedral torsion angles.

        Parameters
        ----------
        bonds : list
            List of bonds.
        dihedrals : list
            List to store the dihedrals.
        dihedral_types : list
            List to store the dihedral types.
        atom_types : list
            List of atom types.
        """
        for b in self.bonds:
            j, k = sorted(b[1])
            for i in self.bond_table[j]:
                for l in self.bond_table[k]:
                    if i != k and l != j:
                        self.dihedrals.append(self._get_dihedral_type([i, j, k, l]))
        self.dihedrals = sorted(self.dihedrals, key = lambda x: x[1][1])

    def _add_impropers(self):
        """
        Traverse the bond list and define all improper angles.

        Parameters
        ----------
        bonds : list of lists
            List containing pairs of bonded atoms.
        atom_types : list of int
            List of atom types.

        Returns
        -------
        list of tuples
            A list of improper angles defined by four atom indices.
        list of lists
            A list of improper angle types corresponding to each improper angle.
        """
        for j, bonded_to_j in enumerate(self.bond_table):
            # The atom needs to have at least three bonds to have an improper
            if len(bonded_to_j) < 3:
                continue
            for i in bonded_to_j:
                for k in bonded_to_j:
                    for l in bonded_to_j:
                        if i < k and i < l and l < k:
                            self.impropers.append(self._get_improper_type([i, j, k, l]))
        self.impropers = sorted(self.impropers, key = lambda x: x[1][1])

    def _normalized(self, r):
        """
        Normalize a vector.

        Parameters
        ----------
        r : numpy.ndarray
            The vector to be normalized.

        Returns
        -------
        numpy.ndarray
            The normalized vector.
        """
        return r/linalg.norm(r) if linalg.norm(r) != 0 else r

    def _bond_vector(self, i, j):
        """
        Calculate the bond vector between two atoms, considering periodic
        boundary conditions (PBC).

        Parameters
        ----------
        atoms : list
            Array of atom coordinates.
        i : int
            Index of the first atom.
        j : int
            Index of the second atom.
        box : numpy.ndarray
            The simulation box dimensions.

        Returns
        -------
        numpy.ndarray
            The bond vector.
        """
        return self._wrap_vector(self.atoms[j][2] - self.atoms[i][2])

    def _wrap_vector(self, dx):
        """
        Return the shortest vector dx + i*A + j*B + k*C considering
        any periodic image i, j, and k.

        Parameters
        ----------
        dx : numpy.ndarray
             The vector image to wrap in the PBC.

        Returns
        -------
        numpy.ndarray
            The wrapped vector.
        """
        dxs = numpy.round(numpy.linalg.solve(self.box.T, dx))
        return dx - dot(dxs, self.box.T)

    def _map_to_PBC(self):
        """
        Maps atom coordinates back to the periodic boundary condition.

        Parameters
        ----------
        box : numpy.ndarray
            Array representing the simulation box dimensions.

        Returns
        -------
        numpy.ndarray
            Array of atom coordinates mapped to the periodic boundary condition.
        """
        for i in range(len(self.atoms)):
            for j in range(3):
                while self.atoms[i][2][j] > self.box[j,j]:
                    self.atoms[i][2][j] -= self.box[j,j]
                while self.atoms[i][2][j] < 0:
                    self.atoms[i][2][j] += self.box[j,j]

