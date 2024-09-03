"""
system.py
=========

This module defines the `PPAmorphousSystem` class, which is used to create and
manage an amorphous polypropylene (PP) system for molecular simulations.

The `PPAmorphousSystem` class is a subclass of `PolymerSystem` and provides
specific implementations for constructing an amorphous PP system, including
adding backbone and methyl carbons, and adding various types of hydrogens
specific to the PP system.

Classes
-------
PPAmorphousSystem
    A subclass of `PolymerSystem` that provides specific implementations for
    constructing an amorphous PP system.

Methods
-------
__init__(self, Nc, Nm, random_variation=True)
    Initialize the settings and call the parent `PolymerSystem` constructor.

_add_carbons(self)
    Adds backbone and methyl carbons specific to the PP system.

_add_hydrogens(self)
    Adds hydrogen atoms specific to the PP system.

_add_initiator(self, i, bond_vec)
    Adds a hydrogen atom as an initiator to the first carbon.

_add_terminator(self, i, bond_vec)
    Adds a hydrogen atom as a terminator to the last carbon.

__add_methyl_carbons(self)
    Adds the carbon atoms of methyl groups in the chain to the system.

__add_chiral_hydrogen(self, i, bond_vec)
    Adds a chiral/pendant hydrogen to each chiral carbon (C1) in the system.

Dependencies
------------
- numpy
- sympy
- math
- generation.lib.polymer_system

"""


import numpy
from numpy import linalg, random, cross, dot
from sympy import symbols, Eq, solve
from math import cos, sin, pi, radians
from generation.lib.polymer_system import PolymerSystem


class PPAmorphousSystem(PolymerSystem):
    """
    A class to represent an amorphous polypropylene (PP) system.

    This class inherits from `PolymerSystem` and provides specific
    implementations for adding carbons and hydrogens, which are unique to PP.
    It includes methods for constructing the PP system, including adding
    backbone carbons, methyl carbons, and specific hydrogens. The class also
    contains private methods that are used internally to handle specific
    molecular geometry calculations unique to PP.

    Attributes
    ----------
    settings : Settings
        Configuration settings specific to the PP system.

    Methods
    -------
    __init__(self, Nc, Nm, random_variation=True)
        Initializes the `PPAmorphousSystem` with specific settings.

    _add_carbons(self)
        Adds backbone and methyl carbons specific to the PP system.

    _add_hydrogens(self)
        Adds hydrogen atoms specific to the PP system.

    _add_initiator(self, i, bond_vec)
        Adds a hydrogen atom as an initiator to the first carbon.

    _add_terminator(self, i, bond_vec)
        Adds a hydrogen atom as a terminator to the last carbon.

    __add_methyl_carbons(self)
        Adds the carbon atoms of methyl groups in the chain to the system.

    __add_chiral_hydrogen(self, i, bond_vec)
        Adds a chiral/pendant hydrogen to each chiral carbon (C1) in the system.
    """
    class Settings():
        """
        A nested class to hold the settings specific to the PP system.

        Attributes
        ----------
        Nc : int
            Number of chains per system.
        Nm : int
            Number of monomers per chain.
        random_variation : bool
            Whether to apply random variation to torsion angles.
        density : float
            Density of the amorphous PP system.
        monomer_atom_numbers : dict
            Number of atoms per monomer, categorized by element.
        forcefield_atom_types : dict
            Mapping of atom types to integer identifiers.
        angle_size : dict
            Bond angles in radians for the PP system.
        """
        def __init__(self, Nc, Nm, random_variation=True):
            self.Nc = Nc
            self.Nm = Nm
            self.random_variation = random_variation
            self.density = 0.855
            # PP monomer has 6 atoms: 2 Carbons and 4 Hydrogens
            self.monomer_atom_numbers = dict(C=3, H=6)
            # Each chain has an initiator and a terminator
            # Types of different forcefield atoms in the system
            self.forcefield_atom_types = dict(C1=0, C2=1, C3=2, H=3)
            # Bond-angles of PP in rad (degrees) (Antoniadis, 1998)
            # R-C-C (C3C1C2) = C-C-R (C2C1C3) = 1.9465 rad = 111.5262 deg
            # aC-C-C (C1C2C1) = 1.8778 rad = 107.5900 deg
            # cC-C-C (C2C1C2) = 1.9380 rad = 111.0392 deg
            # R-C-R (C3C1C3) = 2.0560 rad = 117.8001 deg - initiator and terminator
            self.angle_size = dict(CCC=1.8778, RCC=1.9465, HCC=1.2800)

    def __init__(self, Nc, Nm, random_variation):
        """
        Initialize the PPAmorphousSystem with the provided settings.

        Parameters
        ----------
        Nc : int
            Number of chains per system.
        Nm : int
            Number of monomers per chain.
        random_variation : bool, optional
            Whether to apply random variation to torsion angles (default is True).
        """
        self.settings = self.Settings(Nc, Nm, random_variation)
        self.__methyls   = []
        super().__init__(self.settings)

    def _add_carbons(self):
        """
        Add backbone and methyl carbons specific to the PP system.

        This method overrides the `__add_carbons` placeholder method in the
        PolymerSystem base class.
        """
        atom_id = 0
        # Each monomer has 2 backbones and a methyl C with its C id being 3N in chains
        for nc in range(self.Nc):
            self._add_backbone_carbons()
            self.__add_methyl_carbons()
            for  i in range(self.Nm):
                # Add methyl carbons
                self._add_atom(atom_id, nc, 'C3', self.__methyls[i])
                self._add_bond(atom_id, atom_id+1)
                if i == 0:
                    # After adding a hydrogen as an initiator C1 becomes C2
                    self._add_atom(atom_id+1, nc, 'C2', self._backbones[2*i])
                    self._add_bond(atom_id+1, atom_id+2)
                else:
                    # Add chiral carbons
                    self._add_atom(atom_id+1, nc, 'C1', self._backbones[2*i])
                    self._add_bond(atom_id+1, atom_id+2)
                if i == self.Nm - 1:
                    # After adding a hydrogen as a terminator C2 becomes C3
                    self._add_atom(atom_id+2, nc, 'C3', self._backbones[2*i+1])
                else:
                    # Add gemini carbons
                    self._add_atom(atom_id+2, nc, 'C2', self._backbones[2*i+1])
                    self._add_bond(atom_id+2, atom_id+4)
                atom_id += 3
        self._atom_id += atom_id
        self._construct_bond_table()

    def _add_hydrogens(self):
        """
        Add hydrogen atoms specific to the PP system.

        This method overrides the `__add_hydrogens` placeholder method in the
        PolymerSystem base class.
        """
        nc = 1
        for i, bonded in enumerate(self.bond_table):
            if not bonded:
                continue
            # Bond vectors from atom i to atom in each bond
            bond_vec = [self._bond_vector(i, b) for b in bonded]
            if i % (self.Nm*3) == 1:
                self._add_initiator(i, bond_vec)
            elif i == nc*self.Nm*3 - 1:
                self._add_terminator(i, bond_vec)
                nc += 1
            elif len(bonded) == 1:
                self._add_methyl_hydrogens(i, bond_vec)
            elif len(bonded) == 2:
                self._add_gemini_hydrogens(i, bond_vec)
            elif len(bonded) == 3:
                self.__add_chiral_hydrogen(i, bond_vec)
        self._construct_bond_table()

    def _add_initiator(self, i, bond_vec):
        """
        Add a hydrogen atom as an initiator to the first carbon.

        This method overrides the `_add_initiator` placeholder method in the
        PolymerSystem class.

        Parameters
        ----------
        i : int
            Index of the current carbon atom.
        bond_vec : list of numpy.ndarray
            List of bond vectors for the current atom.
        """
        # The initiator will be similar to a gemini
        self._add_gemini_hydrogens(i, bond_vec)

    def _add_terminator(self, i, bond_vec):
        """
        Add a hydrogen atom as a terminator to the last carbon.

        This method overrides the `_add_terminator` placeholder method in the
        PolymerSystem class.

        Parameters
        ----------
        i : int
            Index of the current carbon atom.
        bond_vec : list of numpy.ndarray
            List of bond vectors for the current atom.
        """
        # The terminator will be similar to a methyl group
        self._add_methyl_hydrogens(i, bond_vec)

    def __add_methyl_carbons(self):
        """
        Add the carbon atoms of methyl groups in the chain to the system.

        This private method calculates the positions of the methyl carbon atoms
        along the polymer backbone. It randomly selects the first methyl carbon's
        position and then iteratively determines the positions of subsequent
        methyl carbons based on bond vectors and angles.

        Returns
        -------
        None
        """
        # Randomly select the first methyl carbon of the chain
        R0 = self._normalized(random.randn(3)) * self._bond_length['CC']
        coords = [self._backbones[0] + R0]
        for i in range(2, 2*self.Nm, 2):
            x, y, z = symbols('x, y, z')
            # Bond vector between backbone carbons Ci and Ci-1
            b1 = self._backbones[i-1] - self._backbones[i]
            # Bond vector between backbone carbons Ci and Ci+1
            b2 = self._backbones[i+1] - self._backbones[i]
            # 3 equations and 3 unknowns x, y, z
            a1 = self._bond_length['CC']**2 * cos(pi - self._angle_size['RCC'])
            eq1 = Eq((b1[0]*x + b1[1]*y + b1[2]*z), a1)
            a2 = self._bond_length['CC']**2 * cos(pi - self._angle_size['RCC'])
            eq2 = Eq((b2[0]*x + b2[1]*y + b2[2]*z), a2)
            a3 = self._bond_length['CC']**2
            eq3 = Eq((x*x + y*y + z*z), a3)
            x, y, z = solve((eq1, eq2, eq3), (x, y, z))[0]
            R = numpy.array([x, y, z], dtype=numpy.float64)
            assert round(linalg.norm(R), 2) == self._bond_length['CC']
            coords.append(self._backbones[i] + R)
        self.__methyls = coords

    def __add_chiral_hydrogen(self, i, bond_vec):
        """
        Add a chiral/pendant hydrogen to each chiral carbon (C1) in the system.

        This private method calculates and adds the position of a chiral hydrogen
        atom to the chiral carbon atom (C1) in the system. The position is
        determined by the bond vectors and ensuring that the vector is perpendicular
        to the plane formed by adjacent atoms.

        Parameters
        ----------
        i : int
            Index of the current carbon atom (C1).
        bond_vec : list of numpy.ndarray
            List of bond vectors for the current atom.

        Returns
        -------
        None
        """
        # Atom C_i is connected to C_i-1 and C_i+i and Ri.
        # Vector n is perpendicular to plane containing C_i+i, C_i-1, and R.
        n = self._normalized(cross(bond_vec[1] - bond_vec[0], bond_vec[2] - bond_vec[0]))
        # Vector n should point away from average of other bond vectors.
        if dot(n, sum(bond_vec)) > 0:
            n *= -1
        rH = self.atom_coords[i] + self._bond_length['CH']*n
        self._add_atom(self._atom_id, self.atom_chains[i], 'H', rH)
        self._add_bond(i, self._atom_id)
        self._atom_id += 1

