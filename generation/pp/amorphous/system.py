"""
system.py
=========

This module defines the `PPAmorphousSystem` class, which is used to create and
manage an amorphous polypropylene (PP) system for molecular simulations.

The `PPAmorphousSystem` class is a subclass of `AmorphousAlkeneSystem` and provides
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
__init__(self, settings)
    Initialize the settings and call the parent `AmorphousAlkeneSystem` constructor.

_initialize_system_parameters(self)
    Initialize system parameters such as forcefield types, bond angles, and torsions.

_build_system(self)
    Generate the polymer system, including atoms, bonds, angles, and dihedrals.

_add_carbon_atoms(self)
    Adds backbone and methyl carbons specific to the PP system.

_add_hydrogen_atoms(self)
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
from generation.lib.amorphous_alkene_system import AmorphousAlkeneSystem
from generation.lib.potential_coefficients import potential_coefficients


class PPAmorphousSystem(AmorphousAlkeneSystem):
    """
    A class to represent an amorphous polypropylene (PP) system.

    This class inherits from `AmorphousAlkeneSystem` and provides specific
    implementations for constructing an amorphous PP system, including adding
    backbone carbons, methyl carbons, and specific hydrogens. The class also
    contains private methods for molecular geometry calculations unique to PP.

    Attributes
    ----------
    resolution : str
        The resolution of the generated system.
    material : str
    	Name of the generated material system.
    Nc : int
        Number of chains per system.
    Nm : int
        Number of monomers per chain.
    random_variation : bool
        Whether to apply random variation to torsion angles.
    density : float
        Density of the amorphous PP system.
    N_C : int
        Number of Carbon atoms per monomer.
    N_H : int
        Number of Hydrogen atoms per monomer.
    forcefield_atom_types : dict
        Mapping of atom types to integer identifiers.
    angle_size : dict
        Bond angles in radians for the PP system.
    settings : dict
        Configuration settings specific to the PP system.

    Methods
    -------
    __init__(self, settings)
        Initializes the `PPAmorphousSystem` with specific settings.

    _initialize_system_parameters(self)
        Initializes system parameters, such as forcefield types and bond angles.

    _build_system(self)
        Generates the polymer system by defining atoms, bonds, angles, and torsions.

    _add_carbon_atoms(self)
        Adds backbone and methyl carbons specific to the PP system.

    _add_hydrogen_atoms(self)
        Adds hydrogen atoms specific to the PP system.

    _add_initiator(self, i, bond_vec)
        Adds a hydrogen atom as an initiator to the first carbon.

    _add_terminator(self, i, bond_vec)
        Adds a hydrogen atom as a terminator to the last carbon.

    __add_methyl_carbons(self)
        Adds the carbon atoms of methyl groups to the system.

    __add_chiral_hydrogen(self, i, bond_vec)
        Adds a chiral/pendant hydrogen to each chiral carbon (C1) in the system.
    """
    def __init__(self, settings):
        """
        Initialize the PPAmorphousSystem with the provided settings.

        Parameters:
        -----------
        settings : dict
            Input settings for generating the PE amorphous system.
        """
        # Add the settings variable as attributes
        self.__dict__.update(settings)

        # Resolution of the generated system
        self.resolution = 'AA'
        # Material name
        self.material = 'PP'
        # Density of the amorphous system (g/cm^3)
        self.rho = 0.865
        # PP monomer has 9 atoms: 3 Carbons and 6 Hydrogens
        # Number of Carbon atoms in each PE monomer
        self.N_C = 3
        # Number of Hydrogen atoms in each PE monomer
        self.N_H = 6
        # Total number of atoms in the system + an initiator and a terminator for each chain
        self.Nt = self.Nc * self.Nm * (self.N_C + self.N_H) + 2 * self.Nc
        # Random variation to be added to dihedral angles for generation
        self.random_variation = True
        # Randomly flip the dihedral angles
        self.random_flip = False

        # Store the methyl carbons
        self.__methyl_carbons = []

        # Add shared attributes from AmorphousAlkeneSystem
        super().__init__()

        # Initialize various parameters for the system
        self._initialize_system_parameters()
        # Construct the system with the given size and store its parameters
        self._build_system()

    def _initialize_system_parameters(self):
        """
        Initialize system's constant parameters including forcefield types including for
        atoms, bonds, angles, dihedrals, and impropers.
        """
        # Forcefield unique types for atoms, bonds, angles, torsions, and impropers
        self._ff_types['atoms'] = dict(C1=0, C2=1, C3=2, H=3)
        self._identify_unique_bond_types()
        # Bond-angles of PP in rad (degrees) (Antoniadis, 1998)
        # R-C-C (C3C1C2) = C-C-R (C2C1C3) = 1.9465 rad = 111.5262 deg
        # aC-C-C (C1C2C1) = 1.8778 rad = 107.5900 deg
        # cC-C-C (C2C1C2) = 1.9380 rad = 111.0392 deg
        # R-C-R (C3C1C3) = 2.0560 rad = 117.8001 deg - Trigonal planar, e.g., terminator
        self._ff_sizes['angles'] = dict(CCC=1.9380, RCC=1.9465, HCC=1.2800)
        # Dihedral angles
        self._ff_sizes['torsions'] = [numpy.pi]

    def _build_system(self):
        """
        Generate the polymer system with specified parameters.

        This method builds a polymer system containing `Nc` chains, each with
        `9 * Nm` atoms, by defining the appropriate bond lengths, bond angles,
        and dihedral angles. The method performs the following steps:

        - Creates the simulation box.
        - Adds carbon atoms.
        - Adds hydrogen atoms.
        - Maps atom coordinates to the periodic boundary condition.
        - Adds angles between bonded atoms.
        - Adds dihedral torsion angles.
        - Adds improper angles.

        Returns
        -------
        None
        """
        self._create_simulation_box()
        self._add_carbon_atoms()
        self._add_hydrogen_atoms()
        self._map_to_PBC()
        self._add_angles()
        self._add_dihedrals()
        self._add_impropers()

    def _add_carbon_atoms(self):
        """
        Add backbone and methyl carbons specific to the PP system.

        This method overrides the `_add_carbon_atoms` placeholder method in the
        PolymerSystem base class.
        """
        atom_id = 0
        # Each monomer has 2 backbones and a methyl C with its C id being 3N in chains
        for nc in range(self.Nc):
            self._add_backbone_carbons()
            self.__add_methyl_carbons()
            for  i in range(self.Nm):
                # Add methyl carbon (C3)
                self._add_atom(atom_id, nc, 'C3', self.__methyl_carbons[i])
                atom_id += 1
                # Add achiral carbon (C1)
                # After adding a hydrogen as an initiator C1 becomes C2
                C_type = 'C2' if i == 0 else 'C1'
                self._add_atom(atom_id, nc, C_type, self._backbones[2*i])
                # C1 is bonded to the previous C3 and after the first monomer to previous C2
                self._add_bond(atom_id-1, atom_id)
                if i > 0:
                    self._add_bond(atom_id-2, atom_id)
                atom_id += 1
                # Add chiral/gemini carbons (C2)
                # After adding a hydrogen as a terminator C2 becomes C3
                C_type = 'C3' if i == self.Nm - 1 else 'C2'
                self._add_atom(atom_id, nc, C_type, self._backbones[2*i+1])
                self._add_bond(atom_id-1, atom_id)
                atom_id += 1
        self._atom_id += atom_id
        self._construct_bond_table()

    def _add_hydrogen_atoms(self):
        """
        Add hydrogen atoms specific to the PP system.

        This method overrides `_add_hydrogen_atoms` in the base class to add
        hydrogens, including chiral hydrogens for each chiral carbon (C1).
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
        # Create the bond table for all the atoms in the system
        self._construct_bond_table()

    def _add_initiator(self, i, bond_vec):
        """
        Add a hydrogen atom as an initiator to the first carbon.

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

        Determines the positions of methyl carbon atoms based on bond vectors
        and angles, ensuring accurate placement along the polymer backbone.
        """
        # Randomly select the first methyl carbon of the chain
        R0 = self._normalized(random.randn(3)) * self._ff_sizes['bonds']['CC']
        coords = [self._backbones[0] + R0]
        for i in range(2, 2*self.Nm, 2):
            x, y, z = symbols('x, y, z')
            # Bond vector between backbone carbons Ci and Ci-1
            b1 = self._backbones[i-1] - self._backbones[i]
            # Bond vector between backbone carbons Ci and Ci+1
            b2 = self._backbones[i+1] - self._backbones[i]
            # 3 equations and 3 unknowns x, y, z
            a1 = self._ff_sizes['bonds']['CC']**2 * cos(pi - self._ff_sizes['angles']['RCC'])
            eq1 = Eq((b1[0]*x + b1[1]*y + b1[2]*z), a1)
            a2 = self._ff_sizes['bonds']['CC']**2 * cos(pi - self._ff_sizes['angles']['RCC'])
            eq2 = Eq((b2[0]*x + b2[1]*y + b2[2]*z), a2)
            a3 = self._ff_sizes['bonds']['CC']**2
            eq3 = Eq((x*x + y*y + z*z), a3)
            x, y, z = solve((eq1, eq2, eq3), (x, y, z))[0]
            R = numpy.array([x, y, z], dtype=numpy.float64)
            assert round(linalg.norm(R), 2) == self._ff_sizes['bonds']['CC']
            coords.append(self._backbones[i] + R)
        self.__methyl_carbons = coords

    def __add_chiral_hydrogen(self, i, bond_vec):
        """
        Add a chiral/pendant hydrogen to each chiral carbon (C1) in the system.

        Calculates the position of a chiral hydrogen atom to ensure it is perpendicular
        to the plane formed by adjacent atoms.

        Parameters
        ----------
        i : int
            Index of the current carbon atom (C1).
        bond_vec : list of numpy.ndarray
            List of bond vectors for the current atom.
        """
        # Atom C_i is connected to C_i-1 and C_i+i and Ri.
        # Vector n is perpendicular to plane containing C_i+i, C_i-1, and R.
        n = self._normalized(cross(bond_vec[1] - bond_vec[0], bond_vec[2] - bond_vec[0]))
        # Vector n should point away from average of other bond vectors.
        if dot(n, sum(bond_vec)) > 0:
            n *= -1
        rH = self.atoms[i][2] + self._ff_sizes['bonds']['CH']*n
        self._add_atom(self._atom_id, self.atoms[i][0], 'H', rH)
        self._add_bond(i, self._atom_id)
        self._atom_id += 1

