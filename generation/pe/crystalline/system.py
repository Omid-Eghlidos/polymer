"""
system.py
=========

This module defines the `PECrystallineSystem` class, which is used to create and
manage a crystalline polyethylene (PE) system for molecular simulations.

The `PECrystallineSystem` class is a subclass of `CrystallineAlkeneSystem` and provides
specific implementations for constructing a crystalline PE system, including
defining unit cells, adding carbon and hydrogen atoms, and managing space group
symmetry operations.

Classes
-------
PECrystallineSystem
    A subclass of `CrystallineAlkeneSystem` that provides specific implementations
    for constructing a crystalline PE system.

Methods
-------
__init__(self, settings)
    Initialize the settings and call the parent `CrystallineAlkeneSystem` constructor.

_initialize_system_parameters(self)
    Initialize system parameters such as forcefield types, dihedral angles, etc.

_build_system(self)
    Generate the polymer system, including atoms, bonds, and angles.

_unit_cell(self)
    Defines the unit cell structure for the crystalline PE system.

_space_group(self)
    Applies space group symmetry operations to construct the unit cell.

__orthorhombic(self)
    Returns parameters for the orthorhombic unit cell of crystalline PE.

__Pnam(self)
    Defines fractional coordinates for atoms in the orthorhombic unit cell of PE.

Dependencies
------------
- generation.lib.crystalline_alkene_system
"""

import numpy
from generation.lib.crystalline_alkene_system import CrystallineAlkeneSystem


class PECrystallineSystem(CrystallineAlkeneSystem):
    """
    A class to represent a crystalline polyethylene (PE) system.

    This class inherits from CrystallineAlkeneSystem and provides specific
    implementations for constructing a crystalline PE system, including
    defining unit cells, adding carbon and hydrogen atoms, and applying
    space group symmetry operations.

    Attributes
    ----------
    resolution : str
        The resolution of the generated system.
    material : str
        The name of the material being generated.
    N_C : int
        Number of Carbon atoms per monomer.
    N_H : int
        Number of Hydrogen atoms per monomer.
    N_m : int
        Number of monomers per chain.
    N_c : int
        Number of chains per unit cell.
    Nt : int
        Total number of atoms in the system.
    bond_offsets : list
        Bond offsets of atoms within and between unit cells.
    unit_types : list
        Atom types arrangement inside each unit cell chain.
    system_types : list
        Atomic symbol types for the entire system.

    Methods
    -------
    __init__(self, settings)
        Initializes the PECrystallineSystem with specific settings.

    _initialize_system_parameters(self)
        Initializes system parameters, such as forcefield types.

    _build_system(self)
        Generates the polymer system by defining atoms, bonds, angles, etc.

    _unit_cell(self)
        Defines the unit cell structure for the crystalline PE system.

    _space_group(self)
        Applies space group symmetry operations to construct the unit cell.

    __orthorhombic(self)
        Provides parameters for the orthorhombic unit cell of crystalline PE.

    __Pnam(self)
        Defines fractional coordinates for atoms in the orthorhombic unit cell.
    """
    def __init__(self, settings):
        """
        Initialize the PECrystallineSystem with the provided settings.

        Parameters
        ----------
        settings : dict
            Input settings for generating the crystalline PE system.
        """
        # Dynamically update instance attributes with settings values
        self.__dict__.update(settings)

        # Resolution of the generated material
        self.resolution = 'AA'
        # Add the material name
        self.material = 'PE'
        # PE monomer has 6 atoms: 2 Carbons and 4 Hydrogens
        # Number of Carbon atoms in each PE monomer
        self.N_C = 2
        # Number of Hydrogen atoms in each PE monomer
        self.N_H = 4
        # Number of monomer per chains
        self.N_m = 1
        # Number of chains per unit cell
        self.N_c = 2
        # Total number of atoms in the system
        self.Nt = self.Na * self.Nb * self.Nc * (self.N_C + self.N_H) * self.N_m * self.N_c
        # First C atom is connected to the third atom in each chain and consequent Hs
        # Bond offsets of atoms inside and between unit cells
        # The Carbon bond order in an iPP unit cell chain
        #   H  (2)   H  (5)
        #   |        |
        #   C2 (1) - C2 (4)
        #   |        |
        #   H  (3)   H  (6)
        # ID of next C2 of the next unit cell connected to the last C2 of the
        # previous unit cell is +3, and the ID of the hydrogens with respect to
        # the second C2 are -1 and -2
        self.bond_offsets = [3, -1, -2, 3, -1, -2]
        # Atom types arrangement inside each unit cell chain
        self.unit_types  = ['C2', 'H', 'H', 'C2', 'H', 'H']
        # Find the system atomic symbole types
        self.system_types = self.unit_types * self.N_c * self.Na * self.Nb * self.Nc

        # Add shared attributes from CrystallineAlkeneSystem
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
        self._ff_types['atoms'] = dict(C2=0, H=1)
        self._identify_unique_bond_types()
        # Dihedral angles
        self._ff_sizes['torsions'] = [numpy.pi]

    def _build_system(self):
        """
        Generate the polymer system with specified parameters.

        This method performs the following steps:

        - Creates the simulation box.
        - Adds carbon atoms.
        - Adds angles between bonded atoms.
        - Adds dihedral torsion angles.
        - Adds improper angles.

        Returns
        -------
        None
        """
        self._create_simulation_box()
        self._add_carbon_atoms()
        self._add_angles()
        self._add_dihedrals()
        self._add_impropers()

    def _unit_cell(self):
        """
        Defines the columns of the unit cell as fractional coordinates (a, b, c).

        Returns
        -------
        numpy.ndarray
            Orthorhombic unit cell dimensions.
        """
        return self.__orthorhombic()

    def __orthorhombic(self):
        """
        Parameters for orthorhombic unit cell of crystalline PE.
        Refer to 1975 - Avitabile - Low Temperature Crystal Structure of PE.

        Returns
        -------
        numpy.ndarray
            Array of unit cell dimensions.
        """
        # Orthorhombic unit cell of PE
        return numpy.array([[7.121, 0.0, 0.0],
                            [0.0, 4.851, 0.0],
                            [0.0,  0.0, 2.548]])

    def _space_group(self):
        """
        Adding chains in a unit cell using space group symmetry operations.

        Returns
        -------
        tuple of numpy.ndarray
            Fractional coordinates for Carbon and Hydrogen atoms.
        """
        return self.__Pnam()

    def __Pnam(self):
        """
        Fractional coordinates of Carbon and Hydrogen atoms in an orthorhombic
        unit cell of PE. Refer to:
        1998 - Bruno et. al. - Thermal expansion of polymers: Mechanisms in orthorhombic PE

        Returns
        -------
        tuple of numpy.ndarray
            Fractional coordinates for two chains in the unit cell.
        """
        # Internal coordinates for C atoms
        w1, w2 = 0.044183, 0.059859
        # Internal coordinates for H atoms
        w3, w4, w5, w6 = 0.186360, 0.011509, 0.027079, 0.277646
        # Pnam space group symmetry operators (Refer to 1939 - Bunn - The Crystal Structure)
        # First chain: x, -y, 0.25; -x, y, -0.25;
        C1 = numpy.array([[      w1,     -w2,  0.25],  # C2
                          [      w3,     -w4,  0.25],  # H
                          [      w5,     -w6,  0.25],  # H
                          [     -w1,      w2, -0.25],  # C2
                          [     -w3,      w4, -0.25],  # H
                          [     -w5,      w6, -0.25]]) # H
        # Second chain: 0.5-x, 0.5-y, -0.25; 0.5+x, 0.5+y, 0.25
        C2 = numpy.array([[  0.5-w1,  0.5-w2, -0.25],  # C2
                          [  0.5-w3,  0.5-w4, -0.25],  # H
                          [  0.5-w5,  0.5-w6, -0.25],  # H
                          [  0.5+w1,  0.5+w2,  0.25],  # C2
                          [  0.5+w3,  0.5+w4,  0.25],  # H
                          [  0.5+w5,  0.5+w6,  0.25]]) # H

        return C1, C2

