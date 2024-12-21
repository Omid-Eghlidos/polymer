"""
system.py
=========

This module defines the `PEAmorphousSystem` class, which is used to create and
manage an amorphous polyethylene (PE) system for molecular simulations.

The `PEAmorphousSystem` class is a subclass of `AmorphousAlkeneSystem` and provides
specific implementations for constructing an amorphous PE system, including
adding backbone and methyl carbons, adding various types of hydrogens, and
building the system by integrating key structural elements.

Classes
-------
PEAmorphousSystem
    A subclass of `PolymerSystem` that provides specific implementations for
    constructing an amorphous PE system.

Methods
-------
__init__(self, Nc, Nm, random_variation=True)
    Initialize the settings and call the parent `PolymerSystem` constructor.

_add_carbons(self)
    Adds backbone and methyl carbons specific to the PE system.

_add_hydrogens(self)
    Adds hydrogen atoms specific to the PE system.

_add_initiator(self, i, bond_vec)
    Adds a hydrogen atom as an initiator to the first carbon.

_add_terminator(self, i, bond_vec)
    Adds a hydrogen atom as a terminator to the last carbon.

_build_system(self)
    Constructs the complete polymer system, integrating carbon and hydrogen atoms,
    angles, and dihedrals for simulation.

Dependencies
------------
- generation.lib.amorphous_alkene_system
- generation.lib.potential_coefficients
"""


import numpy
from generation.lib.amorphous_alkene_system import AmorphousAlkeneSystem
from generation.lib.potential_coefficients import potential_coefficients


class PEAmorphousSystem(AmorphousAlkeneSystem):
    """
    A class to represent an amorphous polyethylene (PE) system.

    This class inherits from PolymerSystem and provides specific
    implementations for adding carbons and hydrogens, which are unique to PE.

    Attributes
    ----------
    resolution : str
        The resolution of the generated system.
    material : str
        The name of the material that is being generated.
    rho : float
        Experimental density of the material.
    Nc : int
        Number of chains per system.
    Nm : int
        Number of monomers per chain.
    random_variation : bool
        Whether to apply random variation to torsion angles.
    density : float
        Density of the amorphous PE system.
    N_C : int
        Number of Carbon atoms per monomer.
    N_H : int
        Number of Hydrogen atoms per monomer.
    forcefield_atom_types : dict
        Mapping of atom types to integer identifiers.
    angle_size : dict
        Bond angles in radians for the PE system.

    Methods
    -------
    __init__(self, Nc, Nm, random_variation=True)
        Initializes the PEAmorphousSystem with specific settings.

    _add_carbon_atoms(self)
        Adds backbone and methyl carbons specific to the PE system.

    _add_hydrogen_atoms(self)
        Adds hydrogen atoms specific to the PE system.

    _add_initiator(self, i, bond_vec)
        Adds a hydrogen atom as an initiator to the first carbon.

    _add_terminator(self, i, bond_vec)
        Adds a hydrogen atom as a terminator to the last carbon.

    _build_system(self)
        Constructs the complete polymer system, integrating carbon and hydrogen atoms,
        angles, and dihedrals for simulation.
    """
    def __init__(self, settings):
        """
        Initialize the PEAmorphousSystem with the provided settings.

        Parameters:
        -----------
        settings : dict
            Input settings for generating the PE amorphous system.
        """
        # Add the settings variable as attributes
        self.__dict__.update(settings)

        # Resolution of the system
        self.resolution = 'AA'
        # Material name
        self.material = 'PE'
        # Density of the amorphous system (g/cm^3)
        self.rho = 0.85
        # PE monomer has 6 atoms: 2 Carbons and 4 Hydrogens
        # Number of Carbon atoms in each PE monomer
        self.N_C = 2
        # Number of Hydrogen atoms in each PE monomer
        self.N_H = 4
        # Total number of atoms in the system + an initiator and a terminator for each chain
        self.Nt = self.Nc * self.Nm * (self.N_C + self.N_H) + 2 * self.Nc
        # Random variation to be added to dihedral angles for generation
        self.random_variation = True
        # Randomly flip the dihedral angles
        self.random_flip = False

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
        self._ff_types['atoms']   = dict(C2=0, C3=1, H=2)
        self._identify_unique_bond_types()
        # Bond-angles sizes in the amorphous structure in rad
        # C-C-C (C2C2C2) = 1.9548 rad = 112 deg
        # H-C-H (HC2H) = H-C-C (HC2C2) = C-C-H (C2C2H) = 1.9024 rad = 109 deg
        self._ff_sizes['angles'] = dict(CCC=1.9548, HCC=1.9024)
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
        Add backbone and methyl carbons specific to the PE system.

        This method overrides the `__add_carbon_atoms` placeholder method in the
        PolymerSystem base class.
        """
        atom_id = 0
        # Each monomer has 2 backbones and a methyl C with its C id being 3N in chains
        for nc in range(self.Nc):
            self._add_backbone_carbons()
            for  i in range(self.Nm):
                if i == 0:
                    # After adding a hydrogen as a terminator C2 becomes C3
                    self._add_atom(atom_id, nc, 'C3', self._backbones[2*i])
                else:
                    # Add the 1st gemini Carbon of the monomer
                    self._add_atom(atom_id, nc, 'C2', self._backbones[2*i])
                    self._add_bond(atom_id-1, atom_id)
                atom_id += 1
                if i == self.Nm - 1:
                    # After adding a hydrogen as a terminator C2 becomes C3 and
                    self._add_atom(atom_id, nc, 'C3', self._backbones[2*i+1])
                    self._add_bond(atom_id-1, atom_id)
                else:
                    # Add the 2nd gemini Carbon of the monomer
                    self._add_atom(atom_id, nc, 'C2', self._backbones[2*i+1])
                    self._add_bond(atom_id-1, atom_id)
                atom_id += 1
        self._atom_id += atom_id
        self._construct_bond_table()

    def _add_hydrogen_atoms(self):
        """
        Add hydrogen atoms specific to the PE system.

        This method overrides the `__add_hydrogen_atoms` placeholder method in the
        PolymerSystem base class.
        """
        nc = 1
        for i, bonded in enumerate(self.bond_table):
            if not bonded:
                continue
            # Bond vectors from atom i to atom in each bond
            bond_vec = [self._bond_vector(i, b) for b in bonded]
            if i % (self.Nm*2) == 0:
                self._add_initiator(i, bond_vec)
            elif i == nc*self.Nm*2 - 1:
                self._add_terminator(i, bond_vec)
                nc += 1
            elif len(bonded) == 2:
                self._add_gemini_hydrogens(i, bond_vec)
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
        self._add_methyl_hydrogens(i, bond_vec)

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

