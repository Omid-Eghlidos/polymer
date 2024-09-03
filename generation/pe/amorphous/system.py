"""
system.py
=========

This module defines the `PEAmorphousSystem` class, which is used to create and
manage an amorphous polyethylene (PE) system for molecular simulations.

The `PEAmorphousSystem` class is a subclass of `PolymerSystem` and provides
specific implementations for constructing an amorphous PE system, including
adding backbone and methyl carbons, and adding various types of hydrogens
specific to the PE system.

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

Dependencies
------------
- generation.lib.polymer_system
"""


from generation.lib.polymer_system import PolymerSystem


class PEAmorphousSystem(PolymerSystem):
    """
    A class to represent an amorphous polyethylene (PE) system.

    This class inherits from PolymerSystem and provides specific
    implementations for adding carbons and hydrogens, which are unique to PE.

    Attributes
    ----------
    settings : Settings
        Configuration settings specific to the PE system.

    Methods
    -------
    __init__(self, Nc, Nm, random_variation=True)
        Initializes the PEAmorphousSystem with specific settings.

    _add_carbons(self)
        Adds backbone and methyl carbons specific to the PE system.

    _add_hydrogens(self)
        Adds hydrogen atoms specific to the PE system.

    _add_initiator(self, i, bond_vec)
        Adds a hydrogen atom as an initiator to the first carbon.

    _add_terminator(self, i, bond_vec)
        Adds a hydrogen atom as a terminator to the last carbon.
    """
    class Settings():
        """
        A nested class to hold the settings specific to the PE system.

        Attributes
        ----------
        Nc : int
            Number of chains per system.
        Nm : int
            Number of monomers per chain.
        random_variation : bool
            Whether to apply random variation to torsion angles.
        density : float
            Density of the amorphous PE system.
        monomer_atom_numbers : dict
            Number of atoms per monomer, categorized by element.
        forcefield_atom_types : dict
            Mapping of atom types to integer identifiers.
        angle_size : dict
            Bond angles in radians for the PE system.
        """
        def __init__(self, Nc, Nm, random_variation=True):
            self.Nc = Nc
            self.Nm = Nm
            self.random_variation = random_variation
            self.density = 0.865
            # PE monomer has 6 atoms: 2 Carbons and 4 Hydrogens
            self.monomer_atom_numbers = dict(C=2, H=4)
            # Each chain has an initiator and a terminator
            # Types of different forcefield atoms in the system
            self.forcefield_atom_types = dict(C2=0, C3=1, H=2)
            # C-C-C (C2C2C2) = 1.9548 rad = 112 deg
            # H-C-H (HC2H) = H-C-C (HC2C2) = 1.9024 rad = 109 deg
            self.angle_size = dict(CCC=1.9548, HCC=1.9024)

    def __init__(self, Nc, Nm, random_variation):
        """
        Initialize the PEAmorphousSystem with the provided settings.

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
        super().__init__(self.settings)

    def _add_carbons(self):
        """
        Add backbone and methyl carbons specific to the PE system.

        This method overrides the `__add_carbons` placeholder method in the
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
                    self._add_bond(atom_id, atom_id+1)
                else:
                    # Add the 1st gemini Carbon of the monomer
                    self._add_atom(atom_id, nc, 'C2', self._backbones[2*i])
                    self._add_bond(atom_id, atom_id+1)
                if i == self.Nm - 1:
                    # After adding a hydrogen as a terminator C2 becomes C3 and
                    self._add_atom(atom_id+1, nc, 'C3', self._backbones[2*i+1])
                else:
                    # Add the 2nd gemini Carbon of the monomer
                    self._add_atom(atom_id+1, nc, 'C2', self._backbones[2*i+1])
                    self._add_bond(atom_id+1, atom_id+2)
                atom_id += 2
        self._atom_id += atom_id
        self._construct_bond_table()

    def _add_hydrogens(self):
        """
        Add hydrogen atoms specific to the PE system.

        This method overrides the `__add_hydrogens` placeholder method in the
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

