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
from scipy.constants import N_A
from sympy import symbols, Eq, solve
from math import cos, sin, pi, radians
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from generation.lib.polymer_system import PolymerSystem
from generation.lib.potential_coefficients import potential_coefficients


class PUAmorphousSystem(PolymerSystem):
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
    def __init__(self, settings):
        """
        Initialize the PPAmorphousSystem with the provided settings.

        Parameters:
        -----------
        settings : dict
            Input settings for generating the PE amorphous system.
        """
        # Add shared attributes from PolymerSystem
        super().__init__()

        # Add the settings variable as attributes
        self.__dict__.update(settings)

        # Resolution of the generated system
        self.resolution = 'AA'
        # Material name
        self.material = 'PU'
        # Density of the amorphous system (g/cm^3) (Amini et al., 2010)
        self.rho = 1.1
        # Total number of atoms in the system + an initiator and a terminator for each chain
        self.Nt = 0

        # Monomer SMILES and structure details
        self._monomer = {}
        # Atom ID counter
        self._atom_id = 0

        # Initialize constant parameters and complete PU monomer
        self._initialize_system_parameters()

        # Construct the system with the given size and store its parameters
        self._build_system()

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
        self._add_chains()
        self._map_to_PBC()
        self._add_angles()
        self._add_dihedrals()
        self._add_impropers()

    def _initialize_system_parameters(self):
        """
        Initialize system's constant parameters including atomic weight and bond lengths.
        """
        # Types of different forcefield atoms in the system
        self._ff_types['atoms'] = { 'c4': 0,  'c4o': 1, "c3'": 2,  'c3"': 3,
                                   'c3a': 4,  'n3m': 5, 'o1=': 6,  'o2e': 7,
                                   'o2s': 8,   'h1': 9, 'h1n': 10, 'h1o': 11}
        # Charges of the atoms inside the molecule
        self._ff_sizes['charges'] = {    'c4': -0.10600,    'c4o':  0.35080,
                                      'c4o2s':  0.11900,  'c4o2e':  0.02700,
                                        "c3'":  0.72000,    'c3"':  0.80700,
                                       'c3ah': -0.12680,   'c3an':  0.23700,
                                       'c3ac':  0.00000,    'n3m': -0.72600,
                                     "o1=c3'": -0.53100, 'o1=c3"': -0.58500,
                                     'o1=c4o': -0.39640,    'o2e': -0.55710,
                                        'o2s': -0.39600,    'h1a':  0.12680,
                                        'h1c':  0.05300,    'h1n':  0.37800,
                                        'h1o':  0.42410}
        # SMILES for the hard segment (hs) for
        hs = 'O=Cc4ccc(NC(=O)Nc3ccc(Cc2ccc(NC(=O)Nc1ccc(C(=O)O{ss})cc1)cc2)cc3)cc4'
        # SMILES for the soft segment (ss)
        ss = 'CCCCO'
        # Construct total PU variant SMILES by concatenating hard and soft segments
        pu = hs.format(ss=ss*self.Nss)
        # Identify molecular wights of atoms existing in the system
        self.__identify_atoms_molecular_weights(pu)
        # Identify bond lengths existing in the system
        self.__identify_bond_types(pu)
        # Create the PU complete monomer
        self.__create_monomer(pu)
        # Total number of atoms in the system
        self.Nt = self.Nc * self.Nm * self._monomer['PU'].GetConformer().GetNumAtoms()

    def __identify_atoms_molecular_weights(self, pu):
        # Create a sample monomer with one hard and one soft segment
        PU = Chem.AddHs(Chem.MolFromSmiles(pu))
        # Iterate over atoms and print atomic weights
        masses = {}
        for atom in PU.GetAtoms():
            if atom.GetSymbol() not in masses:
                masses[atom.GetSymbol()] = atom.GetMass()
        self._ff_sizes['masses'] = {s: masses[s] for s in sorted(masses.keys())}

    def __identify_bond_types(self, pu):
        # Create a sample monomer with one hard and one soft segment
        PU = Chem.AddHs(Chem.MolFromSmiles(pu))
        # Generate 3D coordinates
        status = AllChem.EmbedMolecule(PU, AllChem.ETKDG())
        # Optimize geometry using Merck Molecular Force Field (MMFF)
        result = AllChem.MMFFOptimizeMolecule(PU)

        # Get the 3D coordinates
        for bond in PU.GetBonds():
            i, j = bond.GetBeginAtom(), bond.GetEndAtom()
            ri = PU.GetConformer().GetAtomPosition(i.GetIdx())
            rj = PU.GetConformer().GetAtomPosition(j.GetIdx())
            lij, qij = ri.Distance(rj), ri.AngleTo(rj)
            type_i, type_j = self._get_atom_type(i), self._get_atom_type(j)
            types = self.__get_type_equivalent(sorted([type_i, type_j]))
            bond_type = '-'
            bond_type = bond_type.join(types)
            if bond_type not in self._ff_types['bonds']:
                self._ff_types['bonds'][bond_type] = len(self._ff_types['bonds'])
                #print(f'{bond_type:^7}: {lij:^6.2f} A')

    def __create_monomer(self, pu):
        """
        Create a single PU monomer using its SMILES formula
        """
        # Construct the polyurea from PU SMILES
        PU = Chem.AddHs(Chem.MolFromSmiles(pu))
        # Save a picture of the molecule into a file
        try:
            Draw.MolToFile(PU, 'pu.ps', size=(2400, 360), fitImage=True)
        except ValueError:
            pass
        # Generate 3D coordinates
        status = AllChem.EmbedMolecule(PU, useRandomCoords=True)
        # Optimize geometry using Merck Molecular Force Field (MMFF)
        result = AllChem.MMFFOptimizeMolecule(PU)
        self._monomer = dict(SMILES=pu, PU=PU)

    def _create_simulation_box(self):
        """
        Determine the simulation box dimensions based on the system mass and density.

        Returns
        -------
        None
        """
        # Mass of one monomer minus the initiator and terminator H-atom
        monomer_mass = Descriptors.MolWt(self._monomer['PU']) - 2 * self._ff_sizes['masses']['H']
        # Mass of one chain of the system for Nm monomers plus the initiator and terminator
        chain_mass = self.Nm * monomer_mass + 2 * self._ff_sizes['masses']['H']
        # Total mass of the system of Nc chains (g/mol)
        mass = self.Nc * chain_mass
        # Volume of the system for the given density in Angstrom^3
        # To convert from g/cm^3 to g/A^3 should divide by 10^24
        volume = (mass/N_A) / (self.rho/1e24)
        # Assuming cubic box for the amorphous system, dimension = cubic root of Volume
        for k in range(3):
            self.box[k,k] = volume**(1.0/3.0)

    def _add_chains(self):
        """
        Add Nc chains into the simulation box to create the system.
        """
        count = dict(C=0, N=0, O=0, H=0)
        for nc in range(self.Nc):
            # Create a chain with the given Nm monomers
            chain = self.__build_chain()
            coords = chain.GetConformer().GetPositions()
            atom_id = self._atom_id
            for j, atom in enumerate(chain.GetAtoms()):
                count[atom.GetSymbol()] += 1
                atom_type = self._get_atom_type(atom)
                self._add_atom(atom_id, nc, atom_type, coords[j])
                atom_id += 1
            for bond in chain.GetBonds():
                i = self._atom_id + bond.GetBeginAtomIdx()
                j = self._atom_id + bond.GetEndAtomIdx()
                self._add_bond(i, j, bond.GetBondType().name)
            self._atom_id = atom_id
        self._construct_bond_table()
        if self.atoms_format == 'full':
            self._get_composite_atom_charges()

    def _get_atom_type(self, atom):
        """
        Determine the forcefield type for the given atom.
        """
        symbol = atom.GetSymbol()
        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
        hybridization = atom.GetHybridization().name
        aromatic = atom.GetIsAromatic()
        if symbol == 'N':
            # Nitrogen in carbonyl
            return 'n3m'
        elif symbol == 'O':
            # Oxygen in carbonyl
            if len(neighbors) == 1:
                return 'o1='
            # Oxygen in esthers
            if hybridization == 'SP2':
                return 'o2s'
            # Oxygen in ethers
            return 'o2e'
        elif symbol == 'C':
            if hybridization == 'SP2':
                if aromatic:
                    return 'c3a'
                # Carbon in carbonyl with 2 polar subset (nitrogen and oxygen)
                if neighbors.count('N') == 2 and neighbors.count('O') == 1:
                    return 'c3"'
                # Crabon in carbonyl with 1 polar subset (only oxygen)
                return "c3'"
            elif hybridization == 'SP3':
                # Carbon bonded to oxygen
                if 'O' in neighbors:
                    return 'c4o'
                # Generic carbon bonded to carbon and hydrogen
                return 'c4'
        elif symbol == 'H':
            # Hydrogen bonded to oxygen
            if 'O' in neighbors:
                return 'h1o'
            # Hydrogen bonded to nitrogen
            if 'N' in neighbors:
                return 'h1n'
            # Hydrogen bonded to carbon
            return 'h1'

    def _get_composite_atom_charges(self):
        """
        Return charges for atoms with a composite type in combination with atoms
        that are bonded to.
        """
        for i, q in enumerate(self.atom_charges):
            if q == 0:
                atom_type = [a for a, t in self._ff_types['atoms'].items()
                                                 if t == self.atom_types[i]][0]
                bonded_types = [[a for a, t in self._ff_types['atoms'].items()
                                 if t == self.atom_types[b]][0] for b in self.bond_table[i]]
                if atom_type == 'c3a':
                    if 'n3m' in bonded_types:
                        atom_type += 'n'
                    elif 'h1' in bonded_types:
                        atom_type += 'h'
                    else:
                        atom_type += 'c'
                elif atom_type == 'c4':
                    atom_type += [b for b in bonded_types if b in ['o2e', 'o2s']][0]
                elif atom_type == 'h1':
                    atom_type += bonded_types[0][0]
                elif atom_type == 'o1=':
                    atom_type += bonded_types[0]

                self.atom_charges[i] = self._ff_sizes['charges'][atom_type]

    def _get_bond_type(self, i, j, covalent):
        """
        Return the bond type between atoms i and j.

        Parameters
        ----------
        i : int
            Index of the first atom.
        j : int
            Index of the second atom.
        covalent: str
            Type of the covalent bond
        """
        ai = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[i]][0]
        aj = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[j]][0]
        types = self.__get_type_equivalent(sorted([ai, aj]))
        bond_type = '-'
        bond_type = bond_type.join(types)
        return self._ff_types['bonds'][bond_type]

    def _get_angle_type(self, ijk):
        """
        Return the angle type between atoms i, j, and k.

        Parameters
        ----------
        ijk : list
            List of three atom indices forming an angle.
        """
        i, j, k = ijk[0], ijk[1], ijk[2]
        ti = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[i]][0]
        tj = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[j]][0]
        tk = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[k]][0]
        types = self.__get_type_equivalent([ti, tj, tk])
        if types[0] > types[2]:
            types = types[::-1]
            ijk = ijk[::-1]
        if types not in self.angle_types:
            self.angle_types.append(types)
        return (self.angle_types.index(types), ijk)

    def _get_dihedral_type(self, ijkl):
        """
        Return the torsion angle type between atoms i, j, k, and l.

        Parameters
        ----------
        ijkl : list
            List of three atom indices forming a torsion angle.
        """
        i, j, k, l = ijkl[0], ijkl[1], ijkl[2], ijkl[3]
        ti = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[i]][0]
        tj = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[j]][0]
        tk = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[k]][0]
        tl = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[l]][0]
        types = self.__get_type_equivalent([ti, tj, tk, tl])
        # Prevent double count
        # ABCD <=> DCBA
        if types[0] > types[-1]:
            types = types[::-1]
            ijkl = ijkl[::-1]
        if types not in self.dihedral_types:
            self.dihedral_types.append(types)
        return (self.dihedral_types.index(types), ijkl)

    def _get_improper_type(self, ijkl):
        """
        Return the improper angle type between atoms i, j, k, and l.

        Parameters
        ----------
        ijkl : list
            List of three atom indices forming an improper angle.
        """
        i, j, k, l = ijkl[0], ijkl[1], ijkl[2], ijkl[3]
        ti = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[i]][0]
        tj = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[j]][0]
        tk = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[k]][0]
        tl = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[l]][0]
        types = self.__get_type_equivalent([ti, tj, tk, tl])
        # Prevent double count
        # ABCD <=> DCBA
        if types[0] > types[-1]:
            types = types[::-1]
            ijkl = ijkl[::-1]
        if types not in self.improper_types:
            self.improper_types.append(types)
        return (self.improper_types.index(types), ijkl)

    def __get_type_equivalent(self, types):
        """
        Return the type equivalence for the given atom type when involved in
        forming a bond, angle, torsion, or improper.
        """
        if self.potential_coeffs.upper() == 'PCFF':
            return types

        # Atom type equivalences in non-bonded and bonded interactions for different potentials
        bond_equivalences = { 'c4':  'c4', 'c4o':  'c4', "c3'": "c3'", 'c3"': "c3'",
                             'c3a': 'c3a', 'n3m': 'n3m', 'o1=': 'o1=', 'o2e': 'o2e',
                             'o2s': 'o2e',  'h1':  'h1', 'h1n': 'h1',  'h1o': 'h1'}
        equivalences = { 'c4': 0, 'c4o': 0, "c3'": 2, 'c3"': 2,
                        'c3a': 4, 'n3m': 5, 'o1=': 6, 'o2e': 7,
                        'o2s': 7,  'h1': 9, 'h1n': 9, 'h1o': 9}

        for k in range(len(types)):
            if len(types) == 2:
                types[k] = bond_equivalences[types[k]]
                if types == ["c3'", 'h1']:
                    types = ['c4', 'h1']
            else:
                types[k] = equivalences[types[k]]
        return types

    def __build_chain(self):
        """
        Create a random chain with the given number of monomers.
        """
        # Initialize the polymer chain with the first monomer
        chain   = Chem.RWMol(Chem.MolFromSmiles(self._monomer['SMILES']))
        # Add subsequent monomers to the polymer chain
        for nm in range(self.Nm-1):
            chain.InsertMol(Chem.MolFromSmiles(self._monomer['SMILES']))
        # Add hydrogens to the full chain to complete valence
        chain = Chem.AddHs(chain)
        # Generate 3D coordinates
        status = AllChem.EmbedMolecule(chain, useRandomCoords=True)
        assert status == 0, ValueError('Unable to generate 3D coordinates.')
        # Sanitize the molecule to fix aromaticity issues
        Chem.SanitizeMol(chain)
        # Optimize geometry using Merck Molecular Force Field (MMFF)
        result = AllChem.MMFFOptimizeMolecule(chain)
        return chain

