#!/usr/bin/env python3
"""
system.py
=========

This module defines the `BeadSpringSystem` class, which is used to create and
manage a coarse-grained (CG) polymer system using a bead-spring model for molecular
simulations. It supports generating the CG system from SMILES strings, adding atoms,
defining bonds, angles, dihedrals, and impropers, and writing the system to a data file.

Classes
-------
BeadSpringSystem
    A class for creating and managing a coarse-grained polymer system.

Methods
-------
__init__(self, settings)
    Initializes the system with the given settings.

__parse_SMILES(self)
    Parses the SMILES string to extract the backbone and branches of the polymer.

_initialize_structural_parameters(self, settings)
    Initializes structural parameters for beads, bonds, and angles.

_build_system(self)
    Constructs the complete bead-spring system by adding atoms, bonds, angles,
dihedrals, and impropers.

_create_simulation_box(self)
    Creates an orthogonal simulation box based on the system density and mass.

_add_atom(self)
    Adds backbone and branch beads to the system.

__add_backbones(self, nc)
    Determines positions of backbone beads for a given chain.

__add_branches(self, nc)
    Adds branch beads connected to the backbone beads.

__compute_CDF(self, path)
    Computes the cumulative distribution function (CDF) from a probability
distribution function (PDF).

__random_distribution_value(self, CDF)
    Samples a random value from the cumulative distribution function (CDF).

__random_bond_vector(self, i, j)
    Returns a random bond vector for a bond between beads of specific types.

__add_bond(self, i, j)
    Adds a bond between two specified beads.

__add_angles(self)
    Defines angles between connected atoms in the bond table.

__add_dihedrals(self)
    Defines dihedral torsion angles in the system.

__add_impropers(self)
    Identifies and adds improper torsion angles in the system.

Dependencies
------------
- numpy
- scipy.constants
- sympy
- math
"""

import numpy
from numpy import linalg, random, cross, dot
from scipy.constants import N_A
from sympy import symbols, Eq, solve
from math import cos, pi, radians


class BeadSpringSystem():
    """
    A class to represent a coarse-grained (CG) polymer system using a bead-spring model.

    This class supports generating the CG system from SMILES strings, defining
    structural parameters, and adding atoms, bonds, angles, dihedrals, and impropers.

    Attributes
    ----------
    box_length : float
        Edge length of the simulation box.
    beads : list
        Each element stores [molecule_id, bead_type, [x, y, z]].
    bead_masses : list
        Masses of individual beads (float).
    bonds : list
        Each element stores [bond_type, bead1_index, bead2_index].
    angles : list
        Each element stores [angle_type, bead1_index, bead2_index, bead3_index].
    dihedrals : list
        Each element stores [dihedral_type, bead1_index, bead2_index, bead3_index, bead4_index].
    impropers : list
        Each element stores [improper_type, bead1_index, bead2_index, bead3_index, bead4_index].

    Methods
    -------
    __init__(self, settings)
        Initializes the system with the given settings.

    __parse_SMILES(self)
        Parses the SMILES string to extract backbone and branches.

    _initialize_structural_parameters(self, settings)
        Initializes forcefield parameters for beads, bonds, angles, and their distributions.

    _build_system(self)
        Constructs the bead-spring system by adding atoms, bonds, angles, dihedrals,
        and impropers.

    _create_simulation_box(self)
        Creates an orthogonal simulation box based on the system density and mass.

    _add_atom(self)
        Adds atoms to the system, including backbone and branch beads.

    __add_backbones(self, nc)
        Determines the positions of backbone beads for a given chain index.

    __add_branches(self, nc)
        Adds branch beads and connects them to backbone beads.

    __compute_CDF(self, path)
        Computes the cumulative distribution function (CDF) from a probability
        distribution function (PDF).

    __random_distribution_value(self, CDF)
        Samples a random value from the cumulative distribution function (CDF).

    __random_bond_vector(self, i, j)
        Generates a random bond vector for a bond between two beads.

    __add_bond(self, i, j)
        Adds a bond between two beads, ensuring type consistency.

    __add_angles(self)
        Defines angles between connected beads in the bond table.

    __add_dihedrals(self)
        Defines dihedral torsion angles in the system.

    __add_impropers(self)
        Identifies and adds improper torsion angles.
    """
    def __init__(self, settings):
        """
        Initializes the BeadSpringSystem with the given settings.

        Parameters
        ----------
        settings : dict
            A dictionary containing input settings for generating the CG system.
        """
        # Dynamically update instance attributes with settings values
        self.__dict__.update(settings)

        self.resolution       = 'CG'
        self.Nt               = len(self.beads) * self.Nm * self.Nc
        self.mass             = 0
        self.volume           = 0
        self.box              = numpy.zeros((3,3))
        self.atoms            = []
        self.atom_types       = {}
        self.bonds            = []
        self.bond_types       = {}
        self.angles           = []
        self.angle_types      = {}
        self.dihedrals        = []
        self.dihedral_types   = {}
        self.impropers        = []
        self.improper_types   = {}

        self._bond_table       = [[] for _ in range(self.Nt)]
        self._forcefield_types = dict(beads={}, bonds={}, angles={})

        self.__monomer        = {}
        self.__chain          = ''
        self.__system         = ''
        self.__bead_id        = [[] for _ in range(self.Nc)]
        self.__backbones      = [[] for _ in range(self.Nc)]
        self.__branches       = [[] for _ in range(self.Nc)]
        self.__bond_length    = {}
        self.__angle_size     = {}

        self.__parse_SMILES()
        self.__initialize_structural_parameters(settings)
        self._build_system()

    def __initialize_structural_parameters(self, settings):
        """
        Compute and store the cumulative distribution function (CDF) from probability
        distribution function for each of the structural distributions to be drawn from
        if "random" was specified for it, otherwise set their specified fix values.
        """
        self._forcefield_types['beads'] = {t: b for t, b in settings['beads'].items()}
        for b in settings['bonds']:
            self._forcefield_types['bonds'][b] = settings['bonds'][b][0]
            try:
                self.__bond_length[b] = float(settings['bonds'][b][1])
            except:
                self.__bond_length[b] = self.__compute_CDF(settings['bonds'][b][1])
        if settings['angles']:
            for a in settings['angles']:
                self._forcefield_types['angles'][a] = settings['angles'][a][0]
                try:
                    self.__angle_size[a] = float(settings['angles'][a][1])
                except:
                    self.__angle_size[a] = self.__compute_CDF(settings['angles'][a][1])

    def __parse_SMILES(self):
        """
        Parses the SMILES string to separate backbone and branches.

        Returns:
        --------
        backbone : str
            Backbone beads from the SMILES notation.
        branches : list
            List of branches and the bead they are connected to.

        Example:
        --------
        A(C)B -> Backbone: 'AB', Branches: [('A', 'C')]
        """
        # Input SMILES notation of the polymer system.
        smiles = self.monomer
        backbone = ""
        branches = []
        i = 0

        while i < len(smiles):
            if smiles[i] == '(':
                # Handle the branch
                start = i + 1
                end = smiles.find(')', start)
                branch = smiles[start:end]
                # The bead before '(' is the backbone connected to the branch
                backbone_bead = backbone[-1] if backbone else ""
                branches.append((backbone_bead, branch))
                i = end + 1
            else:
                backbone += smiles[i]
                i += 1
        self.__monomer = dict(backbone=backbone, branch={v: k for k, v in branches})
        self.__chain = smiles.replace('(', '').replace(')', '')*self.Nm
        self.__system = self.__chain * self.Nc

    def _build_system(self):
        """ Creates a bead spring system. """
        self._create_simulation_box()
        self._add_atom()
        self._construct_bond_table()
        self.__add_angles()
        self.__add_dihedrals()
        self.__add_impropers()

    def _create_simulation_box(self):
        """ Create an orthogonal simulation box for the specified sizes for the system. """
        # Mass of each chain of the system
        chain_mass = sum([self.Nm*bead[1] for bead in self._forcefield_types['beads'].values()])
        # Total mass of the system
        self.mass = self.Nc * chain_mass
        # Volume of the system for the given density in Angstrom^3
        # To convert from g/cm^3 to g/A^3 should divide by 10^24
        self.volume = (self.mass/N_A) / (self.rho/1e24)
        # Orthogonal box dimension
        for k in range(3):
            self.box[k,k] = self.volume**(1.0/3.0)

    def _add_atom(self):
        """ First go over backbone beads and add them, then add the branch beads. """
        for nc in range(self.Nc):
            self.__add_backbones(nc)
            self.__add_branches(nc)
            backbone_id, branch_id = 0, 0
            for i, b in enumerate(self.__chain):
                if b in self.__monomer['backbone']:
                    coords = self.__backbones[nc][backbone_id]
                    backbone_id += 1
                elif b in self.__monomer['branch']:
                    coords = self.__branches[nc][branch_id]
                    branch_id += 1
                atom = self._forcefield_types['beads'][b]
                if atom[0] not in self.atom_types:
                    self.atom_types[atom[0]-1] = [b, atom[1]]
                self.atoms.append([nc, atom[0]-1, coords])

    def __add_backbones(self, nc):
        """ Traverse over all the backbone beads and determine their position randomly. """
        # To generate coordinates of a chain in a specific configuration, coorinates of
        # at least the first 2 backbones are required.
        # For 1st backbone generate a random starting coordinates in the simulation box
        coords = [numpy.array([random.uniform(0, self.box[0,0]),
                               random.uniform(0, self.box[1,1]),
                               random.uniform(0, self.box[2,2])])]
        bead_id = [0]
        for i in range(1, len(self.__chain)):
            if self.__chain[i] not in self.__monomer['backbone']:
                continue
            bead_id.append(i)
            if len(bead_id) == 2:
                # Assume 2nd backbone is randomly located from the 1st one at a bond length
                coords.append(coords[-1] + self.__random_bond_vector(bead_id[-2], bead_id[-1]))
            else:
                # Find bead k from previous beads of i and j
                r_i, r_j = coords[-2], coords[-1]
                r_ji = r_i - r_j
                while True:
                    r_k = r_j + self.__random_bond_vector(bead_id[-2], bead_id[-1])
                    r_jk = r_k - r_j
                    cos_ijk = dot(self._normalized(r_ji), self._normalized(r_jk))
                    if self.__angle_size:
                        theta = self.__get_angle_size(bead_id[-3], bead_id[-2], bead_id[-1])
                        # Prevent linear chains cuased by angles in vicinity of 0 or 180 by 10
                        max_cos = cos(radians(theta + 10))
                        min_cos = cos(radians(theta - 10))
                        if cos_ijk > max_cos and cos_ijk < min_cos:
                            break
                    elif abs(cos_ijk) < 0.8:
                        break
                coords.append(r_k)
            N = nc * len(self.__chain)
            self.__add_bond(N+bead_id[-1], N+bead_id[-2])

        assert len(bead_id) == len(coords)
        self.__backbones[nc].extend(coords)
        self.__bead_id[nc].extend(bead_id)

    def __add_branches(self, nc):
        """ Add the branch beads to the corresponding backbone beads. """
        coords  = []
        bead_id = []
        for i in range(len(self.__chain)):
            if self.__chain[i] in self.__monomer['backbone']:
                continue
            bead_id.append(i)
            # Branch's backbone is right before the branch in the chain pattern
            bi = i - 1
            # Index of the backbone branch backbone among the backbones
            idx = self.__bead_id[nc].index(bi)
            if len(bead_id) == 1 or i == len(self.__chain) - 1:
                assert self.__chain[i-1] in self.__monomer['backbone']
                # Assume the first and last branches are randomly bonded to its backbone
                coords.append(self.__backbones[nc][idx] + self.__random_bond_vector(bi, i))
            else:
                # Index of beads j and k in the complete chain
                bj  = self.__bead_id[nc][idx-1]
                bk  = self.__bead_id[nc][idx+1]
                # Bond vector beween the bead behind the branch's backbone
                r_ij = self.__backbones[nc][idx-1] - self.__backbones[nc][idx]
                # Bond vector between the bead after the branch's backbone
                r_ik = self.__backbones[nc][idx+1] - self.__backbones[nc][idx]
                # Bond length between the backbones and the branch
                l = self.__get_bond_length(bi, i)
                if self.__angle_size:
                    # Connect the branch to its backbone at an specific angle theta
                    xyz = []
                    while not xyz:
                        theta = self.__get_angle_size(i, bi, bj)
                        x, y, z = symbols('x, y, z', real=True)
                        eq1 = Eq((r_ij[0]*x + r_ij[1]*y + r_ij[2]*z), l**2 * cos(pi - theta))
                        eq2 = Eq((r_ik[0]*x + r_ik[1]*y + r_ik[2]*z), l**2 * cos(pi - theta))
                        eq3 = Eq((x*x + y*y + z*z), l**2)
                        xyz = solve((eq1, eq2, eq3), (x, y, z))
                    R = numpy.array(xyz[0], dtype=numpy.float64)
                else:
                    # Connect the branch to its backbone perpendicularly
                    R = self._normalized(cross(r_ij, r_ik)) * l
                assert round(linalg.norm(R), 3) == round(l, 3)
                coords.append(self.__backbones[nc][idx] + R)
            N = nc * len(self.__chain)
            self.__add_bond(N+bi, N+i)

        assert len(bead_id) == len(coords)
        self.__branches[nc].extend(coords)
        self.__bead_id[nc].extend(bead_id)

    def __compute_CDF(self, path):
        """
        Compute cumulative distribution function (CDF) from the probability
        distribution function (PDF) of the given path.
        """
        # Open the average PDF file
        with open(path) as fid:
            PDF = numpy.genfromtxt(fid)
            # normalization factor
            nf = sum(PDF[:,1])
            # Compute the Cumulative Distribution Function (CDF)
            CDF = numpy.zeros((len(PDF), 2))
            for i in range(1, len(PDF)):
                CDF[i,0] = PDF[i,0]
                CDF[i,1] = PDF[i-1,1] + PDF[i,1]/nf
        return CDF

    def __random_distribution_value(self, CDF):
        """ Find a random value for the bdf/adf/tdf  from their given distribution\
            by generating a random number in [0.0 1.0] and then find the\
            corresponding value of the cumulative distribution function """
        # Find the corresponding value of the distribution for the random number
        rand = 0
        while rand < 0.01:
            rand = random.random()
        for i in range(len(CDF)):
            if abs(CDF[i,1] - rand) < 0.01 and CDF[i,1] != 0.0:
                return CDF[i,0]
        return self.__random_distribution_value(CDF)

    def __random_bond_vector(self, i, j):
        """ Returns a random bond vector between two beads of a specific type. """
        # The shortest bond length between 2 beads within + and - half box distance
        return self._normalized(random.rand(3) - 0.5) * self.__get_bond_length(i, j)

    def __add_bond(self, i, j):
        """ Add bonds between the two beads i and j. """
        bond_type = self.__get_bond_type(i, j)
        self.bonds.append((self.bond_types[bond_type]-1, sorted([i, j])))

    def __get_bond_type(self, i, j):
        """ Return a bond type between two beads. """
        type_i = self.__system[i]
        type_j = self.__system[j]
        bond_type = ''.join(sorted(type_i + type_j))
        # Ensure the found type is an specified type
        assert bond_type in self._forcefield_types['bonds'], f'Undefined type found: {bond_type}'
        if bond_type not in self.bond_types:
            self.bond_types[bond_type] = self._forcefield_types['bonds'][bond_type]
        return bond_type

    def __get_bond_length(self, i, j):
        """ Return the bond length for the specified bond type. """
        bond_type = self.__get_bond_type(i, j)
        try:
            return float(self.__bond_length[bond_type])
        except:
            return self.__random_distribution_value(self.__bond_length[bond_type])

    def _construct_bond_table(self):
        """
        Create a table where each row contains atoms bonded to an atom with ID
        as the row number.
        """
        for b in self.bonds:
            i, j = b[1]
            self._bond_table[i].append(j)
            self._bond_table[j].append(i)
        for bond in self._bond_table:
            bond.sort()

    def __add_angles(self):
        """
        Traverses the bond list and defines all bond angles.
        """
        for j in range(len(self._bond_table)):
            # Set J contains all atoms bonded to atom j.
            bonded_j = self._bond_table[j]
            for i in bonded_j:
                for k in bonded_j:
                    # Don't add angle twice.
                    if i < k:
                        self.__add_angle(i, j, k)

    def __add_angle(self, i, j, k):
        """
        Function for adding angles to the system.
        """
        angle_type = self.__get_angle_type(i, j, k)
        self.angles.append((self.angle_types[angle_type]-1, [i, j, k]))

    def __get_angle_type(self, i, j, k):
        """ Return the angle type between i, j, k """
        type_i = self.__system[i]
        type_j = self.__system[j]
        type_k = self.__system[k]
        angle_type = type_i + type_j + type_k
        if angle_type[0] > angle_type[-1]:
            angle_type = angle_type[::-1]
        # If the angle types are specified in the input, the found type should be one
        if self._forcefield_types['angles']:
            assert angle_type in self._forcefield_types['angles'],\
                                        f'Undefined type found: {angle_type}'
            if angle_type not in self.angle_types:
                self.angle_types[angle_type] = self._forcefield_types['angles'][angle_type]
            return angle_type
        if angle_type not in self.angle_types:
            self.angle_types[angle_type] = len(self.angle_types) + 1
        return angle_type

    def __get_angle_size(self, i, j, k):
        """ Return the size of the angle formed between 3 consecutive beads. """
        angle_type = self.__get_angle_type(i, j, k)
        try:
            return float(self.__angle_size[angle_type])
        except:
            return self.__random_distribution_value(self.__angle_size[angle_type])

    def __add_dihedrals(self):
        """
        Traverses the bond list and defines all the dihedral torsion angles.
        """
        for b in self.bonds:
            j, k = b[1]
            bonded_j = [i for i in self._bond_table[j] if i != k]
            bonded_k = [i for i in self._bond_table[k] if i != j]
            for i in bonded_j:
                for l in bonded_k:
                    self.__add_dihedral(i, j, k, l)

    def __add_dihedral(self, i, j, k, l):
        """
        Function for adding dihedral torsion angles to the system.
        """
        dihedral_type = self.__get_dihedral_type(i, j, k, l)
        self.dihedrals.append((self.dihedral_types[dihedral_type]-1, [i, j, k, l]))

    def __get_dihedral_type(self, i, j, k, l):
        """ Return the dihedral type between i, j, k, l """
        type_i = self.__system[i]
        type_j = self.__system[j]
        type_k = self.__system[k]
        type_l = self.__system[l]
        dihedral_type = type_i + type_j + type_k + type_l
        if dihedral_type[0] > dihedral_type[-1]:
            dihedral_type = dihedral_type[::-1]
        if dihedral_type not in self.dihedral_types:
            self.dihedral_types[dihedral_type] = len(self.dihedral_types) + 1
        return dihedral_type

    def __add_impropers(self):
        """
        Traverse the bond list and find improper angles and their types.

        This function add the found improper angle and add them to ``impropers``.
        """
        for j, bonded_j in enumerate(self._bond_table):
            if len(bonded_j) < 3:
                continue
            for idx in range(len(bonded_j)-2):
                i = bonded_j[idx]
                k = bonded_j[idx+1]
                l = bonded_j[idx+2]
                if i < j and j < k:
                    bonded_i = self._bond_table[i]
                    bonded_k = self._bond_table[k]
                    bonded_l = self._bond_table[l]
                    # There should be only central atom j be shared between all of them
                    shared = list(set(bonded_i).intersection(bonded_k, bonded_l))
                    if shared == [j]:
                        self.__add_improper(i, j, k, l)

    def __add_improper(self, i, j, k, l):
        """
        Function for adding improper angles to the system.
        """
        improper_type = self.__get_improper_type(i, j, k, l)
        self.impropers.append((self.improper_types[improper_type], [i, j, k, l]))

    def __get_improper_type(self, i, j, k, l):
        """ Return the improper type between i, j, k and l """
        type_i = self.__system[i]
        type_j = self.__system[j]
        type_k = self.__system[k]
        type_l = self.__system[l]
        improper_type = type_i + type_j + type_k + type_l
        if improper_type[0] > improper_type[-1]:
            improper_type = improper_type[::-1]
        if improper_type not in self.improper_types:
            self.improper_types[improper_type] = len(self.improper_types) + 1
        return improper_type

    def _normalized(self, v):
        """ Returns normalized a vector. """
        return v / linalg.norm(v)

