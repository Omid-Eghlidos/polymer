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
from numpy import linalg, random, cross, dot
from scipy.constants import N_A
from scipy.spatial.transform import Rotation
from math import cos, sin, pi, radians
from itertools import cycle
from generation.lib.polymer_system import PolymerSystem


class AmorphousAlkeneSystem(PolymerSystem):
    """
    A class to represent a polymer system for molecular dynamics simulations.

    The `PolymerSystem` class generates a polymer system containing a specified
    number of polymer chains and monomers per chain. It sets up the simulation
    box, adds carbon and hydrogen atoms, and defines the bonding structure,
    including bond lengths, angles, dihedrals, and improper torsions.

    Attributes
    ----------
    None

    Methods
    -------
    _build_system():
        Constructs the polymer system by setting up atoms, bonds, angles, and dihedrals.
    _create_simulation_box():
        Determines the simulation box dimensions based on system mass and density.
    _add_carbons():
        Placeholder method for adding carbon atoms to the system.
    _add_backbone_carbons():
        Adds backbone carbon atoms to the polymer system.
    _add_hydrogens():
        Placeholder method for adding hydrogen atoms to the system.
    _add_methyl_hydrogens(i, bond_vec):
        Adds three hydrogen atoms to each methyl carbon in the system.
    _add_gemini_hydrogens(i, bond_vec):
        Adds two gemini hydrogens to each achiral/gemini carbon in the system.
    _add_initiator(i, bond_vec):
        Placeholder method for adding an initiator hydrogen atom.
    _add_terminator(i, bond_vec):
        Placeholder method for adding a terminator hydrogen atom.
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
        # Add shared attributes from PolymerSystem
        super().__init__()

        # Mass of different atom types (g/mol)
        self._ff_sizes['masses'] = dict(C=12.0112, H=1.00797)
        # Bond length between the atoms in the system
        self._ff_sizes['bonds'] = dict(CC=1.54, CH=1.10)
        # List of atom charges for each type in COMPASS and PCFF (Maple, 1994)
        # C1: Chiral Carbon (cC), C2: Achiral Carbon (aC), C3: Methyl Carbon, H : Hydrogen
        self._ff_sizes['charges'] = {'C1': -0.053, 'C2': -0.106, 'C3': -0.159, 'H': 0.053}
        # Backbone and methyl carbons
        self._backbones = []
        self._hydrogens = []
        # Atom ID counter
        self._atom_id   = 0

    def _create_simulation_box(self):
        """
        Determine the simulation box dimensions based on the system mass and density.

        Returns
        -------
        None
        """
        # Mass of Carbons in one chain of the system
        chain_carbon_mass = self.N_C * self.Nm*self._ff_sizes['masses']['C']
        # Mass of Hydrogens in one chain of the system
        chain_hydrogen_mass = self.N_H * self.Nm*self._ff_sizes['masses']['H']
        # Mass of one chain of the system for the given number of atoms
        chain_mass = chain_carbon_mass + chain_hydrogen_mass
        # Total mass of the system of Nc chains (g/mol)
        mass = self.Nc * chain_mass
        # Volume of the system for the given density in Angstrom^3
        # To convert from g/cm^3 to g/A^3 should divide by 10^24
        volume = (mass/N_A) / (self.rho/1e24)
        # Assuming cubic box for the amorphous system, dimension = cubic root of Volume
        for k in range(3):
            self.box[k,k] = volume**(1.0/3.0)

    def _add_carbon_atoms(self):
        """
        Placeholder method for adding carbons.

        Subclasses should override this method.
        """
        pass

    def _add_backbone_carbons(self):
        """
        Add backbone carbon atoms to the polymer system.

        This method generates `2 * Nm` backbone carbon atoms for the polymer
        chains. The coordinates of the backbone atoms are calculated based on
        the bond lengths, bond angles, and dihedral angles. If multiple dihedral
        angles are specified, the method cycles through them, creating a chain
        where the dihedral angle alternates.

        The method starts by generating random coordinates for the first carbon
        atom in the simulation box. It then calculates the positions of the
        subsequent backbone carbon atoms, ensuring the correct bond angles and
        dihedral angles are maintained.

        Returns
        -------
        None
        """
        # Set the seed for the random function
        #random.seed(0)
        # To generate coordinates of a chain in a configuration, coorinates of
        # the first 3 backbone atoms are required.
        # For 1st backbond atom generate a random starting coordinates in the simulation box
        coords = [numpy.array([random.uniform(0, self.box[0,0]),
                               random.uniform(0, self.box[1,1]),
                               random.uniform(0, self.box[2,2])])]
        # Assume 2nd backbone Carbon coordinates only changes along x-axis with
        # respect to the 1st backbone atom coordinates
        coords.append(coords[-1] + self._ff_sizes['bonds']['CC']*numpy.array([1.0, 0.0, 0.0]))
        # The 3rd backbone Carbon should be located at coordinates that makes the
        # specified CCC angle with the previous 2 backbone Carbons. Therefore,
        # 3rd atom coordinates make a complement angle with the plane of the previous 2 atoms.
        q = pi - self._ff_sizes['angles']['CCC']
        coords.append(coords[-1] + self._ff_sizes['bonds']['CC']*numpy.array([cos(q), sin(q), 0.0]))
        # There are 2 Carbons in the backbone of each monomer
        for _, phi in zip(range(3, self.Nm*2), cycle(self._ff_sizes['torsions'])):
            b1 = coords[-2] - coords[-3]
            b2 = coords[-1] - coords[-2]
            # Rotation of bond angle to cis-conformation
            Ra = self.__rotation_about_axis(numpy.cross(b1, b2), q)
            # Randomly flip from gauche to anti-gauche
            if self.random_flip:
                if abs(phi) < radians(self._ff_sizes['angles']['CCC']) and\
                                                random.uniform(0.0, 1.0) < 0.5:
                    phi = -phi
            # Randomly changes angle from Gaussian distribution with 15 degrees standard dev.
            if self.random_variation:
                mu, sigma = 0.0, pi/12.0
                phi += random.normal(mu, sigma)
            # Rotation to correct dihedral angle
            Rd = self.__rotation_about_axis(b2, phi)
            b = Rd @ Ra @ b2
            coords.append(coords[-1] + b)
        # Scale coordinates by bond length
        self._backbones = numpy.array(coords)

    def _add_hydrogen_atoms(self):
        """
        Placeholder method for adding hydrogens.

        Subclasses should override this method.
        """
        pass

    def _add_methyl_hydrogens(self, i, bond_vec):
        """
        Add 3 hydrogen atoms to each methyl carbon (C3) in the system.

        Updates atom IDs, coordinates, and bonds after adding methyl hydrogens.

        Parameters
        ----------
        i : int
            Index of the current atom.
        bond_vec : list of numpy.ndarray
            List of bond vectors for the current atom.

        Returns
        -------
        None
        """
        j = self.bond_table[i][0]
        bonds = [[j, k] for k in self.bond_table[j] if k != i]
        if len(bonds) == 2:
            # Bond from C_i-1 to C_i+1
            rcc = self._bond_vector(bonds[0][1], bonds[1][1])
        else:
            # For start and end of the chains, we can't use skeleton vector
            rcc = [0, 0, 1]
        # Basis vectors u, v, and w for a coordinate system created about the methyl Carbon
        u = -self._normalized(bond_vec[0])
        v = self._normalized(cross(u, rcc))
        w = cross(u, v)

        # Position the methyl Carbon (C3) at the ideal tetrahedral angle of ~109.5°
        tetrahedral_angle = 109.5
        # Convert to radians and adjust by 90°
        phi = radians(tetrahedral_angle - 90.0)
        # Shift the starting angle - rotate all the 3 methyl hydrogens
        theta_shift = pi/6.0
        # Ideal value for the HCH in the CH3 tetrahedral
        HCH = 2*pi/3.0
        for theta in [0.0, HCH, 2*HCH]:
            theta += theta_shift
            # Unit vector from methyl carbon towards a methyl hydrogen
            r  = sin(phi)*u + cos(phi)*(cos(theta)*v + sin(theta)*w)
            assert round(linalg.norm(r), 1) == 1.0
            # Position of a methyl hydrogen
            rH = self.atoms[i][2] + self._ff_sizes['bonds']['CH']*r
            self._add_atom(self._atom_id, self.atoms[i][0], 'H', rH)
            self._add_bond(i, self._atom_id)
            self._atom_id += 1

    def _add_gemini_hydrogens(self, i, bond_vec):
        """
        Add 2 gemini hydrogens to each achiral/gemini carbon (C2) in the system.

        Parameters
        ----------
        i : int
            Index of the current carbon atom.
        bond_vec : list of numpy.ndarray
            List of bond vectors for the current carbon atom.

        Returns
        -------
        Updated atom IDs, coordinates, and bonds after adding gemini hydrogens.
        """
        # Common bisector between angles H+_i-C_i-H-_i and C_(i-1)-C_i-C_(i+1)
        bi = -self._normalized(bond_vec[0])
        bj =  self._normalized(bond_vec[1])
        # Unit vector along bisector of angle C_(i-1)-C_i-C_(i+1) in the
        # direction from C_i towards H+_i and H-_i
        u = (bi - bj) / numpy.sqrt(2.0 * (1.0 - dot(bi, bj)))
        # Unit vector normal to the plane C_(i-1)-C_i-C_(i+1)
        v = self._normalized(cross(bi, bj))

        # Angle between the gemini hydrogens connected to the C2 in radian
        theta_h = self._ff_sizes['angles']['HCC']
        for j in range(2):
            # Direction of the hydrogen
            n = 1 if j == 0 else -1
            # Position of the gemini hydrogens
            rH = self.atoms[i][2] +\
                 self._ff_sizes['bonds']['CH']*(sin(theta_h/2)*u + n*cos(theta_h/2)*v)
            self._add_atom(self._atom_id, self.atoms[i][0], 'H', rH)
            self._add_bond(i, self._atom_id)
            self._atom_id += 1

    def _add_initiator(self, i, bond_vec):
        """
        Placeholder method for adding a hydrogen atom as an initiator to the first C.

        Subclasses should override this method.
        """
        pass

    def _add_terminator(self, i, bond_vec):
        """
        Placeholder method for adding a hydrogen atom as a terminator to the last C.

        Subclasses should override this method.
        """
        pass

    def _identify_unique_bond_types(self):
        """
        Function for identifying the unique types for the bonds formed by any
        atoms i and j in the system.
        """
        for i in self._ff_types['atoms']:
            for j in self._ff_types['atoms']:
                types = ''.join(sorted([i[0], j[0]]))
                if types == 'HH': continue
                if types not in self._ff_types['bonds']:
                    self._ff_types['bonds'][types] = len(self._ff_types['bonds'])

    def _get_bond_type(self, i, j, covalent):
        """
        Function for getting the unique type for the bond formed by i and j.
        """
        ai = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[i]][0][0]
        aj = [a for a, t in self._ff_types['atoms'].items() if t == self.atom_types[j]][0][0]
        return self._ff_types['bonds'][''.join(sorted([ai.upper(), aj.upper()]))]

    def _get_angle_type(self, ijk):
        """
        Function for getting the unique type for the angle formed by i, j, k.

        Parameters
        ----------
        ijk : list
            List of three atom indices forming an angle.
        """
        types = [1 if self.atom_types[i] == self._ff_types['atoms']['H']\
                                                    else 0 for i in ijk]
        # Accounts for ABC -> CBA symmetry of angle types.
        if types[0] > types[2]:
            types = types[::-1]
            ijk = ijk[::-1]
        if types not in self.angle_types:
            self.angle_types.append(types)
        return (self.angle_types.index(types), ijk)

    def _get_dihedral_type(self, ijkl):
        """
        Function for adding dihedral torsion angles to the system.

        Parameters
        ----------
        dihedrals : list
            List to store the dihedrals.
        dihedral_types : list
            List to store the dihedral types.
        ijkl : list
            List of four atom indices forming a dihedral.
        atom_types : list
            List of atom types.
        """
        types = [1 if self.atom_types[i] == self._ff_types['atoms']['H']\
                                                   else 0 for i in ijkl]
        # Possible dihedral types
        # HCCH <=> HCCH
        # CCCH <=> HCCC
        # CCCC <=> CCCC
        # NOTE: This is not general.
        if types[0] > types[-1]:
            ijkl = ijkl[::-1]
            types = types[::-1]
        if types not in self.dihedral_types:
            self.dihedral_types.append(types)
        return (self.dihedral_types.index(types), ijkl)

    def _get_improper_type(self, ijkl):
        """
        Function for adding improper angles to the system.

        Parameters
        ----------
        ijkl : list of int
            A list of four atom indices defining an improper angle.
        atom_types : list of int
            List of atom types.

        Returns
        -------
        tuple
            A tuple containing the improper angle defined by four atom indices and its type.
        """
        types = sorted([1 if self.atom_types[i] == self._ff_types['atoms']['H']\
                                                             else 0 for i in ijkl])
        # Possible improper types
        # CCCH <=> HCCC
        # CCHH <=> HHCC
        # CHHH <=> HHHC
        # CCCC <=> CCCC (PP)
        # NOTE: This is not general
        if types not in self.improper_types:
            self.improper_types.append(types)
        return (self.improper_types.index(types), ijkl)

    def __rotation_about_axis(self, u, q):
        """
        Generate a rotation matrix that rotates a vector about a specified axis by a given angle.

        Parameters
        ----------
        u : numpy.ndarray
            The axis vector to rotate around.
        q : float
            The angle in radians to rotate.

        Returns
        -------
        numpy.ndarray
            The rotation matrix.
        """
        return Rotation.from_rotvec(numpy.array(u) / linalg.norm(u) * q).as_matrix()


