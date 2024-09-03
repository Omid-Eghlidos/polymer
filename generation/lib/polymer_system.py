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


class PolymerSystem():
    """
    A class to represent a polymer system for molecular dynamics simulations.

    The `PolymerSystem` class generates a polymer system containing a specified
    number of polymer chains and monomers per chain. It sets up the simulation
    box, adds carbon and hydrogen atoms, and defines the bonding structure,
    including bond lengths, angles, dihedrals, and improper torsions.

    Attributes
    ----------
    Nc : int
        Number of chains per system.
    Nm : int
        Number of monomers per chain.
    N_C : int
        Number of carbon atoms in each monomer.
    N_H : int
        Number of hydrogen atoms in each monomer.
    Na : int
        Total number of atoms in the system, including initiators and terminators for each chain.
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
    def __init__(self, settings):
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
        # Number of chains per system
        self.Nc = settings.Nc
        # Number of monomers per chain
        self.Nm = settings.Nm
        # Number of Carbon atoms in each monomer
        self.N_C = settings.monomer_atom_numbers['C']
        # Number of Hydrogen atoms in each monomer
        self.N_H = settings.monomer_atom_numbers['H']
        # Total number of atoms in the system + an initiator and a terminator for each chain
        self.Na = self.Nc * self.Nm * (self.N_C + self.N_H) + 2 * self.Nc
        # Simulation box
        self.box = numpy.zeros((3, 3))
        # Type of each atom
        self.atom_types = -1 * numpy.ones(self.Na, dtype=numpy.int64)
        # Each atom chains
        self.atom_chains = -1 * numpy.ones(self.Na, dtype=numpy.int64)
        # Each atom charges
        self.atom_charges = -1 * numpy.ones(self.Na)
        # Each atom coordinate
        self.atom_coords = -1 * numpy.ones((self.Na, 3))
        # Pair bonds and their types
        self.bonds = []
        self.unique_bond_types = []
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

        # Mass of different atom types (g/cm^3)
        self._atom_masses = dict(C=12.0112, H=1.00797)
        # Bond length between the atoms in the system
        self._bond_length = dict(CC=1.54, CH=1.10)
        # Types of different forcefield atoms in the system
        self._forcefield_atom_types = settings.forcefield_atom_types
        # Density of the amorphous system (g/cm^3)
        self._rho = settings.density
        # Bond-angles sizes in the amorphous structure in rad
        self._angle_size = settings.angle_size
        # Backbone and methyl carbons
        self._atom_id   = 0
        self._backbones = []
        self._hydrogens = []

        # List of atom charges for each type in COMPASS and PCFF (Maple, 1994)
        # C1: Chiral Carbon (cC), C2: Achiral Carbon (aC), C3: Methyl Carbon, H : Hydrogen
        self.__charge_types = {'C1': -0.053, 'C2': -0.106, 'C3': -0.159, 'H': 0.053}
        # Dihedral angles
        self.__dihedral_size = [pi]
        self.__random_flip = False
        if hasattr(settings, 'random_flip'):
        	self.__random_flip = settings.random_flip
        self.__random_variation = settings.random_variation

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
        self._add_carbons()
        self._add_hydrogens()
        self.__map_to_PBC()
        self.__add_angles()
        self.__add_dihedrals()
        self.__add_impropers()

    def _create_simulation_box(self):
        """
        Determine the simulation box dimensions based on the system mass and density.

        Parameters
        ----------
        Nc : int
            Number of chains per system.
        Nm : int
            Number of monomers per chain.
        atom_masses : dict
            Dictionary of atom types and their masses.
        rho : float
            Density of the amorphous system.
        box : numpy.ndarray
            The simulation box dimensions.

        Returns
        -------
        numpy.ndarray
            The dimensions of the simulation box.
        """
        # Mass of Carbons in one chain of the system
        chain_carbon_mass = self.N_C * self.Nm*self._atom_masses['C']
        # Mass of Hydrogens in one chain of the system
        chain_hydrogen_mass = self.N_H * self.Nm*self._atom_masses['H']
        # Mass of one chain of the system for the given number of atoms
        chain_mass = chain_carbon_mass + chain_hydrogen_mass
        # Total mass of the system of Nc chains
        mass = self.Nc * chain_mass
        # Volume of the system for the given density in Angstrom^3
        # To convert from g/cm^3 to g/A^3 should divide by 10^24
        volume = (mass/N_A) / (self._rho/1e24)
        # Assuming cubic box for the amorphous system, dimension = cubic root of Volume
        for k in range(3):
            self.box[k,k] = volume**(1.0/3.0)

    def _add_carbons(self):
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
        coords.append(coords[-1] + self._bond_length['CC']*numpy.array([1.0, 0.0, 0.0]))
        # The 3rd backbone Carbon should be located at coordinates that makes the
        # specified CCC angle with the previous 2 backbone Carbons. Therefore,
        # 3rd atom coordinates make a complement angle with the plane of the previous 2 atoms.
        q = pi - self._angle_size['CCC']
        coords.append(coords[-1] + self._bond_length['CC']*numpy.array([cos(q), sin(q), 0.0]))
        # There are 2 Carbons in the backbone of each monomer
        for _, phi in zip(range(3, self.Nm*2), cycle(self.__dihedral_size)):
            b1 = coords[-2] - coords[-3]
            b2 = coords[-1] - coords[-2]
            # Rotation of bond angle to cis-conformation
            Ra = self.__rotation_about_axis(numpy.cross(b1, b2), q)
            # Randomly flip from gauche to anti-gauche
            if self.__random_flip:
                if abs(phi) < radians(self._angle_size['CCC']) and\
                                                random.uniform(0.0, 1.0) < 0.5:
                    phi = -phi
            # Randomly changes angle from Gaussian distribution with 15 degrees standard dev.
            if self.__random_variation:
                mu, sigma = 0.0, pi/12.0
                phi += random.normal(mu, sigma)
            # Rotation to correct dihedral angle
            Rd = self.__rotation_about_axis(b2, phi)
            b = Rd @ Ra @ b2
            coords.append(coords[-1] + b)
        # Scale coordinates by bond length
        self._backbones = numpy.array(coords)

    def _add_hydrogens(self):
        """
        Placeholder method for adding hydrogens.

        Subclasses should override this method.
        """
        pass

    def _add_methyl_hydrogens(self, i, bond_vec):
        """
        Add 3 hydrogen atoms to each methyl carbon (C3) in the system.

        Parameters
        ----------
        atom_coords : numpy.ndarray
            Array of atom coordinates.
        atom_chains : numpy.ndarray
            Array of atom chain indices.
        atom_types : numpy.ndarray
            Array of atom types.
        atom_charges : numpy.ndarray
            Array of atom charges.
        bonds : list of lists
            List containing pairs of bonded atoms.
        bond_table : list of lists
            List containing bonded atoms for each atom.
        atom_id : int
            The current atom ID.
        bond_vec : list of numpy.ndarray
            List of bond vectors for the current atom.
        i : int
            Index of the current atom.
        bond_length_CH : float
            Bond length between carbon and hydrogen.

        Returns
        -------
        int
            Updated atom ID after adding hydrogens.
        """
        j = next(b for b in self.bonds if i in b)[1]
        bonds = [b[:] for b in self.bonds if j in b and i not in b]
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

        # Position the methyl Carbon (C3) at the tetrahedral angle (~109.0°)
        tetrahedral_angle = 109.0
        # Convert to radians and adjust by 90°
        phi = radians(tetrahedral_angle - 90.0)
        # Theta shift to rotate all the 3 methyl hydrogens
        theta_shift = pi/6.0
        for theta in [0.0, 2.0/3.0*pi, 4.0/3.0*pi]:
            theta += theta_shift
            # Unit vector from methyl carbon towards a methyl hydrogen
            r  = sin(phi)*u + cos(phi)*(cos(theta)*v + sin(theta)*w)
            assert round(linalg.norm(r), 1) == 1.0
            # Position of a methyl hydrogen
            rH = self.atom_coords[i] + self._bond_length['CH']*r
            self._add_atom(self._atom_id, self.atom_chains[i], 'H', rH)
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
        theta_h = self._angle_size['HCC']
        for j in range(2):
            # Direction of the hydrogen
            n = 1 if j == 0 else -1
            # Position of the gemini hydrogens
            rH = self.atom_coords[i] +\
                 self._bond_length['CH']*(sin(theta_h/2)*u + n*cos(theta_h/2)*v)
            self._add_atom(self._atom_id, self.atom_chains[i], 'H', rH)
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

    def _add_atom(self, atom_id, num_chain, atom_type, atom_coords):
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
        self.atom_chains[atom_id]  = num_chain
        self.atom_types[atom_id]   = self._forcefield_atom_types[atom_type]
        self.atom_charges[atom_id] = self.__charge_types[atom_type]
        self.atom_coords[atom_id]  = atom_coords

    def _add_bond(self, i, j):
        """
        Add a bond and its bond type.

        Parameters
        ----------
        bonds : list
            List to store the bonds.
        bond_types : list
            List to store the bond types.
        i : int
            Index of the first atom.
        j : int
            Index of the second atom.
        atom_types : list
            List of atom types.
        """
        self.bonds.append([i, j])
        # Bond-types are either C-C (type 0) or C-H (type 1)
        bond_type = 1 if self._forcefield_atom_types['H'] in\
                                [self.atom_types[i], self.atom_types[j]] else 0
        self.bond_types.append(bond_type)
        types = [1 if self.atom_types[k] == self._forcefield_atom_types['H']\
                                                        else 0 for k in [i, j]]
        if types[0] > types[1]:
            types = types[::-1]
        if types not in self.unique_bond_types:
            self.unique_bond_types.append(types)

    def _construct_bond_table(self):
        """
        Create a table where each row contains atoms bonded to an atom with ID as the row number.

        Parameters
        ----------
        Na : int
            Total number of atoms.
        bonds : list
            List of bonds.

        Returns
        -------
        list
            Bond table.
        """
        self.bond_table = [[] for _ in range(self.Na)]
        for i, j in self.bonds:
            self.bond_table[i].append(j)
            self.bond_table[j].append(i)
        for bond in self.bond_table:
            bond.sort()

    def __add_angles(self):
        """
        Traverses the bond list and defines all bond angles.

        Parameters
        ----------
        atom_coords : numpy.ndarray
            Array of atom coordinates.
        angles : list
            List to store the angles.
        angle_types : list
            List to store the angle types.
        bonds : list
            List of bonds.
        atom_types : list
            List of atom types.
        """
        for j in range(len(self.atom_coords)):
            # Set J contains all atoms bonded to atom j.
            J = self.__atoms_bonded_to(j)
            for i in sorted(J):
                for k in sorted(J):
                    # Don't add angle twice.
                    if i < k:
                        self.__add_angle([i, j, k])

    def __add_angle(self, ijk):
        """
        Function for adding angles to the system.

        Parameters
        ----------
        angles : list
            List to store the angles.
        angle_types : list
            List to store the angle types.
        ijk : list
            List of three atom indices forming an angle.
        atom_types : list
            List of atom types.
        """
        types = [1 if self.atom_types[i] == self._forcefield_atom_types['H'] else 0 for i in ijk]
        # Accounts for ABC -> CBA symmetry of angle types.
        if types[0] > types[2]:
            types = types[::-1]
            ijk = ijk[::-1]
        if types not in self.angle_types:
            self.angle_types.append(types)
        self.angles.append((self.angle_types.index(types), ijk))

    def __add_dihedrals(self):
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
        atom_coords : numpy.ndarray
            Array of atom coordinates.
        atom_types : list
            List of atom types.
        """
        for b in self.bonds:
            j, k = sorted(b)
            J = [i for i in self.__atoms_bonded_to(b[0]) if i != b[1]]
            K = [i for i in self.__atoms_bonded_to(b[1]) if i != b[0]]
            for i in J:
                for l in K:
                    self.__add_dihedral([i,j,k,l])

    def __add_dihedral(self, ijkl):
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
        types = [1 if self.atom_types[i] == self._forcefield_atom_types['H'] else 0 for i in ijkl]
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
        self.dihedrals.append((self.dihedral_types.index(types), ijkl))

    def __add_impropers(self):
        """
        Traverse the bond list and define all improper angles.

        Parameters
        ----------
        bonds : list of lists
            List containing pairs of bonded atoms.
        atom_types : list of int
            List of atom types.
        atom_coords : numpy.ndarray
            Array of atom coordinates.

        Returns
        -------
        list of tuples
            A list of improper angles defined by four atom indices.
        list of lists
            A list of improper angle types corresponding to each improper angle.
        """
        for b in self.bonds:
            j = sorted(b)[0]
            bonded_to_j = sorted(self.__atoms_bonded_to(j))
            # The atom needs to have at least three bonds to have an improper
            if len(bonded_to_j) > 3:
                for i in bonded_to_j:
                    for k in bonded_to_j:
                        for l in bonded_to_j:
                            if i < k and i < l and l < k:
                                self.__add_improper([i, j, k, l])
        self.impropers = sorted(self.impropers, key = lambda x: x[1])

    def __add_improper(self, ijkl):
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
        types = sorted([1 if self.atom_types[i] == self._forcefield_atom_types['H']\
                                                        else 0 for i in ijkl])
        # Possible improper types
        # CCCH <=> HCCC
        # CCHH <=> HHCC
        # CHHH <=> HHHC
        # NOTE: This is not general.
        if types not in self.improper_types:
            self.improper_types.append(types)
        if (self.improper_types.index(types), ijkl) not in self.impropers:
            self.impropers.append((self.improper_types.index(types), ijkl))

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
        atom_coords : numpy.ndarray
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
        return self.__wrap_vector(self.atom_coords[j] - self.atom_coords[i])

    def __wrap_vector(self, v):
        """
        Return the shortest vector considering periodic boundary conditions.

        Parameters
        ----------
        v : numpy.ndarray
            The vector to wrap.
        box : numpy.ndarray
            The simulation box dimensions.

        Returns
        -------
        numpy.ndarray
            The wrapped vector.
        """
        for i in range(3):
            while v[i] > 0.5*self.box[i,i]:
                v[i] -= self.box[i,i]
            while v[i] < -0.5*self.box[i,i]:
                v[i] += self.box[i,i]
        return v

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

    def __atoms_bonded_to(self, i):
        """
        Returns a list of all atoms bonded to atom i.

        Parameters
        ----------
        i : int
            Index of the atom.
        bonds : list
            List of bonds.

        Returns
        -------
        list
            Atoms bonded to atom i.
        """
        return [next(a for a in b if a != i) for b in self.bonds if i in b]

    def __map_to_PBC(self):
        """
        Maps atom coordinates back to the periodic boundary condition.

        Parameters
        ----------
        atom_coords : numpy.ndarray
            Array of atom coordinates.
        box : numpy.ndarray
            Array representing the simulation box dimensions.

        Returns
        -------
        numpy.ndarray
            Array of atom coordinates mapped to the periodic boundary condition.
        """
        for i in range(len(self.atom_coords)):
            for j in range(3):
                while self.atom_coords[i][j] > self.box[j,j]:
                    self.atom_coords[i][j] -= self.box[j,j]
                while self.atom_coords[i][j] < 0:
                    self.atom_coords[i][j] += self.box[j,j]

