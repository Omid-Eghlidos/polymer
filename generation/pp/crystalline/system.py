

import numpy
from numpy import linalg, cross, dot
from math import cos, sin, pi, radians
from generation.lib.crystalline_system import CrystallineSystem


class PPCrystallineSystem(CrystallineSystem):
    def __init__(self, settings):
        # Dynamically update instance attributes with settings values
        self.__dict__.update(settings)

        # Add the material name
        self.material = 'iPP'
        # PP monomer has 9 atoms: 3 Carbons and 6 Hydrogens
        # Number of Carbon atoms in each PP monomer
        self.N_C = 3
        # Number of Hydrogen atoms in each PE monomer
        self.N_H = 6
        # Number of monomers per chain
        self.N_m = 3
        # Number of chains per unit cell
        if self.modification == 'alpha':
            self.N_c = 4
        elif self.modification == 'beta':
            self.N_c = 3
        # Total number of atoms in the system
        self.Nt = self.Na * self.Nb * self.Nc * (self.N_C + self.N_H) * self.N_m * self.N_c
        # Bond offsets between unitcells
        # The Carbon bond order in an iPP unit cell chain
        #   C3 (1)            C3 (4)            C3 (7)
        #   |                 |                 |
        #   C1 (2) - C2 (3) - C1 (5) - C2 (6) - C1 (8) - C2 (9)
        # Id of next C1 connected to C2 is +2, while the rest are +1
        self.bond_offsets = [1, 1, 2, 1, 1, 2, 1, 1, 2]
        # Atom types arrangement inside each unit cell chain
        self.unit_types  = ['C3', 'C1', 'C2', 'C3', 'C1', 'C2', 'C3', 'C1', 'C2']
        # List of forcefield type (e.g. 'C2') for each atom.
        self._forcefield_types = dict(C1=0, C2=1, C3=2, H=3)
        # Bond-angles of PP in rad (degrees) (Antoniadis, 1998)
        self._angle_size = dict(HCC=1.2800)

        # Add shared attributes from CrystallineSystem
        super().__init__()

        # Construct the system with the given size and store its parameters
        self._build_system()

    def _build_system(self):
        """
        Generate the polymer system with specified parameters.
        """
        self._create_simulation_box()
        self._add_carbon_atoms()
        self._add_hydrogen_atoms()
        self._add_angles()
        self._add_dihedrals()
        self._add_impropers()

    def _unit_cell(self):
        """
        Columns of unit cell are the fractional coordinate (a, b, and c).
        """
        if self.modification == 'alpha':
            return self.__monoclinic()
        elif self.modification == 'beta':
            return self.__hexagonal()

    def _space_group(self):
        """
        Adding chains in a unit cell using space group symmetry operations.
        """
        if self.modification == 'alpha':
            return self.__C2_c()
        elif self.modification == 'beta':
            return self.__P31_21()

    def __monoclinic(self):
        """
        Parameters for monoclinic unit cell of the alpha modification of crystalline PP.
        Refer to G. Natta et. al. (1959).
        """
        beta = 99.5 * numpy.pi / 180.0
        return numpy.array([[6.63,  0.00, 6.50*numpy.cos(beta)],
                            [0.00, 20.78, 0.0],
                            [0.00,  0.00, 6.50*numpy.sin(beta)]])

    def __hexagonal(self):
        """
        Parameters for hexagonal unit cell of the beta modification of crystalline PP.
        Refer to Dino R. Ferro et al. (1998).
        """
        gamma = 120.0 * numpy.pi / 180.0
        return numpy.array([[11.03, 11.03*numpy.cos(gamma), 0.00],
                            [0.00 , 11.03*numpy.sin(gamma), 0.00],
                            [0.00 ,  0.00 , 6.50]])

    def __C2_c(self):
        """
        Obtain fractional coordinates of all the chains in the unit cell using
        C2/c space group symmetry given the first chain of the alpha modification
        in its monoclinic unit cell, G. Natta et. al. (1959)
        """
        # Carbons of the first chain inside the unit cell
        C1 = numpy.array([[-0.0727, 0.2291, 0.2004],  #C3 0: 1
                          [-0.0765, 0.1592, 0.2788],  #C1 1: 0 2
                          [-0.1021, 0.1602, 0.5098],  #C2 2: 1 4
                          [-0.3087, 0.0589, 0.4941],  #C3 3: 4
                          [-0.1146, 0.0928, 0.6057],  #C1 4: 2 3 5
                          [-0.1044, 0.0854, 0.8428],  #C2 5: 4 7
                          [ 0.2775, 0.0797, 0.9260],  #C3 6: 7
                          [ 0.0872, 0.1156, 0.9730],  #C1 7: 5 6 8
                          [ 0.1026, 0.1221, 1.2109]]) #C2 8: 7
        C2 = C1.copy()
        C2[:,0] =  C1[:,0]
        C2[:,1] = -C1[:,1]
        C2[:,2] =  C1[:,2] + 0.5

        C3 = C1.copy()
        C3[:,0] =  C2[:,0] - 0.5
        C3[:,1] =  C2[:,1] + 0.5
        C3[:,2] =  C2[:,2]

        C4 = C1.copy()
        C4[:,0] =  C1[:,0] - 0.5
        C4[:,1] =  C1[:,1] - 0.5
        C4[:,2] =  C1[:,2]
        return C1, C2, C3, C4

    def __P31_21(self):
        """
        Adding chains in a unitcell using symmetry operations, coordinates
        from D.L. Dorset et al. (1998) iPP beta-phase: A study in frustration
        """
        # Beta unit cell has three chains A, B, and C, each with 9 Carbon atoms
        # For each chain only the first 3 rabon atoms of the monomer are given
        # while the other 6 will be obtained through symmetry operation
        A = numpy.array([[0.2311 , 0.1785, 0.5951],  # C3
                         [0.0823 , 0.0772, 0.6740],  # C1
                         [0.0696 , 0.0692, 0.9104]]) # C2
        C1 = self.__full_carbon_chain(A, 0)

        B = numpy.array([[0.5426, 0.6813, 0.4169],   # C3
                         [0.4199, 0.6977, 0.4958],   # C1
                         [0.4169, 0.6991, 0.7328]])  # C2
        C2 = self.__full_carbon_chain(B, 1)

        C = numpy.array([[0.8910, 0.4606, 0.6334],   # C3
                         [0.7410, 0.4004, 0.7168],   # C1
                         [0.7444, 0.4040, 0.9538]])  # C2
        C3 = self.__full_carbon_chain(C, 2)
        return C1, C2, C3

    def __full_carbon_chain(self, CC, i):
        """
        Create the full Carbon chain from the first 3 Carbon atoms of a
        beta modification chain using the space group symmetry operation from
        Dino R. Ferro et al. (1998) """
        Helix_center = numpy.array([[0.0    , 0.0    , 0.0],
                                    [1.0/3.0, 2.0/3.0, 0.0],
                                    [2.0/3.0, 1.0/3.0, 0.0]])
        C = numpy.zeros([9,3])
        C[0:3,:] =  CC
        C[0:3,:] =  C[0:3,:] - Helix_center[i,:]
        C[3:6,0] = -C[0:3,1]
        C[3:6,1] =  C[0:3,0] - C[0:3,1]
        C[3:6,2] =  C[0:3,2] + 1/3
        C[6:9,0] = -C[0:3,0] + C[0:3,1]
        C[6:9,1] = -C[0:3,0]
        C[6:9,2] =  C[0:3,2] + 2/3
        C[0:3,:] =  C[0:3,:] + Helix_center[i,:]
        C[3:6,:] =  C[3:6,:] + Helix_center[i,:]
        C[6:9,:] =  C[6:9,:] + Helix_center[i,:]
        return C

    def _add_hydrogen_atoms(self):
        """
        Add hydrogen atoms specific to the PP system.

        This method overrides the `_add_hydrogen_atoms` placeholder method in the
        PolymerSystem base class.
        """
        nc = 1
        for i, bonded in enumerate(self.bond_table):
            if not bonded:
                continue
            # Bond vectors from atom i to atom in each bond
            bond_vec = [self._bond_vector(i, b) for b in bonded]
            if len(bonded) == 1:
                self._add_methyl_hydrogens(i, bond_vec)
            elif len(bonded) == 2:
                self._add_gemini_hydrogens(i, bond_vec)
            elif len(bonded) == 3:
                self.__add_chiral_hydrogen(i, bond_vec)
        # Create the bond table for all the atoms in the system
        self._construct_bond_table()

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

        # Position the methyl Carbon (C3) at the tetrahedral angle (~109.0°)
        tetrahedral_angle = 109.0
        # Convert to radians and adjust by 90°
        phi = radians(tetrahedral_angle - 90.0)
        # Theta shift to rotate all the 3 methyl hydrogens
        theta_shift = pi/6.0
        for theta in [0.0, 2.0/3.0*pi, 4.0/3.0*pi]:
            theta += theta_shift
            # Unit vector from methyl carbon towards a methyl hydrogen
            r  = sin(phi)*u + cos(phi) * (cos(theta)*v + sin(theta)*w)
            assert round(linalg.norm(r), 1) == 1.0
            # Position of a methyl hydrogen
            rH = self.atoms[i][2] + self._bond_length['CH']*r
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
        theta_h = self._angle_size['HCC']
        for j in range(2):
            # Direction of the hydrogen
            n = 1 if j == 0 else -1
            # Position of the gemini hydrogens
            rH = self.atoms[i][2] +\
                 self._bond_length['CH']*(sin(theta_h/2)*u + n*cos(theta_h/2)*v)
            self._add_atom(self._atom_id, self.atoms[i][0], 'H', rH)
            self._add_bond(i, self._atom_id)
            self._atom_id += 1

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
        rH = self.atoms[i][2] + self._bond_length['CH']*n
        self._add_atom(self._atom_id, self.atoms[i][0], 'H', rH)
        self._add_bond(i, self._atom_id)
        self._atom_id += 1

