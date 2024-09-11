
import numpy
from generation.lib.crystalline_system import CrystallineSystem


class PECrystallineSystem(CrystallineSystem):
    def __init__(self, settings):
        # Dynamically update instance attributes with settings values
        self.__dict__.update(settings)

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
        # List of forcefield type (e.g. 'C2') for each atom
        self._forcefield_types = dict(C2=0, H=1)

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
        self._add_angles()
        self._add_dihedrals()
        self._add_impropers()

    def _unit_cell(self):
        """
        Columns of unit cell are the fractional coordinate (a, b, and c).
        """
        return self.__orthorhombic()

    def __orthorhombic(self):
        """
        Parameters for orthorhombic unit cell of crystalline PE.
        Refer to 1975 - Avitabile - Low Temperature Crystal Structure of PE.
        """
        # Orthorhombic unit cell of PE
        return numpy.array([[7.121, 0.0, 0.0],
                            [0.0, 4.851, 0.0],
                            [0.0,  0.0, 2.548]])

    def _space_group(self):
        """
        Adding chains in a unit cell using space group symmetry operations.
        """
        return self.__Pnam()

    def __Pnam(self):
        """
        Fractional coordinates of Carbon and Hydrogen atoms in an orthorhombic
        unit cell of PE. Refer to:
        1998 - Bruno et. al. - Thermal expansion of polymers: Mechanisms in orthorhombic PE
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

