import numpy
from generation.lib.polymer_system import PolymerSystem


class CrystallineAlkeneSystem(PolymerSystem):
    def __init__(self):
        # Add shared attributes from PolymerSystem
        super().__init__()

        # Simulation box dimensions
        self.box = numpy.zeros((3, 3))
        # Mass of different atom types (g/cm^3)
        self._ff_sizes['masses'] = dict(C=12.0112, H=1.00797)
        # Bond length between the atoms in the system
        self._ff_sizes['bonds'] = dict(CC=1.54, CH=1.10)
        # List of atom charges for each type in COMPASS and PCFF (Maple, 1994)
        # C1: Chiral Carbon (cC), C2: Achiral Carbon (aC), C3: Methyl Carbon, H : Hydrogen
        self._ff_sizes['charges'] = {'C1': -0.053, 'C2': -0.106, 'C3': -0.159, 'H': 0.053}
        # Atom ID counter
        self._atom_id   = 0
        # Atoms symbolic types in the whole system
        self._system_types = ''

    def _create_simulation_box(self):
        # Unit cell parameters of the generated crystal system
        self.box = numpy.dot(self._unit_cell(), numpy.diag([self.Na, self.Nb, self.Nc]))

    def _unit_cell(self):
        """
        Columns of unit cell are the fractional coordinate (a, b, and c).
        """
        pass

    def _space_group(self):
        """
        Adding chains in a unit cell using space group symmetry operations.
        """
        pass

    def _add_carbon_atoms(self):
        """
        Add Carbon atoms in the system.
        """
        # Generate chains by adding unit cell chain along chain direction (c-axis)
        chains = [numpy.vstack([chain + [0, 0, k] for k in range(self.Nc)])
                                              for chain in self._space_group()]
        # Find the atom types of a chain by repeating unit cell pattern
        chains_types = self.unit_types * self.Nc
        chain_id = 0
        for na in range(self.Na):
            for nb in range(self.Nb):
                chains_ab = [chain + [na, nb, 0] for chain in chains]
                for chain in chains_ab:
                    self._atom_id = len(chain) * chain_id
                    for idx, coords in enumerate(chain):
                        bond_to = self._atom_id + self.bond_offsets[idx % len(self.bond_offsets)]
                        if bond_to >= (chain_id+1)*len(chain):
                            if self.pbc:
                                bond_to = bond_to%len(self.bond_offsets) + len(chain)*chain_id
                        coords = numpy.dot(coords, self._unit_cell().T)
                        self._add_atom(self._atom_id, chain_id, chains_types[idx], coords)
                        self._add_bond(self._atom_id, bond_to)
                        self._atom_id += 1
                    chain_id += 1
        # Create the bond table for the Carbon atoms added to the system
        self._construct_bond_table()

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
        ai, aj = self.system_types[i][0], self.system_types[j][0]
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
        # NOTE: This is not general
        if types[0] > types[-1]:
            ijkl = ijkl[::-1]
            types = types[::-1]
        if types not in self.dihedral_types:
            self.dihedral_types.append(types)
        if (self.dihedral_types.index(types), ijkl) not in self.impropers:
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

