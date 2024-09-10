import numpy
from generation.lib.polymer_system import PolymerSystem


class CrystallineSystem(PolymerSystem):
    def __init__(self):
        # Add shared attributes from PolymerSystem
        super().__init__()

        # Simulation box dimensions
        self.box = numpy.zeros((3, 3))
        # Mass of different atom types (g/cm^3)
        self._atom_masses = dict(C=12.0112, H=1.00797)
        # Bond length between the atoms in the system
        self._bond_length = dict(CC=1.54, CH=1.10)
        # List of atom charges for each type in COMPASS and PCFF (Maple, 1994)
        # C1: Chiral Carbon (cC), C2: Achiral Carbon (aC), C3: Methyl Carbon, H : Hydrogen
        self._charge_types = {'C1': -0.053, 'C2': -0.106, 'C3': -0.159, 'H': 0.053}
        # Atom ID counter
        self._atom_id   = 0

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

