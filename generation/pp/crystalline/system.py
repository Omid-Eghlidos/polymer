import numpy
from numpy.linalg import norm
from numpy import sin, cos, cross, dot

class system:
    def __init__(self):
        # Store the simulation box dimension
        self.box = numpy.identity(3)
        # Stores atoms properties (list of [id, mol, type, x, y, z]).
        self.atoms = []
        # List of atom types (element) for each atom.
        self.atom_types = []
        # List of atom charges for each type (1998 - Sun - COMPASS)
        self.charge_types = {'c1':-0.053, 'c2':-0.106, 'c3':-0.159, 'h': 0.053}
        # List of atom charges for each type
        self.charges = []
        # List of forcefield type (e.g. 'C3') for each atom.
        self.forcefield_types = []
         # Store the molecule that each atom belongs to
        self.molecule = []
        # Stores bonds (list of pairs of zero-indexed atom ids).
        self.bonds = []
        # Possible bond types in iPP (CH == HC)
        self.bond_types = {'CC' : 1, 'CH' : 2}
        # Stores angles (list of triplets).
        self.angles = []
        # Store unique angle types
        self.angle_types = []
        # Stores dihedrals (list of quartets).
        self.dihedrals = []
        # Store unique dihedral types
        self.dihedral_types = []
        # Stores impropers (list of quartets).
        self.impropers = []
        # Store unique improper types
        self.improper_types = []


    def add_atom(self, x, atom_type, molecule):
        ''' Helper function for adding atoms to the system. '''
        self.atoms.append(x)
        self.atom_types.append(atom_type)
        # Initially forcefield type will be the same as atom type.
        self.forcefield_types.append(atom_type.lower())
        self.charges.append(self.charge_types['h'])
        self.molecule.append(molecule)


    def add_hydrogen_atoms(self):
        ''' Adds hydrogen atoms to the crystalline structure. '''
        for i in range(len(self.atoms)):
            bonds = [b[:] for b in self.bonds if i in b]

            ''' Sorts each bond so that atom i is first, this way
            outward bond vectors from atom i can be computed as:
            dx = self.atoms[b[1]] - self.atoms[b[0]] '''
            for b in bonds:
                b.sort(key = lambda x: x != i)
            # Bond vectors from atom i to atom in each bond.
            bv = [self.bond_vector(*b) for b in bonds]
            if len(bonds) == 1:
                self.forcefield_types[i] = 'c3'
                self.charges[i] = self.charge_types['c3']
                self.add_methyl_carbons(i, bv)
            elif len(bonds) == 2:
                self.forcefield_types[i] = 'c2'
                self.charges[i] = self.charge_types['c2']
                self.add_achiral_carbons(i, bv)
            elif len(bonds) == 3:
                self.forcefield_types[i] = 'c1'
                self.charges[i] = self.charge_types['c1']
                self.add_chiral_carbon(i, bv)


    def add_methyl_carbons(self, i, bv):
        # Gets index of carbon adjacent to methyl carbon, i.
        j = next(b for b in self.bonds if i in b)[1]
        bonds = [b[:] for b in self.bonds if j in b and i not in b]
        for b in bonds:
            b.sort(key = lambda x: x != j)
        if len(bonds) == 2:
            # Bond from C_i-1 to C_i+1 (see Fig 1a in 1998 Theodorou paper).
            rcc = self.bond_vector(bonds[0][1], bonds[1][1])
        else:
            # For finite chains, we can't use skeleton vector at ends.
            rcc = [0,0,1]
		# u,v,w make a coordinate system about the methyl carbon.
        u = -bv[0] / norm(bv[0])
        v = cross(u, rcc) / norm(cross(u, rcc))
        w = cross(u, v)

        phi = (109.0 - 90.0)/180.0 * numpy.pi
        for theta in [0.0, 2.0/3.0*numpy.pi, 4.0/3.0*numpy.pi]:
            theta += numpy.pi/6.0
            r = sin(phi)*u + cos(phi)*(cos(theta)*v + sin(theta)*w)
            self.add_atom(self.atoms[i] + 1.10*r, 'H', self.molecule[i])
        for j in range(1,4):
            self.bonds.append([i, len(self.atoms)-j])


    def add_achiral_carbons(self, i, bv):
        bi = -bv[0] / norm(bv[0])
        bj =  bv[1] / norm(bv[1])
        u = (bi - bj) / numpy.sqrt(2.0*(1.0-dot(bi,bj)))
        v = cross(bi, bj) / norm(cross(bi, bj))

        qh = 1.28
        h1 = self.atoms[i] + 1.10*(sin(qh/2)*u + cos(qh/2)*v)
        h2 = self.atoms[i] + 1.10*(sin(qh/2)*u - cos(qh/2)*v)
        self.add_atom(h1, 'H', self.molecule[i])
        self.add_atom(h2, 'H', self.molecule[i])
        for j in range(1,3):
            self.bonds.append([i, len(self.atoms)-j])


    def add_chiral_carbon(self, i, bv):
        # Atom C_i is connected to C_i-1 and C_i+i and Ri.
        # Vector n is perpendicular to plane containing C_i+i, C_i-1, and R.
        n = cross(bv[1] - bv[0], bv[2] - bv[0])
        n /= norm(n)
        # Vector n should point away from average of other
        # bond vectors.
        if dot(n, sum(bv)) > 0:
            n *= -1

        self.add_atom(self.atoms[i] + 1.10*n, 'H', self.molecule[i])
        self.bonds.append([i, len(self.atoms)-1])


    def add_angles(self):
        ''' Traverses the bond list and defines all bond angles. '''
        for j in range(len(self.atoms)):
            # Set J contains all atoms bonded to atom j.
            J = self.atoms_bonded_to(j)
            for i in sorted(J):
                for k in sorted(J):
                    # Don't add angle twice.
                    if i < k:
                        self.add_angle([i, j, k])


    def add_angle(self, ijk):
        ''' Helper function for adding angles to the system. '''
        types = [self.atom_types[i] for i in ijk]
        # Accounts for ABC -> CBA symmetry of angle types.
        if types[0] > types[2]:
            types = types[::-1]
            ijk = ijk[::-1]
        if types not in self.angle_types:
            self.angle_types.append(types)
        self.angles.append((self.angle_types.index(types), ijk))


    def add_dihedrals(self):
        for b in self.bonds:
            j,k = sorted(b)
            J = [i for i in self.atoms_bonded_to(b[0]) if i != b[1]]
            K = [i for i in self.atoms_bonded_to(b[1]) if i != b[0]]
            for i in J:
                for l in K:
                    self.add_dihedral([i,j,k,l])


    def add_dihedral(self, ijkl):
        ''' Helper function for adding dihedrals to the system. '''
        types = [self.atom_types[i] for i in ijkl]
        # Possible dihedral types
        # HCCH <=> HCCH
        # HCCC <=> CCCH
        # CCCC <=> CCCC
        # Note: this is not very general.
        if types[0] > types[-1]:
            ijkl = ijkl[::-1]
            types = types[::-1]
        if types not in self.dihedral_types:
            self.dihedral_types.append(types)
        self.dihedrals.append((self.dihedral_types.index(types), ijkl))


    def add_impropers(self):
        ''' Traverse the bond list and defines all improper angles. '''
        for b in self.bonds:
            j = sorted(b)[0]
            bonded_to_j = sorted(self.atoms_bonded_to(j))
            # The atom needs to have at least three bonds to have an improper
            if len(bonded_to_j) > 3:
                for i in bonded_to_j:
                    for k in bonded_to_j:
                        for l in bonded_to_j:
                            if i < k and i < l and l < k:
                                self.add_improper([i,j,k,l])
        self.impropers = sorted(self.impropers, key = lambda x: x[1])


    def add_improper(self, ijkl):
        ''' Helper function for adding impropers to the system '''
        types = sorted([self.atom_types[i] for i in ijkl])
        # Possible improper types
        # HCCC <=> CCCH
        # CCHH <=> HHCC
        if types not in self.improper_types:
            self.improper_types.append(types)
        if (self.improper_types.index(types), ijkl) not in self.impropers:
            self.impropers.append((self.improper_types.index(types), ijkl))


    def atoms_bonded_to(self, i):
        ''' Returns a list of all atoms bonded to atom i. '''
        return [next(a for a in b if a != i) for b in self.bonds if i in b]


    def bond_vector(self, i, j):
        ''' Returns the bond vector from atom i to atom j (uses the nearest
        periodic image of atom j so that bond length is correct). '''
        return self.wrap_vector(self.atoms[j] - self.atoms[i])


    def wrap_vector(self, v):
        ''' Returns the shortest vector v + i*A + j*B + k*C for any period
        image i, j, and k '''
        s = numpy.linalg.solve(self.box.T, v)
        return v - dot(numpy.round(s), self.box.T)


def write_dump_file(s, path):
    ''' Writes the system to a LAMMPS dump file '''
    f = open(path, 'w')
    f.write('ITEM: TIMESTEP\n0\n')
    f.write('ITEM: NUMBER OF ATOMS\n{}\n'.format(len(s.atoms)))
    # See https://lammps.sandia.gov/doc/Howto_triclinic.html
    f.write('ITEM: BOX BOUNDS xy xz yz xx yy zz\n')

    xlo, xhi = 0.0, s.box[0,0]
    ylo, yhi = 0.0, s.box[1,1]
    zlo, zhi = 0.0, s.box[2,2]
    xy, xz, yz = s.box[0,1], s.box[0,2], s.box[1,2]

    xlo_bound = xlo + min(0.0, xy, xz, xy+xz)
    xhi_bound = xhi + max(0.0, xy, xz, xy+xz)
    ylo_bound = ylo + min(0.0, yz)
    yhi_bound = yhi + max(0.0, yz)

    f.write('{:6.3f} {:6.3f} {:6.3f}\n'.format(xlo_bound, xhi_bound, xy))
    f.write('{:6.3f} {:6.3f} {:6.3f}\n'.format(ylo_bound, yhi_bound, xz))
    f.write('{:6.3f} {:6.3f} {:6.3f}\n'.format(zlo, zhi, yz))

    f.write('ITEM: ATOMS id type mol x y z\n')
    for i,x in enumerate(s.atoms):
        t = 1 if s.atom_types[i] == 'C' else 2
        f.write('{:6d} {:4d} {:4d} {:8.5f} {:8.5f} {:8.5f}\n'.format(
                i+1, t, s.molecule[i], *x))

def write_data_file(s, path):
    ''' Writes the system to a LAMMPS data file. '''
    f = open(path, 'w')
    f.write('LAMMPS data file generated by ipp-crystal-generator\n\n')
    lammps_atom_types = sorted(list(set(s.forcefield_types)))
    f.write('{:6d} atoms\n'.format(len(s.atoms)))
    f.write('{:6d} atom types\n'.format(len(lammps_atom_types)))
    f.write('{:6d} bonds\n'.format(len(s.bonds)))
    f.write('{:6d} bond types\n'.format(2))
    f.write('{:6d} angles\n'.format(len(s.angles)))
    f.write('{:6d} angle types\n'.format(len(s.angle_types)))
    f.write('{:6d} dihedrals\n'.format(len(s.dihedrals)))
    f.write('{:6d} dihedral types\n'.format(len(s.dihedral_types)))
    f.write('{:6d} impropers\n'.format(len(s.impropers)))
    f.write('{:6d} improper types\n'.format(len(s.improper_types)))

    xlo, xhi = 0.0, s.box[0,0]
    ylo, yhi = 0.0, s.box[1,1]
    zlo, zhi = 0.0, s.box[2,2]
    xy, xz, yz = s.box[0,1], s.box[0,2], s.box[1,2]

    f.write('\n{:10.5f} {:10.5f} xlo xhi\n'.format(xlo, xhi))
    f.write('{:10.5f} {:10.5f} ylo yhi\n'.format(ylo, yhi))
    f.write('{:10.5f} {:10.5f} zlo zhi\n'.format(zlo, zhi))
    f.write('{:10.5f} {:10.5f} {:10.5f} xy xz yz\n'.format(xy, xz, yz))

    f.write('\nMasses\n\n')
    for i,t in enumerate(lammps_atom_types):
        if t.startswith('c'):
            f.write('{:3d} {:10.5f}\t# c2\n'.format(i+1, 12.0112))
        elif t.startswith('h'):
            f.write('{:3d} {:10.5f}\t# hc\n'.format(i+1, 1.00797))
        else:
            print ('Error writing data file: unknown atom type', t)
            return None

    f.write('\nAtoms # full\n\n')
    for i,x in enumerate(s.atoms):
        t = lammps_atom_types.index(s.forcefield_types[i]) + 1
        q = s.charges[i]
        f.write('{:6d} {:6d} {:6d} {:10.5f} {:10.5f} {:10.5f} {:10.5f}\n'.format(
                i+1, s.molecule[i], t, q, *x))

    f.write('\nBonds\n\n')
    for i,b in enumerate(s.bonds):
        if s.atom_types[b[0]] == 'C' and s.atom_types[b[1]] == 'C':
            bond_type = 1
        else:
            bond_type = 2
        f.write('{:6d} {:4d} {:6d} {:6d}\n'.format(i+1, bond_type, b[0]+1, b[1]+1))

    f.write('\nAngles\n\n')
    for i,a in enumerate(s.angles):
        ijk = (i+1 for i in a[1])
        f.write('{:6d} {:4d} {:6d} {:6d} {:6d}\n'.format(i+1, a[0]+1, *ijk))

    f.write('\nDihedrals\n\n')
    for i,d in enumerate(s.dihedrals):
        ijkl = (i+1 for i in d[1])
        f.write('{:6d} {:4d} {:6d} {:6d} {:6d} {:6d}\n'.format(i+1, d[0]+1, *ijkl))

    f.write('\nImpropers\n\n')
    for i,im in enumerate(s.impropers):
        ijkl = (i+1 for i in im[1])
        f.write('{:6d} {:4d} {:6d} {:6d} {:6d} {:6d}\n'.format(i+1, im[0]+1, *ijkl))

    f.write('\n')

