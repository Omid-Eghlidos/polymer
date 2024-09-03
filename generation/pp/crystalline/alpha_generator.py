#!/usr/bin/env python3

import numpy
import system
import sys
import argparse


def main():
    """ Read the lattice parameters from the command line and generate the system """
    # Read and parse the lattice parameters from the command line
    parser = argparse.ArgumentParser(description='Build alpha iPP crystal.')
    parser.add_argument('output_name', default='alpha', nargs='?',
                        help='name to use for output files')
    parser.add_argument('--cells', metavar=('a','b','c'), default=[2,2,2], type=int,
                        nargs=3, help='number unit cells along each direction')
    parser.add_argument('--finite', action='store_true',
                        help='generate crystal with ends at domain boundary')
    opts = parser.parse_args()

    # Generate the PE crystal system with the given size
    print('\nGenerating a {} by {} by {} cell alpha-iPP crystal ...\n'.format(*opts.cells))
    s = generate_carbon_atoms(opts.finite, *opts.cells)
    # Add the hydrogen to the carbon atoms
    s.add_hydrogen_atoms()
    # Find the bond angles
    s.add_angles()
    # Find the dihedral angles
    s.add_dihedrals()
    # Find the improper angles
    s.add_impropers()
    # Show the generated system properties on the screen
    print('Generated {} atoms.'.format(len(s.atoms)))
    print('Generated {} atom types.'.format(2))
    print('   C, H')
    # Bonds
    print('Generated {} bonds.'.format(len(s.bonds)))
    # Angles
    print('Generated {} angles.'.format(len(s.angles)))
    print('Generated {} angle types.'.format(len(s.angle_types)))
    print('  ', ', '.join([ ''.join(at) for at in s.angle_types]))
    # Dihedral angles
    print('Generated {} dihedrals.'.format(len(s.dihedrals)))
    print('Generated {} dihedral types.'.format(len(s.dihedral_types)))
    print('  ', ', '.join([ ''.join(dt) for dt in s.dihedral_types]))
    # Improper angles
    print('Generated {} impropers.'.format(len(s.impropers)))
    print('Generated {} improper types.'.format(len(s.improper_types)))
    print('  ', ', '.join([ ''.join(it) for it in s.improper_types]))

    # Write the generated system into a file
    output_name = 'ipp_alpha_a{}b{}c{}'.format(opts.cells[0], opts.cells[1], opts.cells[2])
    system.write_data_file(s, output_name + '.lammps')
    system.write_dump_file(s, output_name + '.lammpstrj')


def generate_carbon_atoms(finite, a, b, c):
    s = system.system()
    C1 = numpy.array([[-0.0727, 0.2291, 0.2004],  #C3 0: 1
                      [-0.0765, 0.1592, 0.2788],  #C1 1: 0 2
                      [-0.1021, 0.1602, 0.5098],  #C2 2: 1 4
                      [-0.3087, 0.0589, 0.4941],  #C3 3: 4
                      [-0.1146, 0.0928, 0.6057],  #C1 4: 2 3 5
                      [-0.1044, 0.0854, 0.8428],  #C2 5: 4 7
                      [ 0.2775, 0.0797, 0.9260],  #C3 6: 7
                      [ 0.0872, 0.1156, 0.9730],  #C1 7: 5 6 8
                      [ 0.1026, 0.1221, 1.2109]]) #C2 8: 7
    bond_offsets = [1, 1, 2, 1, 1, 2, 1, 1]
    chains = [numpy.vstack([chain + [0,0,k] for k in range(c)])
                 for chain in space_group(C1)]
    mol_id = 0
    for i in range(a):
        for j in range(b):
            chains_ab = [chain + [i,j,0] for chain in chains]
            for chain in chains_ab:
                mol_id += 1
                first_id = len(s.atoms)
                for I,x in enumerate(chain):
                    next_id = len(s.atoms)
                    if I%9 < len(bond_offsets):
                        s.bonds.append([next_id, next_id + bond_offsets[I%9]])
                    else:
                        # The last atom in the unit cell is bonded either to
                        # the second atom in the next unit cell or the second
                        # atom at the beginning of the chain (due to periodic
                        # boundary conditions).
                        if not finite or next_id + 2 < len(chain):
                            bond_to = (next_id + 2) % len(chain) + first_id
                            s.bonds.append(sorted([next_id, bond_to]))
                    s.add_atom(x, 'C', mol_id)

    s.bonds = sorted(s.bonds, key = lambda x: x[0])
    s.atoms = numpy.dot(s.atoms, unit_cell().T)
    s.atoms = [a for a in s.atoms]  # convert back to list of arrays
    s.box = numpy.dot(unit_cell(), numpy.diag([a, b, c]))
    return s


def unit_cell():
    beta = 99.5 * numpy.pi / 180.0
    ''' Columns of unit_cell are the fractional coordinate (a, b, and c) '''
    unit_cell = [[6.63,  0.00, 6.50*numpy.cos(beta)],
                 [0.00, 20.78, 0.0],
                 [0.00,  0.00, 6.50*numpy.sin(beta)]]
    return numpy.array(unit_cell)


def space_group(C1):
    ''' Adding chains in a unitcell using symmetry operations '''
    C2, C3, C4 = C1.copy(), C1.copy(), C1.copy()

    C2[:,0] =  C1[:,0]
    C2[:,1] = -C1[:,1]
    C2[:,2] =  C1[:,2] + 0.5

    C3[:,0] =  C2[:,0] - 0.5
    C3[:,1] =  C2[:,1] + 0.5
    C3[:,2] =  C2[:,2]

    C4[:,0] =  C1[:,0] - 0.5
    C4[:,1] =  C1[:,1] - 0.5
    C4[:,2] =  C1[:,2]

    return C1, C2, C3, C4

if __name__ == '__main__':
    main()
