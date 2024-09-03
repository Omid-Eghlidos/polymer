#!/usr/bin/env python3

import numpy
import system
import sys
import argparse


def main():
    """ Read the lattice parameters from the command line and generate the system """
    # Read and parse the lattice parameters from the command line
    parser = argparse.ArgumentParser(description='Build beta iPP crystal.')
    parser.add_argument('output_name', default='beta', nargs='?',
                        help='name to use for output files')
    parser.add_argument('--cells', metavar=('a','b','c'), default=[2,2,2], type=int,
                        nargs=3, help='number unit cells along each direction')
    parser.add_argument('--finite', action='store_true',
                        help='generate crystal with ends at domain boundary')
    opts = parser.parse_args()

    # Generate the PE crystal system with the given size
    print('\nGenerating a {} by {} by {} cell beta-iPP crystal ...\n'.format(*opts.cells))
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
    output_name = 'ipp_beta_a{}b{}c{}'.format(opts.cells[0], opts.cells[1], opts.cells[2])
    system.write_data_file(s, output_name + '.lammps')
    system.write_dump_file(s, output_name + '.lammpstrj')


def carbon_chain_creator(CC,i):
    ''' Creates a carbon chain of beta_ipp unit_cell, D.R.Ferro 1998'''
    Helix_center = numpy.array([[0.0    , 0.0    , 0.0],
                                [1.0/3.0, 2.0/3.0, 0.0],
                                [2.0/3.0, 1.0/3.0, 0.0]])
    C = numpy.zeros([9,3])
    C[0:3,:] = CC
    C[0:3,:] = C[0:3,:] - Helix_center[i,:]
    C[3:6,0] = -C[0:3,1]
    C[3:6,1] =  C[0:3,0] - C[0:3,1]
    C[3:6,2] =  C[0:3,2] + 1/3
    C[6:9,0] = -C[0:3,0] + C[0:3,1]
    C[6:9,1] = -C[0:3,0]
    C[6:9,2] =  C[0:3,2] + 2/3
    C[0:3,:] = C[0:3,:] + Helix_center[i,:]
    C[3:6,:] = C[3:6,:] + Helix_center[i,:]
    C[6:9,:] = C[6:9,:] + Helix_center[i,:]
    return C


def generate_carbon_atoms(finite, a, b, c):
    s = system.system()
    bond_offsets = [1, 1, 2, 1, 1, 2, 1, 1]
    chains = [numpy.vstack([chain + [0,0,k] for k in range(c)])
              for chain in space_group()]
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
    gamma = 120.0 * numpy.pi / 180.0
    ''' Columns of unit_cell are the fractional coordinate (a, b, and c) '''
    unit_cell = [[11.03, 11.03*numpy.cos(gamma), 0.00],
                 [0.00 , 11.03*numpy.sin(gamma), 0.00],
                 [0.00 ,  0.00 , 6.50]]
    return numpy.array(unit_cell)


def space_group():
    ''' Adding chains in a unitcell using symmetry operations, coordinates
        from D.L. Dorset 1998-iPP, beta-phase: A study in frustration
        - beta unit cell has three chains A, B, and C '''
    CCA = numpy.array([[0.2311 , 0.1785, 0.5951],
                       [0.0823 , 0.0772, 0.6740],
                       [0.0696 , 0.0692, 0.9104]])
    C1 = carbon_chain_creator(CCA,0)
    CCB = numpy.array([[0.5426, 0.6813, 0.4169],
                       [0.4199, 0.6977, 0.4958],
                       [0.4169, 0.6991, 0.7328]])
    C2 = carbon_chain_creator(CCB,1)
    CCC = numpy.array([[0.8910, 0.4606, 0.6334],
                       [0.7410, 0.4004, 0.7168],
                       [0.7444, 0.4040, 0.9538]])
    C3 = carbon_chain_creator(CCC,2)
    return C1, C2, C3


if __name__ == '__main__':
    main()
