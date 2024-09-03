#!/usr/bin/env python3
"""
Goal
----
Generate crystal structure of polyethylene with given lattice parameters
in a finite or infinite chains.

Dependencies
-------------
1) system.py

Deployment
----------
Try:
    ./pe_crystal_generator.py --cell [a] [b] [c] [option]
        * [a], [b], [c]: Lattice parameters in a, b, and c directions
            - NOTE: c direction is the chain direction
        * option: --finite -> for finite chains
"""


import numpy
import system
import sys
import argparse


def main():
    """ Read the lattice parameters from the command line and generate the system """
    # Read and parse the lattice parameters from the command line
    parser = argparse.ArgumentParser(description='Build PE crystal.')
    parser.add_argument('output_name', default='alpha', nargs='?',
                        help='Name to use for output files.')
    parser.add_argument('--cells', metavar=('a','b','c'), default=[1,1,1], type=int,
                        nargs=3, help='Number unit cells along each direction.')
    parser.add_argument('--finite', action='store_true',
                        help='Generate crystal with ends at domain boundary.')
    parser.add_argument('--atom_style', default='molecular', type=str, nargs=1,
                        help='Atom style to be written into data file.')
    opts = parser.parse_args()

    # Generate the PE crystal system with the given size
    print('\nGenerating a {} by {} by {} cell PE crystal ...\n'.format(*opts.cells))
    s = generate_system(opts.finite, *opts.cells)
    # Show the generated system properties on the screen
    print('Generated {} atoms.'.format(len(s.atoms)))
    print('Generated {} atom types.'.format(2))
    print('   C, H')
    # Bonds
    print('Generated {} bonds.'.format(len(s.bonds)))
    print('Generated {} bond types.'.format(len(s.bond_types)))
    print('  ', ', '.join([ at for at in s.bond_types]))
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
    output_name = 'PE_a{}b{}c{}'.format(opts.cells[0], opts.cells[1], opts.cells[2])
    system.write_dump_file(s, output_name + '.lammpstrj')
    system.write_data_file(opts.atom_style, s, output_name + '.lammps')


def generate_system(finite, a, b, c):
    """ Generate the system with the given lattice parameters """
    s = system.system()
    # Generate chains - c direction is the chain direction
    chains = [numpy.vstack([chain + [0,0,k] for k in range(c)]) for chain in space_group()]

    # Generate the atoms and find the bonds
    # First C atom is connected to the third atom in each chain and consequent Hs
    bond_offsets = [3, -1, -2, 3, -1, -2]
    mol_id = 0
    for i in range(a):
        for j in range(b):
            chains_ab = [chain + [i,j,0] for chain in chains]
            for chain in chains_ab:
                mol_id += 1
                first_id = len(s.atoms)
                for I,x in enumerate(chain):
                    next_id = len(s.atoms)
                    bonded_to = next_id + bond_offsets[I%6]
                    # Find bonds
                    if bonded_to < len(chain) * mol_id:
                        s.bonds.append(sorted([next_id, bonded_to]))
                    else:
                        # The last atom in the unit cell is bonded either to
                        # the first atom in the next unit cell or the first
                        # atom at the beginning of the chain (due to periodic
                        # boundary conditions).
                        if not finite:
                            bonded_to = bonded_to % len(chain) + first_id
                            s.bonds.append([next_id, bonded_to])
                    if I%6 in [0, 3]:
                        s.add_atom(x, 'C', mol_id)
                    else:
                        s.add_atom(x, 'H', mol_id)
    # Sort the bonds based on the first atom id
    s.bonds = sorted(s.bonds, key = lambda x: x[0])
    # Map the atoms into the PBC
    s.atoms = numpy.dot(s.atoms, unit_cell())
    s.atoms = [a for a in s.atoms]  # convert back to list of arrays
    # Find the xhi-, yhi-, zhi-bound of the simulation box assuming low ones all zero
    s.box = numpy.dot(unit_cell(), numpy.diag([a, b, c]))
    # Find the angles
    s.add_angles()
    # Find the dihedral angles
    s.add_dihedrals()
    # Find the improper angles
    s.add_impropers()
    return s


def unit_cell():
    ''' Columns of unit_cell are the fractional coordinate (a, b, and c).
    Refer to 1975 - Avitabile - Low Temperature Crystal Structure of PE '''
    # Orthorhombic unit cell of PE
    unit_cell = [[7.121, 0.0, 0.0],
                 [0.0, 4.851, 0.0],
                 [0.0,  0.0, 2.548]]
    return numpy.array(unit_cell)


def space_group():
    ''' Adding chains in a unitcell using symmetry operations '''
    # For unit cell parameters read Section II. of:
    # 1998 - Bruno et. al. - Thermal expansion of polymers: Mechanisms in orthorhombic PE
    # Internal coordinates for C atoms
    w1, w2 = 0.044183, 0.059859
    # Internal coordinates for H atoms
    w3, w4, w5, w6 = 0.186360, 0.011509, 0.027079, 0.277646
    # For Space Group Pnam symmetry operators (as given by 1939 - Bunn - The Crystal Structure)
    # First chain: x, -y, 0.25; -x, y, -0.25;
    C1 = numpy.array([[      w1,     -w2,  0.25],  #C2
                      [      w3,     -w4,  0.25],  #H
                      [      w5,     -w6,  0.25],  #H
                      [     -w1,      w2, -0.25],  #C2
                      [     -w3,      w4, -0.25],  #H
                      [     -w5,      w6, -0.25]]) #H
    # Second chain: 0.5-x, 0.5-y, -0.25; 0.5+x, 0.5+y, 0.25
    C2 = numpy.array([[  0.5-w1,  0.5-w2, -0.25],  #C2
                      [  0.5-w3,  0.5-w4, -0.25],  #H
                      [  0.5-w5,  0.5-w6, -0.25],  #H
                      [  0.5+w1,  0.5+w2,  0.25],  #C2
                      [  0.5+w3,  0.5+w4,  0.25],  #H
                      [  0.5+w5,  0.5+w6,  0.25]]) #H

    return C1, C2

if __name__ == '__main__':
    main()
