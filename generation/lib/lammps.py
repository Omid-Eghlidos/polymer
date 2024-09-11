"""
lammps.py
=========

This module provides functions for writing LAMMPS format data files.

Functions
---------
- write_data_file(system): Writes the system to a LAMMPS format data file.
"""


def write_data_file(system):
    """
    Writes the system to a LAMMPS format data file.

    Parameters:
    -----------
    system : object
        An object containing the system's atom coordinates, types, bonds, angles,
        dihedrals, impropers, and other relevant information.
    """
    f = open(system.output, 'w')
    f.write('LAMMPS data file generated by polymer.generation library.\n\n')
    f.write(f'{len(system.atoms):<6d} atoms\n')
    f.write(f'{max(system.atom_types)+1:<6d} atom types\n')
    f.write(f'{len(system.bonds):<6d} bonds\n')
    f.write(f'{len(system.bond_types):<6d} bond types\n')
    if system.angles:
        f.write(f'{len(system.angles):<6d} angles\n')
        f.write(f'{len(system.angle_types):<6d} angle types\n')
    if system.dihedrals:
        f.write(f'{len(system.dihedrals):<6d} dihedrals\n')
        f.write(f'{len(system.dihedral_types):<6d} dihedral types\n')
    if system.impropers:
        f.write(f'{len(system.impropers):<6d} impropers\n')
        f.write(f'{len(system.improper_types):<6d} improper types\n')

    f.write(f'\n{0.0:^9.6f} {system.box[0,0]:^9.6f} xlo xhi\n')
    f.write(f'{0.0:^9.6f} {system.box[1,1]:^9.6f} ylo yhi\n')
    f.write(f'{0.0:^9.6f} {system.box[2,2]:^9.6f} zlo zhi\n')
    if any([system.box[0,j] != 0 for j in range(1,3)]):
        xy, xz, yz = system.box[0,1], system.box[0,2], system.box[1,2]
        f.write(f'{xy:^9.6f} {xz:^9.6f} {yz:^9.6f} xy xz yz\n')
    f.write('\n')

    f.write('Masses\n\n')
    if system.resolution == 'AA':
        for i, at in enumerate(system._forcefield_types):
            atom_type = f'# c{int(at[1])}\n' if at[0] == 'C' else f'# hc\n'
            m = system._atom_masses['C'] if at[0] == 'C' else system._atom_masses['H']
            f.write(f'{i+1:<2d} {m:>9.6f} {atom_type}')
    else:
        for t, b in system.atom_types.items():
            f.write(f'{t+1:<2d} {b[1]:>9.6f} # {b[0]}\n')

    if hasattr(system, 'coeffs'):
        f.write(system.coeffs)

    f.write(f'\nAtoms # {system.atoms_format}\n\n')
    for i, a in enumerate(system.atoms):
        if system.atoms_format == 'full':
            q = system.atom_charges[i]
            f.write(f'{i+1:<6d} {a[0]+1:<6d} {a[1]+1:<6d} {q:>9.6f} '
                    f'{a[2][0]:>9.6f} {a[2][1]:>9.6f} {a[2][2]:>9.6f}\n')
        else:
            f.write(f'{i+1:<6d} {a[0]+1:<6d} {a[1]+1:<6d} '
                    f'{a[2][0]:>9.6f} {a[2][1]:>9.6f} {a[2][2]:>9.6f}\n')

    f.write('\nBonds\n\n')
    for i, b in enumerate(system.bonds):
        f.write(f'{i+1:<6d} {b[0]+1:<6d} {b[1][0]+1:<6d} {b[1][1]+1:<6d}\n')

    if system.angles:
        f.write('\nAngles\n\n')
        for i, a in enumerate(system.angles):
            f.write(f'{i+1:<6d} {a[0]+1:<6d} '
                    f'{a[1][0]+1:<6d} {a[1][1]+1:<6d} {a[1][2]+1:<6d}\n')

    if system.dihedrals:
        f.write('\nDihedrals\n\n')
        for i, d in enumerate(system.dihedrals):
            f.write(f'{i+1:<6d} {d[0]+1:<6d} '
                    f'{d[1][0]+1:<6d} {d[1][1]+1:<6d} {d[1][2]+1:<6d} {d[1][3]+1:<6d}\n')

    if system.impropers:
        f.write('\nImpropers\n\n')
        for i, d in enumerate(system.impropers):
            f.write(f'{i+1:<6d} {d[0]+1:<6d} '
                    f'{d[1][0]+1:<6d} {d[1][1]+1:<6d} {d[1][2]+1:<6d} {d[1][3]+1:<6d}\n')

    f.write('\n')

