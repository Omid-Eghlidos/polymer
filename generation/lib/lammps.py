"""
lammps.py
=========

This module provides functions for writing LAMMPS format data files.

Functions
---------
- write_data_file(path, system, output_format, coeffs): Writes the system to a LAMMPS format data file.
"""


def write_data_file(path, system, output_format, coeffs):
    """
    Writes the system to a LAMMPS format data file.

    Parameters
    ----------
    path : str
        The file path where the LAMMPS data file will be written.
    output_format : str
    	Type of Atoms description writte into file (full or molecular)
    system : object
        An object containing the system's atom coordinates, types, bonds, angles,
        dihedrals, impropers, and other relevant information.
    coeffs : str or None
        Coefficient data to be written in the file.
    """
    f = open(path, 'w')
    f.write('LAMMPS data file generated by polymer.generation library.\n\n')
    f.write(f'{len(system.atom_coords):<4d} atoms\n')
    f.write(f'{max(system.atom_types)+1:<4d} atom types\n')
    f.write(f'{len(system.bonds):<4d} bonds\n')
    f.write(f'{len(system.unique_bond_types):<4d} bond types\n')
    if system.angles:
        f.write(f'{len(system.angles):<4d} angles\n')
        f.write(f'{len(system.angle_types):<4d} angle types\n')
    if system.dihedrals:
        f.write(f'{len(system.dihedrals):<4d} dihedrals\n')
        f.write(f'{len(system.dihedral_types):<4d} dihedral types\n')
    if system.impropers:
        f.write(f'{len(system.impropers):<4d} impropers\n')
        f.write(f'{len(system.improper_types):<4d} improper types\n')

    f.write(f'\n{0.0:^9.6f} {system.box[0,0]:^9.6f} xlo xhi\n')
    f.write(f'{0.0:^9.6f} {system.box[1,1]:^9.6f} ylo yhi\n')
    f.write(f'{0.0:^9.6f} {system.box[2,2]:^9.6f} zlo zhi\n')

    f.write('\nMasses\n\n')
    for i, at in enumerate(system._forcefield_atom_types):
        atom_type = f'# c{int(at[1])}\n' if at[0] == 'C' else f'# hc\n'
        m = system._atom_masses['C'] if at[0] == 'C' else system._atom_masses['H']
        f.write(f'{i+1:<2d} {m:>9.6f} ' + atom_type)

    if coeffs:
        f.write(coeffs)

    f.write('\nAtoms # full\n\n')
    for i, x in enumerate(system.atom_coords):
        c = system.atom_chains[i] + 1
        t = system.atom_types[i] + 1
        q = system.atom_charges[i]
        if output_format == 'full':
            f.write(f'{i+1:<4d} {c:<4d} {t:<4d} {q:>9.6f}'
                    f' {x[0]:>9.6f} {x[1]:>9.6f} {x[2]:>9.6f}\n')
        else:
            f.write(f'{i+1:<4d} {c:<4d} {t:<4d} {x[0]:>9.6f} {x[1]:>9.6f} {x[2]:>9.6f}\n')

    f.write('\nBonds\n\n')
    for i, b in enumerate(system.bonds):
        f.write(f'{i+1:<4d} {system.bond_types[i]+1:<4d} {b[0]+1:<4d} {b[1]+1:<4d}\n')

    if system.angles:
        f.write('\nAngles\n\n')
        for i, a in enumerate(system.angles):
            f.write(f'{i+1:<4d} {a[0]+1:<4d} '
                    f'{a[1][0]+1:<4d} {a[1][1]+1:<4d} {a[1][2]+1:<4d}\n')

    if system.dihedrals:
        f.write('\nDihedrals\n\n')
        for i, d in enumerate(system.dihedrals):
            f.write(f'{i+1:<4d} {d[0]+1:<4d} '
                    f'{d[1][0]+1:<4d} {d[1][1]+1:<4d} {d[1][2]+1:<4d} {d[1][3]+1:<4d}\n')

    if system.impropers:
        f.write('\nImpropers\n\n')
        for i, d in enumerate(system.impropers):
            f.write(f'{i+1:<4d} {d[0]+1:<4d} '
                    f'{d[1][0]+1:<4d} {d[1][1]+1:<4d} {d[1][2]+1:<4d} {d[1][3]+1:<4d}\n')
