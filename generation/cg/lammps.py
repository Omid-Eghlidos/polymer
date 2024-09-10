
def write_data(system):
    """ Writes an input file to LAMMPS data format, see:
        http://lammps.sandia.gov/doc/read_data.html """
    with open(system.output, 'w') as f:
        f.write("LAMMPS data file generated by bead-spring-generator.\n\n")

        f.write(f'{len(system.beads)} atoms\n')
        f.write(f'{len(system.bead_types)} atom types\n')
        if system.bonds:
            f.write(f'{len(system.bonds)} bonds\n')
            f.write(f'{len(system.bond_types)} bond types\n')
        if system.angles:
            f.write(f'{len(system.angles)} angles\n')
            f.write(f'{len(system.angle_types)} angle types\n')
        if system.dihedrals:
            f.write(f'{len(system.dihedrals)} dihedrals\n')
            f.write(f'{len(system.dihedral_types)} dihedral types\n')
        if system.impropers:
            f.write(f'{len(system.impropers)} impropers\n')
            f.write(f'{len(system.improper_types)} improper types\n')

        # Simulation box dimension
        f.write(f'\n{0.0:9.6f} {system.box[0,0]:9.6f} xlo xhi\n')
        f.write(f'{0.0:9.6f} {system.box[1,1]:9.6f} ylo yhi\n')
        f.write(f'{0.0:9.6f} {system.box[2,2]:9.6f} zlo zhi\n\n')

        f.write('Masses\n\n')
        for t, b in system.bead_types.items():
            f.write(f'{b[0]} {b[1]:9.6f} # {t}\n')

        f.write('\nAtoms # molecular\n\n')
        for i, b in enumerate(system.beads):
            f.write(f'{i+1} {b[0]} {b[1]} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n')

        if system.bonds:
            f.write('\nBonds\n\n')
            for i, b in enumerate(system.bonds):
                f.write(f'{i+1} {b[0]} {b[1]+1} {b[2]+1}\n')

        if system.angles:
            f.write('\nAngles\n\n')
            for i, a in enumerate(system.angles):
                f.write(f'{i+1} {a[0]} {a[1]+1} {a[2]+1} {a[3]+1}\n')

        if system.dihedrals:
            f.write('\nDihedrals\n\n')
            for i, d in enumerate(system.dihedrals):
                f.write(f'{i+1} {d[0]} {d[1]+1} {d[2]+1} {d[3]+1} {d[4]+1}\n')

        if system.impropers:
            f.write('\nImpropers\n\n')
            for j, i in enumerate(system.impropers):
                f.write(f'{j+1} {i[0]} {i[1]+1} {i[2]+1} {i[3]+1} {i[4]+1}\n')

        f.write('\n')
