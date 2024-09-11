#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generation.generate import generate


def main(Ns=1, material='PE', phase='Amorphous', resolution='AA', lammps=True):
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg.upper() in ['PE', 'Pe', 'pe', 'PP', 'Pp', 'pp']:
                material = arg.upper()
            elif arg.capitalize() in ['Amorphous', 'Crystalline']:
                phase = arg.capitalize()
            elif arg.upper() in ['AA', 'Aa', 'aa', 'CG', 'Cg', 'cg']:
                resolution = arg.upper()

    print(f'# Generating {Ns} {resolution} {phase} {material} system(s) ...')
    if resolution == 'AA':
        system = generate_all_atomistic_system(Ns, material, phase)
    elif resolution == 'CG':
        system = generate_coarse_grained_system(Ns, material)

    if lammps:
        with open(f'equilibrate.{resolution.lower()}.ini', 'w') as fid:
            fid.write(lammps_equilibrium_script[resolution.upper()].format(
                      output=system.output, ts=1, dt_nve=2000, dt_npt=2000))


def generate_all_atomistic_system(Ns, material, phase):
    settings = {}
    settings['Amorphous'] = {'Ns': Ns, 'Nc': 10, 'Nm': 90, 'potential_coeffs': 'compass'}
    settings['Crystalline'] = {'Ns': 1, 'Na': 3, 'Nb': 3, 'Nc': 3,
                               'potential_coeffs': 'compass', 'modification': 'alpha'}

    system = generate(material, phase, 'AA', settings[phase])
    return system


def generate_coarse_grained_system(Ns, material, phase='Amorphous'):
    settings = {}
    settings['Amorphous'] = {'rho': 0.799, 'Ns': Ns, 'Nc': 50, 'Nm': 500, 'monomer': 'A',
					 		 'beads': {'A': [1, 28.0538]}, 'bonds': {'AA': [1, 2.57]},
						 	 'angles': {'AAA': [1, 175.0]}}

    system = generate(material, phase, 'CG', settings[phase])
    return system


lammps_equilibrium_script = {}
lammps_equilibrium_script['AA'] = """
############################# Initialization ##################################
units           real
atom_style      full
boundary        p p p

#-------------------------- Forcefield Parameters -----------------------------
pair_style      lj/class2/coul/long 10.0
pair_modify     tail yes
bond_style      class2
angle_style     class2
dihedral_style	class2
improper_style	class2

#---------------------------- System Input ------------------------------------
read_data       {output}

#------------------------------ Settings --------------------------------------
kspace_style    pppm 5.0e-6
neighbor        3.0 bin
neigh_modify    delay 0 every 1 check yes
special_bonds   lj/coul 0.0 0.0 1.0

############################### Simulations ###################################
#-------------------------- Simulation Settings -------------------------------
# Variables definition
variable        ts equal {ts}

thermo_style    custom step temp press vol density pe ke etotal
thermo          100
timestep        $(v_ts)
velocity        all create 300 12345

#------------------------------ Equilibrate -----------------------------------
# Remove high-energy states
reset_timestep 	0
fix             1 all nve/limit 1.0
fix             2 all temp/rescale 1 300 300 1 1.0
run             {dt_nve}
unfix           1
unfix           2

# Equilibrate the system at the specified temperature
timestep        $(v_ts/4)
reset_timestep 	0
fix             1 all npt temp 300 300 100 aniso 1 1 1000
fix             2 all momentum 1 linear 1 1 1 angular
#dump			1 all custom 1 sample.lammpstrj id mol type x y z
run             {dt_npt}
unfix           1
unfix           2

#---------------------------- System Output -----------------------------------
write_data      {output}.data

################################# All Done ####################################
"""


lammps_equilibrium_script['CG'] = """
############################# Initialization ##################################
#------------------------------ Settings --------------------------------------
units           real
atom_style      molecular
boundary        p p p
neighbor        3.0 bin
neigh_modify    delay 0 every 1 check yes
special_bonds   lj/coul 0.0 0.0 1.0

#---------------------------- System Input ------------------------------------
read_data       {output} nocoeff

#-------------------------- Forcefield Parameters -----------------------------
pair_style 	    table linear 1000
bond_style 	    table linear 1000
angle_style     table linear 1000

pair_coeff 	    1 1 potentials/pair.table.AA   AA
bond_coeff 	    1   potentials/bond.table.AA   AA
angle_coeff     1   potentials/angle.table.AAA AAA

############################### Simulations ###################################
#-------------------------- Simulation Settings -------------------------------
# Variables definition
variable        ts equal {ts}

thermo_style    custom step temp press vol density pe ke etotal
thermo          100
timestep        $(v_ts)
velocity        all create 300 12345

#------------------------------ Equilibrate -----------------------------------
# Remove high-energy states
reset_timestep 	0
fix             1 all nve/limit 2.0
fix             2 all temp/rescale 1 300 300 5.0 1.0
run             {dt_nve}
unfix           1
unfix           2

# Equilibrate the system at the specified temperature
reset_timestep 	0
fix             1 all npt temp 300 300 100 aniso 1 1 1000
fix             2 all momentum 1 linear 1 1 1 angular
#dump			1 all custom 1 sample.lammpstrj id mol type x y z
run             {dt_npt}
unfix           1
unfix           2

#---------------------------- System Output -----------------------------------
write_data      {output}.data

################################# All Done ####################################
"""


if __name__ == '__main__':
    main()

