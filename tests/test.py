#!/usr/bin/env python3
"""
Test Script
-----------
This script provide a simple test for using the Polymer library for generating
various polymeric materials.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generation.generate import generate


def main(Ns=1, material='PE', phase='Amorphous', resolution='AA', lammps=True, verbose=True):
    """
    Read general settings from the command line and generate the desired system(s).
    """
    # Read the settings from the command line if given
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg.upper() in ['PE', 'pe', 'PP', 'pp', 'PU', 'pu']:
                material = arg.upper()
            elif arg.capitalize() in ['Amorphous', 'Crystalline']:
                phase = arg.capitalize()
            elif arg.upper() in ['AA', 'Aa', 'aa', 'CG', 'Cg', 'cg']:
                resolution = arg.upper()
            else:
                try:
                    Ns = int(arg)
                except:
                    Ns = 1

    settings = adjust_settings(Ns, resolution, phase)
    # Generate Ns systems with the settings
    system = generate(material, phase, resolution, settings)
    # Generate LAMMPS equilibrium script for the generated system if required
    if lammps:
        lammps_input_script(Ns, resolution, material, phase, settings)


def adjust_settings(Ns, resolution, phase):
    """
    Adjust the system generation settings based on the given parameters.
    """
    if resolution == 'AA' and phase == 'Amorphous':
        return {'Ns': Ns, 'Nc': 10, 'Nm': 90, 'Nss': 1, 'atoms_format': 'full',
                'potential_coeffs': 'pcff'}
    if resolution == 'AA' and phase == 'Crystalline':
        return {'Ns': 1, 'Na': 1, 'Nb': 1, 'Nc': 1, 'modification': 'beta',
        		'potential_coeffs': 'compass'}
    if resolution == 'CG' and phase == 'Amorphous':
        return {'rho': 0.799, 'Ns': Ns, 'Nc': 50, 'Nm': 500, 'monomer': 'A',
			    'beads': {'A': [1, 28.0538]}, 'bonds': {'AA': [1, 2.57]},
				'angles': {'AAA': [1, 175.0]}}


def lammps_input_script(Ns, resolution, material, phase, settings):
    """
    Generate a LAMMPS input script for equilibrating the system.
    """
    print(f'# Generating {Ns} LAMMPS input script(s) ...')
    # Create the system name and adjust the LAMMPS equilibrium simulation settings
    system = f'{resolution.lower()}_{material}_'
    if phase == 'Amorphous':
        system += f'{settings["Nm"]}m{settings["Nc"]}c'
        if resolution == 'AA':
            simulations = {'nve_dt': 1e4, 'nve_xmax': 1.0, 'npt_dt': 1e4}
        else:
            simulations = {'nve_dt': 5e3, 'nve_xmax': 1.0, 'npt_dt': 10e6}
    if phase == 'Crystalline':
        if material == 'PP':
            system += f'{settings["modification"]}_'
        system += f'a{settings["Na"]}b{settings["Nb"]}c{settings["Nc"]}'
        if resolution == 'AA':
            simulations = {'nve_dt': 5e3, 'nve_xmax': 1.0, 'npt_dt': 5e5}

    for ns in range(Ns):
        system += f'_{ns+1:02d}'
        script_name = f'{system}.equilibrate.ini'
        with open(script_name, 'w') as fid:
            script = initialization[resolution] + relaxing[phase] + equilibrium[phase] + write_data
            fid.write(script.format(system=system,
                                    nve_dt=simulations['nve_dt'],
                                    nve_xmax=simulations['nve_xmax'],
                                    npt_dt=simulations['npt_dt']))
        print(f'---- LAMMPS Input: {script_name}\n')


initialization = {}
initialization['AA'] = """
# Log all the details in the log file
log             {system}.equilibrate.log
############################# Initialization ##################################
#------------------------- Variables Definition -------------------------------
# Simulation timestep (fs)
variable 	    ts equal 1
# Simulation temperature (K)
variable        T equal 300.0

#------------------------------ Settings --------------------------------------
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
read_data       {system}.lammps

#------------------------- Forcefield Settings --------------------------------
kspace_style    pppm 5.0e-6
neighbor        3.0 bin
neigh_modify    delay 0 every 1 check yes
special_bonds   lj/coul 0.0 0.0 1.0
"""

initialization['CG'] = """
# Log all the details in the log file
log             equilibrium.log
############################# Initialization ##################################
#------------------------- Variables Definition -------------------------------
# Simulation timestep (fs)
variable 	    ts equal 5
# Simulation temperature (K)
variable        T equal 300.0

#------------------------------ Settings --------------------------------------
units           real
atom_style      molecular
boundary        p p p
neighbor        3.0 bin
neigh_modify    delay 0 every 1 check yes
special_bonds   lj/coul 0.0 0.0 1.0

#---------------------------- System Input ------------------------------------
read_data       {system}.lammps nocoeff

#-------------------------- Forcefield Parameters -----------------------------
pair_style 	    table linear 1000
bond_style 	    table linear 1000
angle_style     table linear 1000

pair_coeff 	    1 1 potentials/pair.table.AA   AA
bond_coeff 	    1   potentials/bond.table.AA   AA
angle_coeff     1   potentials/angle.table.AAA AAA
"""


relaxing = {}
relaxing['Amorphous'] = """
############################### Simulations ###################################
#------------------------------- Settings -------------------------------------
thermo_style    custom step temp press vol density pe ke etotal
thermo          1000
velocity        all create $(v_T) 12345

#------------------------------- Relaxing -------------------------------------
timestep        $(v_ts)
# Minimize energy of the system
minimize 		0.0 1.0e-8 1000 100000

# Remove high-energy states
reset_timestep 	0
fix             1 all nve/limit {nve_xmax}
fix             2 all temp/rescale 1 $(v_T) $(v_T) 1.0 1.0
run             $(floor({nve_dt}/v_ts))
unfix           1
unfix           2

# Relax the system under NpT
timestep        $(v_ts)
reset_timestep 	0
fix             1 all npt temp $(v_T) $(v_T) $(100*v_ts) iso 1.0 1.0 $(1000*v_ts)
fix             2 all momentum 1 linear 1 1 1 angular
run             $(floor({nve_dt}/v_ts))
unfix           1
unfix           2
"""

relaxing['Crystalline'] = """
############################### Simulations ###################################
#------------------------------- Settings -------------------------------------
thermo_style    custom step temp press vol density cella cellb cellc cellalpha cellbeta cellgamma ke pe etotal
thermo          1000
velocity        all create $(v_T) 12345

#------------------------------- Relaxing -------------------------------------
timestep        $(v_ts)
# Minimize energy of the system
minimize 		0.0 1.0e-8 1000 100000

# Remove high-energy states
reset_timestep 	0
fix             1 all nve/limit {nve_xmax}
fix             2 all temp/rescale 1 $(v_T) $(v_T) 1.0 1.0
run             $(floor({nve_dt}/v_ts))
unfix           1
unfix           2
"""


equilibrium = {}
equilibrium['Amorphous'] = """
#----------------------------- Equilibrating ----------------------------------
timestep        $(v_ts)
reset_timestep 	0
fix             1 all npt temp $(v_T) $(v_T) $(100*v_ts) iso 1.0 1.0 $(1000*v_ts)
fix             2 all momentum 1 linear 1 1 1 angular
run             $(floor({npt_dt}/v_ts))
unfix           1
unfix           2
"""

equilibrium['Crystalline'] = """
#----------------------------- Equilibrating ----------------------------------
timestep        $(v_ts)
reset_timestep 	0
fix             1 all npt temp $(v_T) $(v_T) $(100*v_ts) aniso 1.0 1.0 $(1000*v_ts)
fix             2 all momentum 1 linear 1 1 1 angular
run             $(floor({npt_dt}/v_ts))
unfix           1
unfix           2
"""


sampling = """
#------------------------------- Sampling -------------------------------------
timestep        $(v_ts)
reset_timestep 	0
fix             1 all nvt temp $(v_T) $(v_T) $(100*v_ts)
fix             2 all momentum 1 linear 1 1 1 angular
dump			1 all custom $({nvt_dt}/1000) {system}.lammpstrj id mol type x y z
dump_modify     1 sort id
run             $(floor({nvt_dt}/v_ts))
unfix           1
unfix           2
"""


write_data = """
#---------------------------- System Output -----------------------------------
write_data      {system}.$(v_T).lammps

################################# All Done ####################################

"""


if __name__ == '__main__':
    main()

