"""
generate.py
===========

This module provides centralized functions to generate various polymer systems,
including amorphous and crystalline forms of polyethylene (PE) and polypropylene (PP).

Classes:
--------
PEAmorphousSystem
    A class to represent an amorphous polyethylene (PE) system.
PPAmorphousSystem
    A class to represent an amorphous polypropylene (PP) system.

Functions:
----------
generate(material, phase, settings)
    Generates the specified polymer system based on input settings.

amorphous_system(material, Ns, Nc, Nm, potential_coeffs='compass',
                 atoms_format='full', random_variation=True, screen=True)
    Generates an amorphous PE or PP system with the given parameters and writes it to a file.

crystal_system(material, Ns, Na, Nb, Nc, potential_coeffs='compass',
               atoms_format='full', random_variation=True, screen=True)
    Placeholder function for generating crystalline polymer systems.
"""

from generation.pe.amorphous.system import PEAmorphousSystem
from generation.pp.amorphous.system import PPAmorphousSystem
from generation.lib.potential_coefficients import potential_coefficients
from generation.lib.lammps import write_data_file


def generate(material, phase, settings):
    """
    Generate a specified polymer system based on the given settings.

    Parameters
    ----------
    material : str
        Type of polymer system to generate ('PE' or 'PP').
    phase : str
        Phase of the polymer system ('amorphous' or 'crystalline').
    settings : dict
        Configuration settings for the polymer system.

    Returns
    -------
    PEAmorphousSystem or PPAmorphousSystem
        The generated polymer system instance.

    Notes
    -----
    This function generates the specified polymer system and writes it to a file.
    """
    if material == 'PE':
        if phase == 'amorphous':
            system = amorphous_system(material, **settings)
        elif phase == 'crystalline':
            system = crystalline_system(**settings)
        else:
            raise ValueError('Invalid phase for PE.')
    elif material == 'PP':
        if phase == 'amorphous':
            system = amorphous_system(material, **settings)
        elif phase == 'crystalline':
            system = crystalline_system(**settings)
        else:
            raise ValueError('Invalid phase for PP.')
    else:
        raise ValueError('Invalid material.')


def amorphous_system(material, Ns, Nc, Nm, potential_coeffs='compass',
                     atoms_format='full', random_variation=True, screen=True):
    """
    Generate an amorphous polymer system and save it to a LAMMPS data file.

    Parameters
    ----------
    material : str
        Type of polymer system to generate ('PE' or 'PP').
    Ns : int
        Number of systems to generate.
    Nc : int
        Number of chains per system.
    Nm : int
        Number of monomers per chain.
    potential_coeffs : dict
        Potential coefficients for forcefield (e.g., COMPASS or PCFF).
    atoms_format : str
        LAMMPS atom format to write the molecular structure ('full' or 'molecular').
    random_variation : bool, optional
        Whether to apply random variation to bond angles and lengths (default is True).
    screen : bool, optional
        Whether to print the generated system's properties to the console (default is True).

    Returns
    -------
    PEAmorphousSystem or PPAmorphousSystem
        The generated amorphous polymer system instance.

    Notes
    -----
    The system is also saved to a LAMMPS-format data file in the specified atom format.
    """
    if screen:
        # Print detailed information
        print(f'# Generating {Ns} amorphous {material} system(s) ...')
    for ns in range(Ns):
        if material == 'PE':
            system = PEAmorphousSystem(Nc, Nm, random_variation)
        else:
            system = PPAmorphousSystem(Nc, Nm, random_variation)
        output = f'{material}_{Nm}m{Nc}c_{ns+1:02d}.lammps'
        coeffs = potential_coefficients(material, potential_coeffs)
        write_data_file(output, system, atoms_format, coeffs)

        if screen:
            # Show the system properties on the screen
            print(f'-- System {ns+1}')
            name = {0: 'C', 1: 'H'}
            atom_types = {a: t + 1 for (a, t) in system._forcefield_atom_types.items()}
            print(f'---- {system.Na:>4d} atoms of types:\t{atom_types}')
            if system.bonds:
                bond_types = {name[a[0]]+name[a[1]]: i + 1
                               for i, a in enumerate(system.unique_bond_types)}
                print(f'---- {len(system.bonds):>4d} bonds of types:\t{bond_types}')
            if system.angles:
                angle_types = {name[a[0]]+name[a[1]]+name[a[2]]: i + 1
                                     for i, a in enumerate(system.angle_types)}
                print(f'---- {len(system.angles):>4d} angles of types:\t{angle_types}')
            if system.dihedrals:
                dihedral_types = {name[a[0]]+name[a[1]]+name[a[2]]+name[a[3]]: i + 1
                                  for i, a in enumerate(system.dihedral_types)}
                print(f'---- {len(system.dihedrals):>4d} dihedrals of types:\t{dihedral_types}')
            if system.impropers:
                improper_types = {name[a[0]]+name[a[1]]+name[a[2]]+name[a[3]]: i + 1
                                  for i, a in enumerate(system.improper_types)}
                print(f'---- {len(system.impropers):>4d} impropers of types:\t{improper_types}')
            print(f'---- Output: {output}\n')
    return system


def crystal_system(material, Ns, Na, Nb, Nc, potential_coeffs='compass',
                   atoms_format='full', random_variation=True, screen=True):
    """
    Placeholder function for generating crystalline polymer systems.

    Parameters:
    -----------
    material : str
        Type of the polymer system to generate (e.g., 'PE', 'PP').
    Ns : int
        Number of systems to generate.
    Na, Nb, Nc : int
        Number of unit cells along the a, b, c crystallographic directions.
    atoms_format : str
        LAMMPS' atoms format to write the molecular structure in 'full' or 'molecular'
    potential_coeffs : dict
        Potential coefficients (e.g., COMPASS or PCFF).
    random_variation : bool, optional
        Whether to apply random variation to bond angles and lengths (default is True).
    screen : bool, optional
        Whether to print the generated system's properties to the console (default is True).

    Returns:
    --------
    None
        The function will generate the crystalline polymer system and write it to a file.
    """
    pass

