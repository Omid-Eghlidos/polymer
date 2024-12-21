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
from generation.pe.crystalline.system import PECrystallineSystem
from generation.pp.amorphous.system import PPAmorphousSystem
from generation.pp.crystalline.system import PPCrystallineSystem
from generation.pu.amorphous.system import PUAmorphousSystem
from generation.bead_spring.system import BeadSpringSystem
from generation.lib.potential_coefficients import potential_coefficients
from generation.lib.lammps import write_data_file


def generate(material, phase, resolution, settings, verbose=True):
    """
    Generate a specified polymer system based on the given settings.

    Parameters:
    -----------
    material : str
        Type of polymer system to generate ('PE' or 'PP').
    phase : str
        Phase of the polymer system ('amorphous' or 'crystalline').
    settings : dict
        Configuration settings for the polymer system.

    Returns:
    --------
    system : obj
        The generated polymer system instance including PEAmorphousSystem or
        PECrystallineSystem or PPAmorphousSystem or PPCrystallineSystem or BeadSpringSystem

    Notes:
    ------
    This function generates the specified polymer system and writes it to a
    LAMMPS-format data file in the specified atom format.
    """
    if verbose:
        modification = settings['modification'].capitalize() if 'modification' in settings else ''
        system = f'{resolution} {modification} {phase} {material}'
        print(f'# Generating {settings["Ns"]} {system} system(s) ...')

    if phase == 'Amorphous':
        if resolution == 'AA':
            system = amorphous_aa_system(material, settings.copy(), verbose)
        elif resolution == 'CG':
            system = amorphous_cg_system(material, settings.copy(), verbose)
        else:
            raise ValueError('Invalid resolution: {resolution}')
    elif phase == 'Crystalline':
        if resolution == 'AA':
            system = crystalline_aa_system(material, settings.copy(), verbose)
        elif resolution == 'CG':
            system = crystalline_cg_system(material, settings.copy(), verbose)
    else:
        raise ValueError('Invalid phase: {phase}')
    return system


def amorphous_aa_system(material, settings, verbose):
    """
    Generate an all-atomistic (AA) amorphous polymer system and save it to a
    LAMMPS data file.

    Parameters:
    -----------
    material : str
        Type of polymer system to generate ('PE' or 'PP').
    settings: dict
        Required settings for generating the amorphous polymer system.
    verbose : bool, optional
        Whether to print the generated system's properties to the console (default is True).

    Returns:
    --------
    PEAmorphousSystem or PPAmorphousSystem : obj
        The generated amorphous polymer system instance.
    coeffs : str
        Class 2 forcefield coefficients of COMPASS or PCFF for the specified material.
    """
    required = ['Ns', 'Nc', 'Nm']
    defaults = {'atoms_format': 'full', 'random_variation': True}
    # Number of repetitions of soft segment (Liu 2019) for PU
    if material == 'PU':
        required += ['Nss']
        defaults['Nss'] = 1
    key_types = {'Ns': int, 'Nc': int, 'Nm': int, 'Nss': int, 'potential_coeffs': str,
                 'atoms_format': str}
    value_constraints = {'Ns': lambda x: x > 0, 'Nc': lambda x: x > 0,
                         'Nm': lambda x: x > 0, 'Nss': lambda x: x > 0}
    # Validate the input settings
    settings = validate_settings(settings, required, key_types, value_constraints, defaults)

    for ns in range(settings['Ns']):
        if material == 'PE':
            system = PEAmorphousSystem(settings)
        elif material == 'PP':
            system = PPAmorphousSystem(settings)
        elif material == 'PU':
            system = PUAmorphousSystem(settings)
        else:
            raise ValueError('Not implemented material: {material}')
        # Specified forcefield potential coefficients for the material
        if 'potential_coeffs' in settings:
            system.coeffs = potential_coefficients(material, 'Amorphous',
                                                  settings['potential_coeffs'])
        system.resolution = 'AA'
        system.atoms_format = settings['atoms_format']
        system.output = f'aa_{material}_{settings["Nm"]}m{settings["Nc"]}c_{ns+1:02d}'
        # Write the system into a file
        write_data_file(system)
        if verbose:
            verbose_aa_details('Amorphous', ns, system)
    return system


def amorphous_cg_system(material, settings, verbose):
    """
    Generate a coarse-grained (CG) amorphous polymer system using bead-spring model
    and save it to a LAMMPS data file.

    Parameters:
    -----------
    material : str
        Type of polymer system to generate ('PE' or 'PP').
    settings: dict
        Required settings for generating the amorphous polymer system.
    verbose : bool, optional
        Whether to print the generated system's properties to the console (default is True).

    Returns:
    --------
    BeadSpringSystem : obj
        The generated CG amorphous polymer system instance.

    Notes:
    ------
    The system is also saved to a LAMMPS-format data file in the specified atom format.
    """
    required = ['rho', 'Ns', 'Nc', 'Nm', 'beads', 'bonds']
    key_types = {'rho': float, 'Ns': int, 'Nc': int, 'Nm': int, 'monomer': str,
                 'beads': dict, 'bonds': dict, 'angles': dict}
    value_constraints = {'rho': lambda x: x > 0, 'Ns': lambda x: x > 0, 'Nc': lambda x: x > 0,
                         'Nm': lambda x: x > 0, 'monomer': lambda x: len(x) > 0,
                         'beads': lambda x: len(x) > 0, 'bonds': lambda x: len(x) > 0,
                         'angles': lambda x: len(x) > 0}
    defaults = dict(atoms_format='molecular')
    # Validate the input settings
    settings = validate_settings(settings, required, key_types, value_constraints, defaults)

    # Generate n random system with the given properties
    for ns in range(settings['Ns']):
        system = BeadSpringSystem(settings)
        system.output = f'cg_{material}_{settings["Nm"]}m{settings["Nc"]}c_{ns+1:02d}'
        # Write the system into a file
        system.material = material
        system.resolution = 'CG'
        system.atoms_format = settings['atoms_format']
        write_data_file(system)
        if verbose:
            verbose_cg_details('Amorphous', ns, system)
    return system


def crystalline_aa_system(material, settings, verbose):
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
    required = ['Ns', 'Na', 'Nb', 'Nc']
    key_types = {'Ns': int, 'Na': int, 'Nb': int, 'Nc': int, 'modification': str,
                 'potential_coeffs': str, 'atoms_format': str, 'pbc': bool}
    value_constraints = {'Ns': lambda x: x > 0, 'Na': lambda x: x > 0,
                         'Nb': lambda x: x > 0, 'Nc': lambda x: x > 0,
                         'modification': lambda x: x in ['alpha', 'beta']}
    defaults = dict(atoms_format='full', modification='alpha', pbc=True)
    # Validate the input settings
    settings = validate_settings(settings, required, key_types, value_constraints, defaults)

    for ns in range(settings['Ns']):
        if material == 'PE':
            system = PECrystallineSystem(settings)
            modification = ''
        elif material == 'PP':
            system = PPCrystallineSystem(settings)
            modification = f'{settings["modification"]}_'
        else:
            raise ValueError('Not implemented material: {material}')
        # Specified forcefield potential coefficients for the material
        if 'potential_coeffs' in settings:
            system.coeffs = potential_coefficients(material, 'Crystalline',
                                                  settings['potential_coeffs'])
        system.resolution = 'AA'
        system.atoms_format = settings['atoms_format']
        system.output = f'aa_{material}_{modification}'\
                        f'a{settings["Na"]}b{settings["Nb"]}c{settings["Nc"]}_'\
                        f'{ns+1:02d}'
        # Write the system into a file
        write_data_file(system)
        if verbose:
            verbose_aa_details('Crystalline', ns+1, system)
    return system


def crystalline_cg_system(material, settings, verbose):
    raise ValueError('Not implemented: {resolution} {phase}')


def validate_settings(settings, required_keys, key_types, value_constraints, defaults=None):
    """
    Validates the settings provided by the user.
    """
    class SettingsError(Exception):
        """
        Custom exception for invalid settings.
        """
        pass

    # Check if all required keys are present
    for key in required_keys:
        if key not in settings:
            raise SettingsError(f'Missing required setting: {key}')

    # Check if the types of values are correct
    for key, value in settings.items():
        if key not in key_types:
            raise SettingsError(f'Invalid input setting: {key}')
        if not isinstance(value, key_types[key]):
            raise SettingsError(f'Invalid "{key}" type: '
                                f'Expected {key_types[key].__name__}, '
                                f'got {type(value).__name__}')

    # Check if the values meet additional constraints
    for key, constraint_func in value_constraints.items():
        if key in settings and not constraint_func(settings[key]):
            raise SettingsError(f'Invalid value for "{key}": {input_settings[key]}')

    # Add default values if not given in settings (if applicable)
    if defaults:
        for key, value in defaults.items():
            if key not in settings:
                settings[key] = value
    return settings


def verbose_aa_details(phase, ns, system):
    # Show the system properties on the screen
    modification = system.modification.capitalize() if hasattr(system, 'modification') else ''
    print(f'-- Generated {system.resolution} {modification} {phase} {system.material} system {ns+1}')
    if system.material == 'PU':
        types = forcefield_atom_types_equivalence(system.potential_coeffs)
    else:
        types = system._ff_types['atoms']
    name = {}
    for a, t in types.items():
        if a[0].upper() == a[0]:
            if a[0] not in name.values():
                name[len(name)] = a[0]
        else:
            name[t] = a
    atom_types = {a: t + 1 for a, t in types.items()}
    # Space between lines for better clarity
    s = '\n' if len(atom_types) > 5 else ''
    print(f'---- {system.Nt:>4d} atoms of types:\t{atom_types}{s}')
    bond_types = {b: t + 1 for b, t in system._ff_types['bonds'].items()}
    print(f'---- {len(system.bonds):>4d} bonds of types:\t{bond_types}{s}')
    if system.material == 'PU' and system.potential_coeffs.upper() == 'PCFF':
        name = forcefield_atom_types_equivalence(None, True)
    if system.angles:
        angle_types = {f'{name[a[0]]}-{name[a[1]]}-{name[a[2]]}': i + 1
                                     for i, a in enumerate(system.angle_types)}
        print(f'---- {len(system.angles):>4d} angles of types:\t{angle_types}{s}')
    if system.dihedrals:
        dihedral_types = {f'{name[a[0]]}-{name[a[1]]}-{name[a[2]]}-{name[a[3]]}': i + 1
                          for i, a in enumerate(system.dihedral_types)}
        print(f'---- {len(system.dihedrals):>4d} dihedrals of types:\t{dihedral_types}{s}')
    if system.impropers:
        improper_types = {f'{name[a[0]]}-{name[a[1]]}-{name[a[2]]}-{name[a[3]]}': i + 1
                                  for i, a in enumerate(system.improper_types)}
        print(f'---- {len(system.impropers):>4d} impropers of types:\t{improper_types}{s}')
    print(f'---- Output: {system.output}.lammps\n')


def forcefield_atom_types_equivalence(args, equivalences=False):
    """
    Convert the type symbol and ID from PCFF to COMPASS and vice versa.
    """
    compass2pcff = { 'c4':  'c2',  'c4o': 'c_0', "c3'": 'c_1', 'c3"': 'c_2',
                    'c3a':  'cp',  'n3m': 'n_2', 'o1=': 'o_1', 'o2e':  'oh',
                    'o2s': 'o_2',   'h1':  'hc', 'h1n': 'hn2', 'h1o':  'ho'}
    if equivalences:
        return compass2pcff
    if args.upper() == 'PCFF':
        return {pcff: idx for idx, (compass, pcff) in enumerate(compass2pcff.items())}
    else:
        return {compass: idx for idx, (compass, pcff) in enumerate(compass2pcff.items())}


def verbose_cg_details(phase, ns, system):
    """ Show the details of the system on the screen in verbose mode. """
    print(f'-- Generated {system.resolution} {phase} {system.material} system {ns+1} ...')
    masses = {b[0]: b[1] for t, b in sorted(system.atom_types.items())}
    masses['System'] = round(system.mass, 3)
    print(f'{"Masses":>10}: {masses} (g/mol)')
    dimensions = {'Box': f'{system.box[0,0]:.3f} x '
                         f'{system.box[1,1]:.3f} x '
                         f'{system.box[2,2]:.3f}'}
    dimensions['Volume'] = round(system.volume, 3)
    print(f'{"Dimensions":>10}: {dimensions} (A^3)')
    beads = {f'{b[0]:^1}': system.Nm * system.Nc
                                 for t, b in sorted(system.atom_types.items())}
    beads['Total'] = len(system.atoms)
    print(f'{"Beads":>10}: {beads}')
    bonds = {t: len([b for b in system.bonds if b[0]+1 == n])
                                 for t, n in sorted(system.bond_types.items())}
    bonds['Total'] = len(system.bonds)
    print(f'{"Bonds":>10}: {bonds}')
    if system.angle_types:
        angles = {t: len([a for a in system.angles if a[0]+1 == n])
                                for t, n in sorted(system.angle_types.items())}
        angles['Total'] = len(system.angles)
        print(f'{"Angles":>10}: {angles}')
    if system.dihedral_types:
        dihedrals = {t: len([d for d in system.dihedrals if d[0]+1 == n])
                             for t, n in sorted(system.dihedral_types.items())}
        dihedrals['Total'] = len(system.dihedrals)
        print(f'{"Dihedrals":>10}: {dihedrals}')
    if system.improper_types:
        impropers = {t: len([i for i in system.impropers if i[0]+1 == n])
                             for t, n in sorted(system.improper_types.items())}
        impropers['Total'] = len(system.impropers)
        print(f'{"Impropers":>10}: {impropers}')
    print(f'{"Output":>10}: {system.output}.lammps\n')

