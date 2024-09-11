"""
potential_coefficients.py
==========================

This module provides potential coefficients for use with the LAMMPS
molecular dynamics simulation software. The coefficients are available
for two force fields:

    1) Condensed-phase Optimized Molecular Potentials for Atomistic Simulation Studies (COMPASS)
    2) Polymer Consistent Force Field (PCFF)

Supported materials:
    - Polyethylene (PE)
    - Polypropylene (PP)

Functions
---------
- potential_coefficients(material, potential): Returns the appropriate potential coefficients
  for the specified material and potential type.
"""


def potential_coefficients(material, phase, potential):
    """
    Return the COMPASS or PCFF potential coefficients for the generated amorphous
    or crystal of PP or PE.

    Parameters
    ----------
    material : str
        The material type, either 'PE' for polyethylene or 'PP' for polypropylene.
    potential : str
        The potential type, either 'compass' or 'pcff'.

    Returns
    -------
    str
        The LAMMPS potential coefficients as a formatted string corresponding to
        the specified material and potential type.

    Raises
    ------
    Exception
        If an unsupported material or potential type is specified.
    """
    if material == 'PE':
        if potential == 'compass':
            if phase == 'Amorphous':
                return pe_compass_amorphous
            else:
                return pe_compass_crystalline
        elif potential == 'pcff':
            if phase == 'Amorphous':
                return pe_pcff_amorphous
            else:
                return pe_pcff_crystalline
        else:
            raise Exception('Available potentials: COMPASS and PCFF for PE')
    elif material == 'PP':
        if potential == 'compass':
            return pp_compass
        elif potential == 'pcff':
            return pp_pcff
        else:
            raise Exception('Available potentials: COMPASS and PCFF for PP')
    else:
        raise Exception('Available materials: PE and PP')


pe_compass_amorphous = '''
Pair Coeffs # lj/class2/coul/long

1 0.0620 3.8540 # c2
2 0.0620 3.8540 # c3
3 0.0230 2.8780 # hc

Bond Coeffs # class2

1 1.5300 299.6700 -501.7700 679.8100 # c-c
2 1.1010 345.0000 -691.8900 844.6000 # c-hc

Angle Coeffs # class2

1 107.6600 39.6410 -12.9210 -2.4320 # hc-c-hc
2 110.7700 41.4530 -10.6040  5.1290 # c-c-hc
3 112.6700 39.5160  -7.4430 -9.5583 # c-c-c

BondBond Coeffs

1 5.3316 1.1010 1.1010 # hc-c-hc
2 3.3872 1.5300 1.1010 # c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c

BondAngle Coeffs

1 18.1030 18.1030 1.1010 1.1010 # hc-c-hc
2 20.7540 11.4210 1.5300 1.1010 # c-c-hc
3  8.0160  8.0160 1.5300 1.5300 # c-c-c

Dihedral Coeffs # class2

1  0.0000 0.0000 0.0316 0.0000 -0.1681 0.0000 # c-c-c-hc
2 -0.1432 0.0000 0.0617 0.0000 -0.1530 0.0000 # hc-c-c-hc
3  0.0000 0.0000 0.0514 0.0000 -0.1430 0.0000 # c-c-c-c

AngleAngleTorsion Coeffs

1 -16.1640 112.6700 110.7700 # c-c-c-hc
2 -12.5640 110.7700 110.7700 # hc-c-c-hc
3 -22.0450 112.6700 112.6700 # c-c-c-c

EndBondTorsion Coeffs

1  0.2486 0.2422 -0.0925  0.0814 0.0591  0.2219 1.5300 1.1010 # c-c-c-hc
2  0.2130 0.3120  0.0777  0.2130 0.3120  0.0777 1.1010 1.1010 # hc-c-c-hc
3 -0.0732 0.0000  0.0000 -0.0732 0.0000  0.0000 1.5300 1.5300 # c-c-c-c

MiddleBondTorsion Coeffs

1 -14.8790 -3.6581 -0.3138 1.5300 # c-c-c-hc
2 -14.2610 -0.5322 -0.4864 1.5300 # hc-c-c-hc
3 -17.7870 -7.1877  0.0000 1.5300 # c-c-c-c

BondBond13 Coeffs

1 0.0000 1.5300 1.1010 # c-c-c-hc
2 0.0000 1.1010 1.1010 # hc-c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c-c

AngleTorsion Coeffs

1 -0.2454  0.0000 -0.1136  0.3113  0.4516 -0.1988 112.6700 110.7700 # c-c-c-hc
2 -0.8085  0.5569 -0.2466 -0.8085  0.5569 -0.2466 110.7700 110.7700 # hc-c-c-hc
3  0.3886 -0.3139  0.1389  0.3886 -0.3139  0.1389 112.6700 112.6700 # c-c-c-c

Improper Coeffs # class2

1 0.0000 0.0000 # c-c-hc-hc
2 0.0000 0.0000 # c-c-c-hc
3 0.0000 0.0000 # c-hc-hc-hc

AngleAngle Coeffs

1  0.2738 -0.4825  0.2738 110.7700 107.6600 110.7700 # c-c-hc-hc
2 -1.3199 -1.3199  0.1184 112.6700 110.7700 110.7700 # c-c-c-hc
3 -0.3157 -0.3157 -0.3157 107.6600 107.6600 107.6600 # c-hc-hc-hc
'''


pe_compass_crystalline = '''
Pair Coeffs # lj/class2/coul/long

1 0.0620 3.8540 # c2
2 0.0230 2.8780 # hc

Bond Coeffs # class2

1 1.5300 299.6700 -501.7700 679.8100 # c-c
2 1.1010 345.0000 -691.8900 844.6000 # c-hc

Angle Coeffs # class2

1 107.6600 39.6410 -12.9210 -2.4320 # hc-c-hc
2 110.7700 41.4530 -10.6040  5.1290 # c-c-hc
3 112.6700 39.5160  -7.4430 -9.5583 # c-c-c

BondBond Coeffs

1 5.3316 1.1010 1.1010 # hc-c-hc
2 3.3872 1.5300 1.1010 # c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c

BondAngle Coeffs

1 18.1030 18.1030 1.1010 1.1010 # hc-c-hc
2 20.7540 11.4210 1.5300 1.1010 # c-c-hc
3  8.0160  8.0160 1.5300 1.5300 # c-c-c

Dihedral Coeffs # class2

1  0.0000 0.0000 0.0316 0.0000 -0.1681 0.0000 # c-c-c-hc
2 -0.1432 0.0000 0.0617 0.0000 -0.1530 0.0000 # hc-c-c-hc
3  0.0000 0.0000 0.0514 0.0000 -0.1430 0.0000 # c-c-c-c

AngleAngleTorsion Coeffs

1 -16.1640 112.6700 110.7700 # c-c-c-hc
2 -12.5640 110.7700 110.7700 # hc-c-c-hc
3 -22.0450 112.6700 112.6700 # c-c-c-c

EndBondTorsion Coeffs

1  0.2486 0.2422 -0.0925  0.0814 0.0591  0.2219 1.5300 1.1010 # c-c-c-hc
2  0.2130 0.3120  0.0777  0.2130 0.3120  0.0777 1.1010 1.1010 # hc-c-c-hc
3 -0.0732 0.0000  0.0000 -0.0732 0.0000  0.0000 1.5300 1.5300 # c-c-c-c

MiddleBondTorsion Coeffs

1 -14.8790 -3.6581 -0.3138 1.5300 # c-c-c-hc
2 -14.2610 -0.5322 -0.4864 1.5300 # hc-c-c-hc
3 -17.7870 -7.1877  0.0000 1.5300 # c-c-c-c

BondBond13 Coeffs

1 0.0000 1.5300 1.1010 # c-c-c-hc
2 0.0000 1.1010 1.1010 # hc-c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c-c

AngleTorsion Coeffs

1 -0.2454  0.0000 -0.1136  0.3113  0.4516 -0.1988 112.6700 110.7700 # c-c-c-hc
2 -0.8085  0.5569 -0.2466 -0.8085  0.5569 -0.2466 110.7700 110.7700 # hc-c-c-hc
3  0.3886 -0.3139  0.1389  0.3886 -0.3139  0.1389 112.6700 112.6700 # c-c-c-c

Improper Coeffs # class2

1 0.0000 0.0000 # c-c-hc-hc
2 0.0000 0.0000 # c-c-c-hc

AngleAngle Coeffs

1  0.2738 -0.4825  0.2738 110.7700 107.6600 110.7700 # c-c-hc-hc
2 -1.3199 -1.3199  0.1184 112.6700 110.7700 110.7700 # c-c-c-hc
'''


pe_pcff_amorphous = '''
Pair Coeffs # lj/class2/coul/long

1 0.0540 4.0100 # c2
2 0.0540 4.0100 # c3
3 0.0200 2.9950 # hc

Bond Coeffs # class2

1 1.5300 299.6700 -501.7700 679.8100 # c-c
2 1.1010 345.0000 -691.8900 844.6000 # c-hc

Angle Coeffs # class2

1 107.6600 39.6410 -12.9210 -2.4318 # hc-c-hc
2 110.7700 41.4530 -10.6040  5.1290 # c-c-hc
3 112.6700 39.5160  -7.4430 -9.5583 # c-c-c

BondBond Coeffs

1 5.3316 1.1010 1.1010 # hc-c-hc
2 3.3872 1.1010 1.5300 # c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c

BondAngle Coeffs

1 18.1030 18.1030 1.1010 1.1010 # hc-c-hc
2 11.4210 20.7540 1.1010 1.5300 # c-c-hc
3  8.0160  8.0160 1.5300 1.5300 # c-c-c

Dihedral Coeffs # class2

1  0.0000 0.0000 0.0316 0.0000 -0.1681 0.0000 # c-c-c-hc
2 -0.1432 0.0000 0.0617 0.0000 -0.1083 0.0000 # hc-c-c-hc
3  0.0000 0.0000 0.0514 0.0000 -0.1430 0.0000 # c-c-c-c

AngleAngleTorsion Coeffs

1 -16.1640 110.7700 112.6700 # c-c-c-hc
2 -12.5640 110.7700 110.7700 # hc-c-c-hc
3 -22.0450 112.6700 112.6700 # c-c-c-c

EndBondTorsion Coeffs

1  0.0814 0.0591 0.2219  0.2486 0.2422 -0.0925 1.1010 1.5300 # c-c-c-hc
2  0.2130 0.3120 0.0777  0.2130 0.3120  0.0777 1.1010 1.1010 # hc-c-c-hc
3 -0.0732 0.0000 0.0000 -0.0732 0.0000  0.0000 1.5300 1.5300 # c-c-c-c

MiddleBondTorsion Coeffs

1 -14.8790 -3.6581 -0.3138 1.5300 # c-c-c-hc
2 -14.2610 -0.5322 -0.4864 1.5300 # hc-c-c-hc
3 -17.7870 -7.1877  0.0000 1.5300 # c-c-c-c

BondBond13 Coeffs

1 0.0000 1.1010 1.5300 # c-c-c-hc
2 0.0000 1.1010 1.1010 # hc-c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c-c

AngleTorsion Coeffs

1  0.3113  0.4516 -0.1988 -0.2454  0.0000 -0.1136 110.7700 112.6700 # c-c-c-hc
2 -0.8085  0.5569 -0.2466 -0.8085  0.5569 -0.2466 110.7700 110.7700 # hc-c-c-hc
3  0.3886 -0.3139  0.1389  0.3886 -0.3139  0.1389 112.6700 112.6700 # c-c-c-c

Improper Coeffs # class2

1 0.0000 0.0000 # c-c-hc-hc
2 0.0000 0.0000 # c-c-c-hc
3 0.0000 0.0000 # c-hc-hc-hc

AngleAngle Coeffs

1  0.2738  0.2738 -0.4825 107.6600 110.7700 110.7700 # c-c-hc-hc
2 -1.3199  0.1184 -1.3199 110.7700 112.6700 110.7700 # c-c-c-hc
3 -0.3157 -0.3157 -0.3157 107.6600 107.6600 107.6600 # c-hc-hc-hc
'''


pe_pcff_crystalline = '''
Pair Coeffs # lj/class2/coul/long

1 0.0540 4.0100 # c2
2 0.0540 4.0100 # c3

Bond Coeffs # class2

1 1.5300 299.6700 -501.7700 679.8100 # c-c
2 1.1010 345.0000 -691.8900 844.6000 # c-hc

Angle Coeffs # class2

1 107.6600 39.6410 -12.9210 -2.4318 # hc-c-hc
2 110.7700 41.4530 -10.6040  5.1290 # c-c-hc
3 112.6700 39.5160  -7.4430 -9.5583 # c-c-c

BondBond Coeffs

1 5.3316 1.1010 1.1010 # hc-c-hc
2 3.3872 1.1010 1.5300 # c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c

BondAngle Coeffs

1 18.1030 18.1030 1.1010 1.1010 # hc-c-hc
2 11.4210 20.7540 1.1010 1.5300 # c-c-hc
3  8.0160  8.0160 1.5300 1.5300 # c-c-c

Dihedral Coeffs # class2

1  0.0000 0.0000 0.0316 0.0000 -0.1681 0.0000 # c-c-c-hc
2 -0.1432 0.0000 0.0617 0.0000 -0.1083 0.0000 # hc-c-c-hc
3  0.0000 0.0000 0.0514 0.0000 -0.1430 0.0000 # c-c-c-c

AngleAngleTorsion Coeffs

1 -16.1640 110.7700 112.6700 # c-c-c-hc
2 -12.5640 110.7700 110.7700 # hc-c-c-hc
3 -22.0450 112.6700 112.6700 # c-c-c-c

EndBondTorsion Coeffs

1  0.0814 0.0591 0.2219  0.2486 0.2422 -0.0925 1.1010 1.5300 # c-c-c-hc
2  0.2130 0.3120 0.0777  0.2130 0.3120  0.0777 1.1010 1.1010 # hc-c-c-hc
3 -0.0732 0.0000 0.0000 -0.0732 0.0000  0.0000 1.5300 1.5300 # c-c-c-c

MiddleBondTorsion Coeffs

1 -14.8790 -3.6581 -0.3138 1.5300 # c-c-c-hc
2 -14.2610 -0.5322 -0.4864 1.5300 # hc-c-c-hc
3 -17.7870 -7.1877  0.0000 1.5300 # c-c-c-c

BondBond13 Coeffs

1 0.0000 1.1010 1.5300 # c-c-c-hc
2 0.0000 1.1010 1.1010 # hc-c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c-c

AngleTorsion Coeffs

1  0.3113  0.4516 -0.1988 -0.2454  0.0000 -0.1136 110.7700 112.6700 # c-c-c-hc
2 -0.8085  0.5569 -0.2466 -0.8085  0.5569 -0.2466 110.7700 110.7700 # hc-c-c-hc
3  0.3886 -0.3139  0.1389  0.3886 -0.3139  0.1389 112.6700 112.6700 # c-c-c-c

Improper Coeffs # class2

1 0.0000 0.0000 # c-c-hc-hc
2 0.0000 0.0000 # c-c-c-hc

AngleAngle Coeffs

1  0.2738  0.2738 -0.4825 107.6600 110.7700 110.7700 # c-c-hc-hc
2 -1.3199  0.1184 -1.3199 110.7700 112.6700 110.7700 # c-c-c-hc
'''


pp_compass = '''
Pair Coeffs # lj/class2/coul/long

1 0.0400 3.8540 # c1
2 0.0620 3.8540 # c2
3 0.0620 3.8540 # c3
4 0.0230 2.8780 # hc

Bond Coeffs # class2

1 1.5300 299.6700 -501.7700 679.8100 # c-c
2 1.1010 345.0000 -691.8900 844.6000 # c-hc

Angle Coeffs # class2

1 110.7700 41.4530 -10.6040  5.1290 # c-c-hc
2 107.6600 39.6410 -12.9210 -2.4320 # hc-c-hc
3 112.6700 39.5160  -7.4430 -9.5583 # c-c-c

BondBond Coeffs

1 3.3872 1.5300 1.1010 # c-c-hc
2 5.3316 1.1010 1.1010 # hc-c-hc
3 0.0000 1.5300 1.5300 # c-c-c

BondAngle Coeffs

1 20.7540 11.4210 1.5300 1.1010 # c-c-hc
2 18.1030 18.1030 1.1010 1.1010 # hc-c-hc
3  8.0160  8.0160 1.5300 1.5300 # c-c-c

Dihedral Coeffs # class2

1  0.0000 0.0000 0.0316 0.0000 -0.1681 0.0000 # c-c-c-hc
2 -0.1432 0.0000 0.0617 0.0000 -0.1530 0.0000 # hc-c-c-hc
3  0.0000 0.0000 0.0514 0.0000 -0.1430 0.0000 # c-c-c-c

AngleAngleTorsion Coeffs

1 -16.1640 112.6700 110.7700 # c-c-c-hc
2 -12.5640 110.7700 110.7700 # hc-c-c-hc
3 -22.0450 112.6700 112.6700 # c-c-c-c

EndBondTorsion Coeffs

1  0.2486 0.2422 -0.0925  0.0814 0.0591 0.2219 1.5300 1.1010 # c-c-c-hc
2  0.2130 0.3120  0.0777  0.2130 0.3120 0.0777 1.1010 1.1010 # hc-c-c-hc
3 -0.0732 0.0000  0.0000 -0.0732 0.0000 0.0000 1.5300 1.5300 # c-c-c-c

MiddleBondTorsion Coeffs

1 -14.8790 -3.6581 -0.3138 1.530  # c-c-c-hc
2 -14.2610 -0.5320 -0.4864 1.530  # hc-c-c-hc
3 -17.7870 -7.1877  0.0000 1.530  # c-c-c-c

BondBond13 Coeffs

1 0.0000 1.5300 1.1010 # c-c-c-hc
2 0.0000 1.1010 1.1010 # hc-c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c-c

AngleTorsion Coeffs

1 -0.2454  0.0000 -0.1136  0.3113  0.4516 -0.1988 112.6700 110.7700 # c-c-c-hc
2 -0.8085  0.5569 -0.2466 -0.8085  0.5569 -0.2466 110.7700 110.7700 # hc-c-c-hc
3  0.3886 -0.3139  0.1389  0.3886 -0.3139  0.1389 112.6700 112.6700 # c-c-c-c

Improper Coeffs # class2

1 0.0000 0.0000 # c-c-hc-hc
2 0.0000 0.0000 # c-hc-hc-hc
3 0.0000 0.0000 # c-c-c-c
4 0.0000 0.0000 # c-c-c-hc

AngleAngle Coeffs

1  0.2738  0.2738 -0.4825 107.6600 110.7700 110.7700 # c-c-hc-hc
2 -0.3157 -0.3157 -0.3157 107.6600 107.6600 107.6600 # c-hc-hc-hc
3 -0.1729 -0.1729 -0.1729 112.6700 112.6700 110.7700 # c-c-c-c
4 -1.3119  0.1184 -1.3199 110.7700 112.6700 110.7700 # c-c-c-hc
'''


pp_pcff = '''
Pair Coeffs # lj/class2/coul/long

1 0.0540 4.0100 # c1
2 0.0200 2.9950 # c2
3 0.0620 3.8540 # c3
4 0.0230 2.8780 # hc

Bond Coeffs # class2

1 1.5300 299.6700 -501.7700 679.8100 # c-c
2 1.1010 345.0000 -691.8900 844.6000 # c-hc

Angle Coeffs # class2

1 110.7700 41.4530 -10.6040  5.1290 # c-c-hc
2 107.6600 39.6410 -12.9210 -2.4320 # hc-c-hc
3 112.6700 39.5160 -7.44300 -9.5583 # c-c-c

BondBond Coeffs

1 3.3872 1.5300 1.1010 # c-c-hc
2 5.3316 1.1010 1.1010 # hc-c-hc
3 0.0000 1.5300 1.5300 # c-c-c

BondAngle Coeffs

1 20.7540 11.4210 1.5300 1.1010 # c-c-hc
2 18.1030 18.1030 1.1010 1.1010 # hc-c-hc
3  8.0160  8.0160 1.5300 1.5300 # c-c-c

Dihedral Coeffs # class2

1  0.0000 0.0000 0.0316 0.0000 -0.1681 0.0000 # c-c-c-hc
2 -0.1432 0.0000 0.0617 0.0000 -0.1083 0.0000 # hc-c-c-hc
3  0.0000 0.0000 0.0514 0.0000 -0.1431 0.0000 # c-c-c-c

AngleAngleTorsion Coeffs

1 -16.1640 112.6700 110.7700 # c-c-c-hc
2 -12.5640 110.7700 110.7700 # hc-c-c-hc
3 -22.0450 112.6700 112.6700 # c-c-c-c

EndBondTorsion Coeffs

1  0.2486 0.2422 -0.0925  0.0814 0.0591 0.2219 1.5300 1.1010 # c-c-c-hc
2  0.2130 0.3120  0.0777  0.2130 0.3120 0.0777 1.1010 1.1010 # hc-c-c-hc
3 -0.0732 0.0000  0.0000 -0.0732 0.0000 0.0000 1.5300 1.5300 # c-c-c-c

MiddleBondTorsion Coeffs

1 -14.8790 -3.6581 -0.3138 1.5300 # c-c-c-hc
2 -14.2610 -0.5320 -0.4864 1.5300 # hc-c-c-hc
3 -17.7870 -7.1877  0.0000 1.5300 # c-c-c-c

BondBond13 Coeffs

1 0.0000 1.5300 1.1010 # c-c-c-hc
2 0.0000 1.1010 1.1010 # hc-c-c-hc
3 0.0000 1.5300 1.5300 # c-c-c-c

AngleTorsion Coeffs

1 -0.2454  0.0000 -0.1136  0.3113  0.4516 -0.1988 112.6700 110.7700 # c-c-c-hc
2 -0.8085  0.5569 -0.2466 -0.8085  0.5569 -0.2466 110.7700 110.7700 # hc-c-c-hc
3  0.3886 -0.3139  0.1389  0.3886 -0.3139  0.1389 112.6700 112.6700 # c-c-c-c

Improper Coeffs # class2

1 0.0000 0.0000 # c-c-hc-hc
2 0.0000 0.0000 # c-hc-hc-hc
3 0.0000 0.0000 # c-c-c-c
4 0.0000 0.0000 # c-c-c-hc

AngleAngle Coeffs

1  0.2738  0.2738 -0.4825 107.6600 110.7700 110.7700 # c-c-hc-hc
2 -0.3157 -0.3157 -0.3157 107.6600 107.6600 107.6600 # c-hc-hc-hc
3 -0.1729 -0.1729 -0.1729 112.6700 112.6700 110.7700 # c-c-c-c
4 -1.3119  0.1184 -1.3199 110.7700 112.6700 110.7700 # c-c-c-hc
'''

