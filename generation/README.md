# Generation Module

The `generation` module is part of the Polymer Library and provides tools for creating and managing amorphous polymer systems, such as polyethylene (PE) and polypropylene (PP), for molecular dynamics simulations.

## Overview

This module includes specific implementations for constructing amorphous polymer systems, including:

- **PolymerSystem**: 	 Shared class for generating polymer systems.
- **PEAmorphousSystem**: For generating amorphous polyethylene systems.
- **PPAmorphousSystem**: For generating amorphous polypropylene systems.

## Usage

Here's an example of how to use the `generation` module within the Polymer Library:

### Example: Creating a PE or PP Amorphous System

```python
from polymer.generation import generate

# Define the material, phase, and settings for generating a LAMMPS-formatted data file
# Generating 1 PE amorphous system of 25 chains, each with 80 monomers
settings_pe = {
    'Ns': 1,              			# Number of systems
    'Nc': 25,             			# Number of chains per system
    'Nm': 80,             			# Number of monomers per chain
    'potential_coeffs': 'compass', 	# Type of potential coefficients to include
    'output_format': 'full',       	# Molecular structure format of the data file
    'random_variation': True,      	# Add random variation in the torsion angle
    'screen': True                 	# Whether to print details to the console
}
pe_system = generate('PE', 'amorphous', settings_pe)

# Generating 1 PP amorphous system of 10 chains, each with 90 monomers
settings_pp = {
    'Ns': 1,              			# Number of systems
    'Nc': 10,             			# Number of chains per system
    'Nm': 90,             			# Number of monomers per chain
    'potential_coeffs': 'compass', 	# Type of potential coefficients to include
    'output_format': 'full',       	# Molecular structure format of the data file
    'random_variation': True,      	# Add random variation in the torsion angle
    'screen': True                 	# Whether to print details to the console
}
pp_system = generate('PP', 'amorphous', settings_pp)
```

## Structure

* `lib/`: Contains shared libraries used by different polymer systems.
* `pe/amorphous/`: Contains `system.py`, which defines the `PEAmorphousSystem` class for generating amorphous PE systems.
* `pp/amorphous/`: Contains `system.py`, which defines the `PPAmorphousSystem` class for generating amorphous PP systems.

## Installation

As this module is part of the broader Polymer Library, please refer to the main `README.md` in the root of the repository for installation instructions.

## Contributing

Contributions specific to the `generation` module should follow the same guidelines as the Polymer Library.
Please ensure that changes are consistent with the overall structure and standards of the Polymer Library.

## License

This project is licensed under the terms of the [MIT License](../LICENSE).

