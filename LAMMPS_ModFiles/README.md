This directory contains the LAMMPS modification files to incorporate TOIIfilaments to the July 2025 LAMMPS release.

# Contents
- `src/` - contains all the LAMMPS modification C++ files:
    - `fix_treadmilling.cpp`
    - `fix_treadmilling.h`
    - `fix_type_transform.cpp`
    - `fix_type_transform.h`
- ...

## Turnover
...

## Intermittent Interactions
...

# Installation
To implement TOIIfilaments in LAMMPS simply copy the `.cpp` and `.h` files from `/src` in this repository to `$LAMMPS_DIR/src` (where `$LAMMPS_DIR` is your local LAMMPS installation directory). After that you simply recompile LAMMPS on your machine using either `make serial` or `make mpi` with your requirements.

## LAMMPS installation
If you have not installed LAMMPS on your machine yet follow these simple steps to install the July 2025 release (which these modifications have been developed and tested on).

Open your terminal and, in the desired location, run
```...```

Next, ...
```...```

## Modifications inclusion
...

# Authors
Christian Vanhille-Campos (vanhillechristian@gmail.com / christian.vanhille@sorbonne.universite.fr)

