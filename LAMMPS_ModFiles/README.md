This directory contains the LAMMPS modification files to incorporate TOIIfilaments to the July 2025 LAMMPS release.

# Contents
...

## Turnover
...

## Intermittent Interactions
...

# Installation
To implement TOIIfilaments in LAMMPS simply copy the `.cpp` and `.h` files from `/src` in this repository to `$LAMMPS_DIR/src` (where `$LAMMPS_DIR` is your local LAMMPS installation directory). After that you simply recompile LAMMPS on your machine using either `make serial` or `make mpi` with your requirements.

# Authors
Christian Vanhille-Campos (vanhillechristian@gmail.com / christian.vanhille@sorbonne.universite.fr)

