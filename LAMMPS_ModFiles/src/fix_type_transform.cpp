/* ----------------------------------------------------------------------

Extension fix to implement particle type changes in LAMMPS.

Particles of type i switch to type j at a constant rate r_ij and particles of type j switch to type i at a constant rate r_ji

Structure:
Every timestep, for every particle, compute switch probability p_switch = r_ij * dt and change particle type from i to j is rand() < p_switch

LAMMPS details:

LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
https://www.lammps.org/, Sandia National Laboratories
LAMMPS development team: developers@lammps.org

Copyright (2003) Sandia Corporation.  Under the terms of Contract
DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
certain rights in this software.  This software is distributed under
the GNU General Public License.

See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
Contributing Author: Christian Vanhille Campos (christian.vanhille@sorbonne-universite.fr)
------------------------------------------------------------------------- */

#include "fix_type_transform.h"

#include "atom.h"
#include "atom_vec.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_bond_history.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "reset_atoms_mol.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <cctype>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <random>
#include <utility>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixTypeTransform::FixTypeTransform(class LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg), random(nullptr), transform_rates(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal fix type/transform command");

  // Get atom types
  ntypes = atom->ntypes;

  // Allocate rate matrix (ntypes x ntypes)
  memory->create(transform_rates,ntypes,ntypes,"type/transform:transform_rates");
  for (int i = 0; i < ntypes; i++) {
    for (int j = 0; j < ntypes; j++) {
      transform_rates[i][j] = 0.0;
    }
  }

  // Default values
  seed = 12345 + comm->me;
  nevery = 1;
  respa_level = -1;
  nlevels_respa = 0;

  // Parse arguments: pairs of type_i type_j rate_ij [type_k type_l rate_kl ...]
  nevery = utils::inumeric(FLERR,arg[3],false,lmp); // sample nevery first
  if (nevery <= 0) error->all(FLERR,"Illegal fix type/transform nevery value");
  int iarg = 4;
  while (iarg < narg) {
    if (iarg + 2 >= narg) error->all(FLERR,"Illegal fix type/transform - incomplete transformation");
    
    int type_i = utils::inumeric(FLERR,arg[iarg],false,lmp);
    int type_j = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
    double rate = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    
    if (type_i < 1 || type_i > ntypes || type_j < 1 || type_j > ntypes)
      error->all(FLERR,"Type index out of bounds in fix type/transform");
    
    transform_rates[type_i-1][type_j-1] = rate;  // Convert to 0-indexed
    iarg += 3;
  }

  // Create random number generator
  random = new RanMars(lmp,seed);

  // Print read type transform rates (debug purposes)
  fprintf(lmp->screen, "\nfix type transform initialised!\n");
  fprintf(lmp->screen, "    number of atom types: %d\n", ntypes);
  for (int i = 0; i < ntypes; i++) {
    for (int j = 0; j < ntypes; j++) {
      fprintf(lmp->screen, "  rate for type transform %d -> %d = %.2f\n", i+1, j+1, transform_rates[i][j]);
    }
  }
  fprintf(lmp->screen, "\n\n");
}

/* ---------------------------------------------------------------------- */

FixTypeTransform::~FixTypeTransform()
{
  delete random;
  memory->destroy(transform_rates);
}

/* ---------------------------------------------------------------------- */

int FixTypeTransform::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTypeTransform::init()
{
  // Check for RESPA
  if (strstr(update->integrate_style,"respa")) {
    nlevels_respa = ((Respa *)update->integrate)->nlevels;
    respa_level = nlevels_respa - 1;
  }
}

/* ---------------------------------------------------------------------- */

void FixTypeTransform::post_integrate()
{
  if (update->ntimestep % nevery) return;

  transform_particle_types();
}

/* ---------------------------------------------------------------------- */

void FixTypeTransform::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == respa_level) post_integrate();
}

/* ---------------------------------------------------------------------- */

void FixTypeTransform::transform_particle_types()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double dt = update->dt;
  tagint *tag = atom->tag;

  for (int i = 0; i < nlocal; i++) {
    int current_type = type[i] - 1;  // Convert to 0-indexed
    
    // Check all possible transformations from current type
    int target_type = 0;
    bool transformed = false;
    while (target_type < ntypes && !transformed) {
      if (target_type == current_type) continue;  // Skip same type
      double rate = transform_rates[current_type][target_type];
      if (rate > 0.0) {
        double p_transform = rate * 1.0;
        // fprintf(lmp->screen, "    attempting %d -> %d on atom %d... rate = %.2f; prob = %.4f;\n", 
          // current_type+1, target_type+1, tag[i], rate, p_transform);
        if (random->uniform() < p_transform) {
          type[i] = target_type + 1;  // Convert back to 1-indexed
          transformed = true;  // Only one transformation per timestep
          // fprintf(lmp->screen, "      rng above prob, transformed.\n\n");
        } // else {fprintf(lmp->screen, "      rng under prob, failed to transform.\n");}
      }
      target_type++;
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixTypeTransform::memory_usage()
{
  double bytes = 0.0;
  bytes += (double)(ntypes * ntypes) * sizeof(double);  // transform_rates
  return bytes;
}