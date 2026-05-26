/* ----------------------------------------------------------------------

Extension fix to implement treadmilling filaments in LAMMPS.

Treadmilling filaments are modeled as bead-spring polymers that grow and shrink on opposite sides at controlled rates:
- Growth: a new particle is added at the filament head (position = current head + sigma + noise) at fixed growth rate r_on
- Shrinkage: an existing particle is removed from the filament tail (only the filament tail) with lifetime-dependent rate r_off = r_0 (1 - exp(-r_hyd lifetime))
    where r_0 is the constant unbinding rate of hydrolysed monomers and r_hyd is the constant hydrolysis rate -- see Vanhille Campos et al 2024 (https://doi.org/10.1038/s41567-024-02597-8) for details

Structure:
Every timestep, 
I) for each filament (MOLECULE):
    1. Compute a growth probability p_on = r_on * dt
        If rand() < p_on: GROW:
            1.A. Find filament head (tagged as such, pointer)
            1.B. Sample new head position: new_position = head_position + sigma + small_noise
            1.C. Check overlaps for new position
            1.D. If no overlaps:
                    - create new particle at that position (store current simulation time as birth time)
                    - create bonds and angles for filament continuity
                    - update filament head pointer
                else:
                    - try again up to MAXTRIALS
    2. Compute a shrink proabbilty p_off = r_off * dt = r_0 (1 - exp(-r_hyd lifetime)) * dt
        If rand() < p_off: SHRINK:
            2.A. Find filament tail (tagged as such, pointer)
            2.B. Delete particle and associated bonds and angles & update tail pointer
II) maybe nucleate a new filament:
    Mode 1: free nucleation at constant rate r_nuc:
        A. Define nucleation probability p_nuc = r_nuc * dt, if rand() < p_nuc:
            A.1. Sample new dimer position: uniform sampling of center of mass and orientation
            A.2. Check overlaps and try again if overlapping up to MAXTRIALS
            A.3. If no overlaps create two particles and a bond between them at sampled positions
    Mode 2: nucleate new branch from existing filament:
        B. For each nucleating particle type (some type, e.g. type 4), compute a nucleation probability p_nuc = r_nuc * dt
            if rand() < p_nuc:
            B.1. Sample new particle position: nucleator position + angled vector size sigma + noise
            B.2. Check overlaps and try again if overlapping up to MAXTRIALS
            B.3. If no overlaps create two particles and a bond between them at sampled positions

Timeline in integration step: ALL REACTIONS OCCUR AFTER FORCES INTEGRATION
1. LAMMPS force computation
2. LAMMPS position and velocity update
3. Treadmilling reactions (particle creation and deletion)

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

#include "fix_treadmilling.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"

#include "molecule.h"
// #include "atom_vec.h"
// #include "citeme.h"
// #include "compute.h"
// #include "domain.h"
// #include "fix_bond_history.h"
// #include "group.h"
// #include "input.h"
// #include "math_extra.h"
// #include "reset_atoms_mol.h"
// #include "variable.h"

#include <cctype>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <random>
#include <utility>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

// Constants
#define MAXTRIALS 100
#define OVERLAP_SKIN 1.5

/* ---------------------------------------------------------------------- */
// Filament bounds structure
struct filbounds {
  int head_idx;
  int tail_idx;
  int subhead_idx;
};
/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

FixTreadmilling::FixTreadmilling(class LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), 
  bondcount(nullptr), created(nullptr), copy(nullptr), random(nullptr), list(nullptr)
  // random(nullptr), nlocalkeep(nullptr), nghostlykeep(nullptr)
{
  std::string fixname = fmt::format("fix {}", style);
  if (narg < 6) utils::missing_cmd_args(FLERR, fixname, error);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);  // how often to call this fix (in timesteps) -- inherited from fix.h in fix_treadmilling.h, no need to declare
  if (nevery <= 0) error->all(FLERR, 3, "Illegal fix {} nevery value {}", style, nevery);

  // Fix.h variables
  dynamic_group_allow = 1;              // set to 1 because it can be used with dynamic group
  force_reneighbor = 1;                 // set to 1 so the fix forces reneighboring
  next_reneighbor = -1;                 // next timestep to force a reneighboring -- set to -1 like in fix_bond_create.cpp
  vector_flag = 1;                      // 0/1 if compute_vector() function exists
  size_vector = 6;                      // length of global vector -- 6 variables: timestep and cumul # of atoms, bonds and angles created
  extvector = 0;                        // 0/1/-1 if global vector is all int/ext/extlist
  global_freq = 1;                      // frequency s/v data is available at -- LIKE IN fix_bond_create -- NOT SURE

  // Default values for fix parameters
  // Kinetics: default to passive
  r_on = 0.0;                           // growth rate (constant) -- default: no growth (passive)
  r_off_base = 0.0;                     // off rate after hydrolysis (constant) -- default: no shrinking (passive)
  r_hyd = 0.0;                          // hydrolysis rate (constant) -- default: no hydrolysis (passive)
  r_nuc = 0.0;                          // nucleation rate (constant) -- default: no nucleation (passive)
  nuc_mode = 0;                         // nucleation mode -- 0 no, 1 random, 2 branching
  nucleator_type = -1;                  // branching particle type (if nuc_mode=2)
  // Structure and algorithm
  sigma = 1.0;                          // particle size
  noise_sigma = 0.1;                    // creation position sampling noise magnitude
  max_trials = MAXTRIALS;               // maximum # of trials allowed before giving up on creation
  overlap_cut = 1.2;                    // overlap criterion distance (don't allow creation below this)
  ptype = 1;                            // particle type to create -- default: type 1
  btype = 1;                            // bond type to create -- default: type 1
  atype = 1;                            // angle type to create -- default: type 1
  
//   creation_time_flag = -1;
//   filament_id_flag = -1;
//   filament_pos_flag = -1;
  
  seed = 12345;
  nlevels_respa = 0;                    // from fix_bond_create.cpp / .h -- NOT SURE WE NEED

  // Initialize per-atom property pointers
//   creation_time_data = nullptr;
//   filament_id_data = nullptr;
//   filament_pos_data = nullptr;
  birth_time_index = -1;
  has_creation_time = false;

  // Parse arguments
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"r_on") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling r_on");
      r_on = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"r_off") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling r_off");
      r_off_base = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"r_hyd") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling r_hyd");
      r_hyd = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"r_nuc") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling r_nuc");
      r_nuc = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"nuc_mode") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling nuc_mode");
      nuc_mode = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"nucleator_type") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling nucleator_type");
      nucleator_type = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"sigma") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling sigma");
      sigma = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"noise") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling noise");
      noise_sigma = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"maxtrials") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling maxtrials");
      max_trials = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"overlap") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling overlap");
      overlap_cut = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"ptype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling ptype");
      ptype = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"btype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling btype");
      btype = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"atype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling atype");
      atype = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"seed") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling seed");
      seed = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else {
      error->all(FLERR,fmt::format("Unknown fix treadmilling keyword: {}",arg[iarg]));
    }
  }

  // Print parameters
  fprintf(lmp->screen, "\nfix treadmilling initialised!\n");
  fprintf(lmp->screen, "  r_on        = %.2f\n", r_on);
  fprintf(lmp->screen, "  r_off_base  = %.2f\n", r_off_base);
  fprintf(lmp->screen, "  r_hyd       = %.2f\n", r_hyd);
  fprintf(lmp->screen, "  r_hyd       = %.2f\n", r_hyd);
  fprintf(lmp->screen, "  r_nuc       = %.2f\n", r_nuc);
  fprintf(lmp->screen, "  nuc_mode    = %d\n", nuc_mode);
  fprintf(lmp->screen, "  nuc_type    = %d\n", nucleator_type);
  fprintf(lmp->screen, "  sigma       = %.2f\n", sigma);
  fprintf(lmp->screen, "  noise_sigma = %.2f\n", noise_sigma);
  fprintf(lmp->screen, "  max_trials  = %d\n", max_trials);
  fprintf(lmp->screen, "  overlap_cut = %.2f\n", overlap_cut);
  fprintf(lmp->screen, "  ptype        = %d\n", ptype);
  fprintf(lmp->screen, "  btype        = %d\n", btype);
  fprintf(lmp->screen, "  atype        = %d\n", atype);
  fprintf(lmp->screen, "  seed        = %d\n", seed);
  fprintf(lmp->screen, "\n\n");

  // Error check
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR, Error::NOLASTLINE, "Cannot use fix {} with non-molecular systems", style);
  if (ptype < 1 || ptype > atom->ntypes)
    error->all(FLERR, Error::NOLASTLINE, "Illegal fix {} particle type {} (must be between 1 and {})", style, ptype, atom->ntypes);
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR, Error::NOLASTLINE, "Illegal fix {} bond type {} (must be between 1 and {})", style, btype, atom->nbondtypes);
  if (atype < 1 || atype > atom->nangletypes)
    error->all(FLERR, Error::NOLASTLINE, "Illegal fix {} angle type {} (must be between 1 and {})", style, atype, atom->nangletypes);

  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp, seed + comm->me);

  // Update squared cutoff distance
  overlap_sq = overlap_cut*overlap_cut;

  // Set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors
  comm_forward = MAX(2,2+atom->maxspecial);
  comm_reverse = 2;

  // Initialize arrays for created bonds
  ncreated_bonds = 0;
  maxcreated_bonds = 0;
  created_bonds = nullptr;

  // Allocate copy
  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special_one() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms
  // copied straight from fix_bond_create.cpp
  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  // Zero out stats
  natoms = 0;
  nbonds = 0;
  nangles = 0;
  natomstotal = 0;
  nbondstotal = 0;
  nanglestotal = 0;
}

/* ---------------------------------------------------------------------- */

FixTreadmilling::~FixTreadmilling()
{
  delete random;
  memory->destroy(created);
  delete [] copy;
}

/* ---------------------------------------------------------------------- */

int FixTreadmilling::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::init()
{
  // Check and validate per-atom property "birth_time" exists - get its index
  // User must define it in input script before calling this fix:
  //   fix prop1 all property/atom d_birth_time
  int flag, cols;
  birth_time_index = atom->find_custom("birth_time", flag, cols);

  if (birth_time_index < 0 || flag != 1) {
    has_creation_time = false;
    if (comm->me == 0)
      error->warning(FLERR, "Fix treadmilling: d_birth_time property not found. "
                            "Shrinkage will use constant rate only.");
  } else {
    has_creation_time = true;
  }

  // Check and validate per-atom property "filpos" exists - get its index
  // User must define it in input script before calling this fix:
  //  fix prop2 all property/atom i_filpos
  int flagFP, colsFP;
  filpos_index = atom->find_custom("filpos", flagFP, colsFP);
  if (filpos_index < 0 || flagFP != 1) {error->all(FLERR, "Fix treadmilling: i_filpos property not found. ");}

  // Check for pair style
  if (force->pair == nullptr)
    error->all(FLERR,"Fix treadmilling requires a pair style");

  // Check for bond style
  if (force->bond == nullptr)
    error->all(FLERR, "Fix treadmilling requires a bond style");

  // Check for angle style
  if (force->angle == nullptr)
    error-all(FLERR, "Fix treadmilling requires an angle style");

  // Check for RESPA and get number of RESPA levels if so
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;

  // Need a half neighbor list, built every Nevery steps
  // from fix_bond_create.cpp
  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);
  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::post_integrate()
{
  if (update->ntimestep % nevery) return;

  if (!has_creation_time) return;

  int ncreated_local = 0;

  double dt = update->dt;
  int nlocal = atom->nlocal;

  // Get unique molecule IDs to iterate over filaments
  int *tag = atom->tag;
  int *molecule = atom->molecule;
  int *type = atom->type;
  double *birth_step = atom->dvector[birth_time_index];
  int *filpos = atom->ivector[filpos_index];
  double **x = atom->x;
  
  // Find current max particle and molecule ID
  tagint local_max_tag = 0;
  tagint local_max_mol = 0;
  for (int i = 0; i < nlocal; i++) {
    local_max_tag = MAX(local_max_tag, tag[i]);
    local_max_mol = MAX(local_max_mol, molecule[i]);
  }
  MPI_Allreduce(&local_max_tag, &MAX_TAG, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  MPI_Allreduce(&local_max_mol, &MAX_MOL, 1, MPI_LMP_TAGINT, MPI_MAX, world);

  // Collect unique filament IDs on this processor as well as the *local* indices for their corresponding head and tail
  // Needs heads to be filpos 1, tails to be filpos 2, subheads to be filpos 3 or filpos 4 if also tail. filpos = 0 in bulk
  std::unordered_map<int, filbounds> filament_bounds;
  for (int i = 0; i < nlocal; i++) {
    int m = molecule[i];
    if (m <= 0) continue;
    int k = filpos[i];
    if (k == 0) continue;

    // Insert with sentinel values if not yet seen
    auto it = filament_bounds.find(m);
    if (it == filament_bounds.end()) {
      it = filament_bounds.emplace(m, filbounds{-1, -1, -1, -1}).first;
    }
    // Set index when special non-bulk particle found
    // Non-bulk particle filpos legend:
    //   - Head: 1 always
    //   - Subhead: 2 if pure subhead, 4 if subhead of dimer (also tail)
    //   - Tail: 3 if pure tail, 4 if tail of dimer (also subhead)
    if (k == 1) it->second.head_idx  = i;                   // head index
    if (k == 2 || k == 4) it->second.subhead_idx = i;       // subhead index
    if (k == 3 || k == 4) it->second.tail_idx = i;          // subhead index
  }

  // For each filament on this processor
  for (auto& [mol_id, bounds] : filament_bounds) {
    auto hid = bounds.head_idx;
    auto tid = bounds.tail_idx;
    auto shid = bounds.subhead_idx;
    // Growth event
    if (r_on > 0 && hid != -1) {
      double p_grow = r_on * dt;
      if (random->uniform() < p_grow) {
        // Grow filament
        grow_filament(mol_id, hid, shid);
        ncreated_local++;
      }
    }

    // Shrinkage event
    if (r_off_base > 0.0 && tid != -1) {
      double tage = (update->ntimestep - birth_step[tid]) * dt;
      double p_shrink = compute_shrinkage_rate(tage) * dt;
      if (random->uniform() < p_shrink) {
        shrink_filament(tid);
      }
    }
  } // end of filaments loop

  // Nucleation of new filaments
  if (r_nuc > 0.0) {
    did_nucleate = 0;
    if (comm->me == 0) {
      double p_nuc = r_nuc * dt;
      if (random->uniform() < p_nuc) {
        // Update nucleation flag
        did_nucleate = 1;
      }
    }
    MPI_Bcast(&did_nucleate, 1, MPI_INT, 0, world);
    if (did_nucleate) {
      if (nuc_mode == 1) {
        nucleate_filament();
        ncreated_local += 2;
      } else if (nuc_mode == 2 && nucleator_type > 0) {
        // TO-DO: Implement branching
        // nucleate_branch();
        // // Find random nucleator particle // REVISE THIS
        // for (int i = 0; i < nlocal; i++) {
        //   if (type[i] == nucleator_type && random->uniform() < p_nuc / nlocal) {
        //     nucleate_branch(i);
        //     break;
        //   }
        // }
      }
    }
  }

  // Get global creations
  int ncreated_global = 0;
  MPI_Allreduce(&ncreated_local, &ncreated_global, 1, MPI_INT, MPI_SUM, world);
  atom->natoms += ncreated_global;
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::grow_filament(tagint mol_id, int hidx, int shidx)
{
  int nlocal = atom->nlocal;
  int *molecule = atom->molecule;
  double **x = atom->x;
  int *type = atom->type;
  int *filpos = atom->ivector[filpos_index];

  // Sample new position near head
  double head_pos[3], shead_pos[3], new_pos[3];
  int trials = 0;
  bool placed = false;
  int hproc, shproc;

  // If current head owned by this processor define head position
  int hproc_loc = -1;
  if (hidx != -1) {
    if (domain->is_local(x[hidx])) {
      head_pos[0] = x[hidx][0]; head_pos[1] = x[hidx][1]; head_pos[2] = x[hidx][2];
      hproc_loc = comm->me;
    }
  }
  // Broadcast to all other processors
  MPI_Allreduce(&hproc_loc, &hproc, 1, MPI_INT, MPI_MAX, world);
  if (hproc == -1) {error->all(FLERR, "Fix treadmilling: head particle not found on any processor! Aborting! ");}
  MPI_Bcast(head_pos, 3, MPI_DOUBLE, hproc, world);

  // If current subhead owned by this processor define subhead position
  int shproc_loc = -1;
  if (shidx != -1) {
    if (domain->is_local(x[shidx])) {
      shead_pos[0] = x[shidx][0]; shead_pos[1] = x[shidx][1]; shead_pos[2] = x[shidx][2];
      shproc_loc = comm->me;
    }
  }
  // Broadcast to all other processors
  MPI_Allreduce(&shproc_loc, &shproc, 1, MPI_INT, MPI_MAX, world);
  if (shproc == -1) {error->all(FLERR, "Fix treadmilling: subhead particle not found on any processor! Aborting! ");}
  MPI_Bcast(shead_pos, 3, MPI_DOUBLE, shproc, world);

  // Sample new position
  while (!placed && trials < max_trials) {
    // Only sample new position on a main thread (rank 0)
    if (comm->me == 0) {
      // Displacement
      double dx = head_pos[0] - shead_pos[0];
      double dy = head_pos[1] - shead_pos[1];
      double dz = head_pos[2] - shead_pos[2];
      double len = sqrt(dx*dx+dy*dy+dz*dz);
      if (len > 0) {dx /= len; dy /= len; dz /= len;}
      // New position
      new_pos[0] = head_pos[0] + dx + noise_sigma * random->gaussian();
      new_pos[1] = head_pos[1] + dy + noise_sigma * random->gaussian();
      new_pos[2] = head_pos[2] + dz + noise_sigma * random->gaussian();
      // Check PBCs
      domain->remap(new_pos);
    }
    // Broadcast new position to all threads
    MPI_Bcast(new_pos, 3, MPI_DOUBLE, 0, world);
    // Check overlaps
    if (!check_overlap(new_pos)) {placed=true;}
    trials++;
  }

  if (!placed) {      // return early if failed placement
    error->warning(FLERR, "Fix treadmilling: failed particle placement in filament growth. ");
    return;
  }

  // Create new particle
  //// Find new particle ID
  tagint new_tag = 0;
  if (comm->me == 0) {
    new_tag = MAX_TAG+1;
  }
  //// Broadcast 
  MPI_Bcast(&new_tag, 1, MPI_LMP_TAGINT, 0, world);
  MAX_TAG = new_tag; // Global update of MAX_TAG -- executed by all processors after broadcast
  //// Locally create particle, if rank owns position of new particle
  if (domain->is_local(new_pos)) {
    create_particle(new_pos, new_tag, create_type, mol_id, 1);
  }

  // Create bonds
  // TO-DO

  // Create angles
  // TO-DO
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::shrink_filament(int tail_idx)
{
  int *filpos = atom->ivector[filpos_index];
  // Delete bonds involving tail_idx (should be a single one) and return the index of the bound particle (new tail)
  int stidx = delete_bonds(tail_idx);
  if (stidx == -1) {error->all(FLERR, "Fix treadmilling: invalid bound particle index upon tail depolymerisation. ")}
  // If dimer remove all, if not change filpos os subtail
  bool dimer = false;
  if (filpos[tail_idx] == 4) {
    // Tail is also subhead! This is a dimer!
    dimer = true;
  } else {
    // Not a dimer, change filpos of subtail
    if (filpos[stidx] == 0) {
      // Bulk becomes pure tail
      filpos[stidx] = 3;
    } else if (filpos[stidx] == 2) {
      // Trimer! Becomes mixed tail/subhead
      filpos[stidx] = 4;
    } else {
        error->all(FLERR, "Fix treadmilling: invalid subtail filpos value. ");
    }
  }
  delete_particle(tail_idx);
  if (dimer) {delete_particle(stidx);}
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::nucleate_filament()
{
  // Relevant variables
  double pos1[3], pos2[3];
  int trials = 0;
  int placed = 0;
  int nlocal = atom->nlocal;
  tagint new_mol_id = 0;

  // Attempt placement
  while (placed!=1 && trials < max_trials) {
    // First, sample new placement: only on Rank 0
    if (comm->me == 0) {  
      // Random tail position
      pos1[0] = domain->boxlo[0] + random->uniform() * (domain->boxhi[0] - domain->boxlo[0]);
      pos1[1] = domain->boxlo[1] + random->uniform() * (domain->boxhi[1] - domain->boxlo[1]);
      pos1[2] = domain->boxlo[2] + random->uniform() * (domain->boxhi[2] - domain->boxlo[2]);

      // Orient bond randomly
      double theta = 2.0 * M_PI * random->uniform();
      double phi = acos(2.0 * random->uniform() - 1.0);
      
      // Corresponding head positions
      pos2[0] = pos1[0] + sigma * sin(phi) * cos(theta);
      pos2[1] = pos1[1] + sigma * sin(phi) * sin(theta);
      pos2[2] = pos1[2] + sigma * cos(phi);

      // Check PBCs
      domain->remap(pos1);
      domain->remap(pos2);
    }

    // Broadcast candidate positions to all ranks
    MPI_Bcast(pos1, 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(pos2, 3, MPI_DOUBLE, 0, world);

    // Check overlaps globally
    if (!check_overlap(pos1) && !check_overlap(pos2)) {
      placed = 1;
    }

    // Increment trials count
    trials++;
    }

  // Return early if not placed
  if (placed!=1) {return;}

  // Broadcast molecule id counters
  if (comm->me == 0) new_mol_id = MAX_MOL+1;
  MPI_Bcast(&new_mol_id, 1, MPI_LMP_TAGINT, 0, world);
  MAX_MOL = new_mol_id; // Global update of MAX_MOL -- executed on all processors after broadcast

  // Create dimer
  //// Find new particle IDs
  tagint new_tag_1 = 0;
  tagint new_tag_2 = 0;
  if (comm->me == 0) {
    new_tag_1 = MAX_TAG+1;
    new_tag_2 = MAX_TAG+2;
  }
  //// Broadcast 
  MPI_Bcast(&new_tag_1, 1, MPI_LMP_TAGINT, 0, world);
  MPI_Bcast(&new_tag_2, 1, MPI_LMP_TAGINT, 0, world);
  MAX_TAG = new_tag_2; // Global update of MAX_TAG -- executed on all processors after broadcast

  //// Locally create particles
  if (domain->is_local(pos1)) {
    // If rank owns position of tail it creates tail
    int tail_fp = 4;                                                    // tail has filpos = 4 in dimer
    create_particle(pos1, new_tag_1, create_type, new_mol_id, tail_fp);
  }
  if (domain->is_local(pos2)) {
    // If rank owns position of head it creates head
    int head_fp = 1;                                                    // head has filpos = 1 always, including in dimer
    create_particle(pos2, new_tag_2, create_type, new_mol_id, head_fp);
  }

  // Create dimer bonds
  // TO-DO
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::nucleate_branch(int nucleator_idx)
{
  // Similar to nucleate_filament but starting from nucleator position
  double **x = atom->x;
  double pos1[3], pos2[3];
  int trials = 0;
  bool placed = false;
  static int mol_counter = 0;  // Simple counter for new molecule IDs

  while (!placed && trials < max_trials) {
    double theta = 2.0 * M_PI * random->uniform();
    double phi = acos(2.0 * random->uniform() - 1.0);
    
    pos1[0] = x[nucleator_idx][0] + sigma * sin(phi) * cos(theta) + noise_sigma * random->gaussian();
    pos1[1] = x[nucleator_idx][1] + sigma * sin(phi) * sin(theta) + noise_sigma * random->gaussian();
    pos1[2] = x[nucleator_idx][2] + sigma * cos(phi) + noise_sigma * random->gaussian();

    pos2[0] = pos1[0] + sigma * sin(phi) * cos(theta);
    pos2[1] = pos1[1] + sigma * sin(phi) * sin(theta);
    pos2[2] = pos1[2] + sigma * cos(phi);

    if (pos1[0] >= domain->boxlo[0] && pos1[0] <= domain->boxhi[0] &&
        pos1[1] >= domain->boxlo[1] && pos1[1] <= domain->boxhi[1] &&
        pos1[2] >= domain->boxlo[2] && pos1[2] <= domain->boxhi[2] &&
        pos2[0] >= domain->boxlo[0] && pos2[0] <= domain->boxhi[0] &&
        pos2[1] >= domain->boxlo[1] && pos2[1] <= domain->boxhi[1] &&
        pos2[2] >= domain->boxlo[2] && pos2[2] <= domain->boxhi[2]) {
      int dummy;
      if (!check_overlap(pos1, -1, dummy) && !check_overlap(pos2, -1, dummy)) {
        placed = true;
      }
    }
    trials++;
  }

  if (placed) {
    int nlocal = atom->nlocal;
    tagint mol_id = ++mol_counter;  // Simple molecule ID counter
    
    create_particle(pos1, particle_type, mol_id);
    create_particle(pos2, particle_type, mol_id);
    create_bonds(nlocal, nlocal + 1);
  }
}

/* ---------------------------------------------------------------------- */

bool FixTreadmilling::in_this_proc(double *pos)
{
  double *sublo,*subhi;
  if (domain->triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  if (pos[0] >= sublo[0] && pos[0] < subhi[0] &&
      pos[1] >= sublo[1] && pos[1] < subhi[1] &&
      pos[2] >= sublo[2] && pos[2] < subhi[2]) {return true};
  else {return false;}
}

/* ---------------------------------------------------------------------- */

bool FixTreadmilling::check_overlap(double *pos)
{
  double **x = atom->x;
  int nlocal = atom->nlocal;
  int flag = 0;
  double delx, dely, delz;
  for (i = 0; i < nlocal; i++) {
    delx = pos[0] - x[i][0];
    dely = pos[1] - x[i][1];
    delz = pos[2] - x[i][2];
    domain->minimum_image(FLERR, delx, dely, delz);
    rsq = delx*delx + dely*dely + delz*delz;
    if (rsq < overlap_sq) flag = 1;
  }
  int flagall = 0;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);
  if (flagall == 0) {return false;}
  else {return true;}
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::create_particle(double *pos, tagint tag, int type_id, tagint mol_id, int fp)
{
  int nlocal = atom->nlocal;
  
  atom->avec->create_atom(type_id, pos);
  
  // Set atom properties
  atom->tag[nlocal]       = tag;
  atom->molecule[nlocal]  = mol_id; // Set molecule id - same as filament
  atom->mask[nlocal]      = groupbit;
  
  // Initialize velocity from Boltzmann distribution
  double mass = atom->mass[type_id];
  if (mass <= 0.0) {
    error->warning(FLERR, "Fix treadmilling: mass is <= 0.0 so set to 1.0 for velocity initialisation during particle creation. ");
    mass = 1.0;  // Default if mass not set
  }
  double boltzfac = sqrt(force->boltz * temp / mass);
  atom->v[nlocal][0] = boltzfac * random->gaussian();
  atom->v[nlocal][1] = boltzfac * random->gaussian();
  atom->v[nlocal][2] = boltzfac * random->gaussian();
  
  // Per-atom custom properties
  if (has_creation_time) {atom->dvector[birth_time_index][nlocal] = update->ntimestep;}
  atom->ivector[filpos_index][nlocal] = fp;

  // Update atom map so new tag is findable immediately
  if (atom->map_style != Atom::MAP_NONE) {atom->map_one(tag, nlocal);}
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::delete_particle(int idx)
{
  // Delete particle by swapping with last atom and decrementing count
  int nlocal = atom->nlocal;
  if (idx < nlocal - 1) {
    atom->avec->copy(nlocal - 1, idx, 1);
  }
  atom->nlocal--;
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::create_bond(tagint tagi, tagint tagj, int btype)
{
  // Order atoms by tag
  tagint tag1, tag2;
  if (tagi <= tagj) {tag1 = tagi; tag2 = tagj;}
  else {tag1 = tagj; tag2 = tagi;}
  
  // Find local indices (may be -1 if not owned by this rank)
  int i1 = atom->map(tag1);
  int i2 = atom->map(tag2);

  // Declare relevant variables and pointers
  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int *mask = atom->mask;
  int *type = atom->type;
  int num_bond = atom->num_bond;
  int newton_bond = force->newton_bond;
  int ncreate = 0;
  
  // Update bond_atom and bond_type for atoms 1 and 2 consistently whatever processor they're on
  // If newton_bond is set, only store bond on atom 1 (smaller tag)
  // If not newton_bond, store on both atoms (each owns copy)
  
  if (i1 >= 0 && i1 < atom->nlocal) {
    if (num_bond[i1] >= atom->bond_per_atom)
      error->one(FLERR, "Too many bonds per atom");
    int nb = num_bond[i1];
    bond_atom[i1][nb] = tag2;
    bond_type[i1][nb] = btype;
    num_bond[i1]++;
    ncreate++;
  }
  
  if (i2 >= 0 && i2 < atom->nlocal && !newton_bond) {
    if (num_bond[i2] >= atom->bond_per_atom)
      error->one(FLERR, "Too many bonds per atom");
    int nb = atom->num_bond[i2];
    bond_atom[i2][nb] = tag1;
    bond_type[i2][nb] = btype;
    num_bond[i2]++;
  }
  
  // Update special neighbor lists (add 1-2 neighbors) consistently across processors
  
  // Add tag2 to special list of atom i1 (if owned locally)
  if (i1 >= 0 && i1 < atom->nlocal) {
    int n1 = nspecial[i1][0];  // current number of 1-2 neighbors
    int n3 = nspecial[i1][2];  // current number of 1-4 neighbors
    if (n3 >= atom-maxspecial) {error->one(FLERR, "Special neighbor list overflow in fix treadmilling");}
    
    // Check if tag2 already in special list -- should not be the case as typically follows particle creation
    bool already_there = false;
    for (int k = 0; k < n3; k++) {
      if (special[i1][k] == tag2) {
        already_there = true;
        break;
      }
    }
    if (already_there) {error->one(FLERR, "Special neighbor already assigned before particle creation in fix treadmilling. Should be impossible!");}
    
    // Shift 1-3 and 1-4 neighbors to make room for new 1-2 neighbor
    for (int k = n3; k > n1; k--) {
      special[i1][k] = special[i1][k-1];
    }
    special[i1][n1] = tag2;
    nspecial[i1][0]++;
    nspecial[i1][1]++;
    nspecial[i1][2]++;
  }
  
  // Add tag1 to special list of atom i2 (if owned locally)
  if (i2 >= 0 && i2 < atom->nlocal) {
    int n1 = nspecial[i2][0];
    int n3 = nspecial[i2][2];
    if (n3 >= atom-maxspecial) {error->one(FLERR, "Special neighbor list overflow in fix treadmilling");}
    
    bool already_there = false;
    for (int k = 0; k < n3; k++) {
      if (special[i2][k] == tag1) {
        already_there = true;
        break;
      }
    }
    if (already_there) {error->one(FLERR, "Special neighbor already assigned before particle creation in fix treadmilling. Should be impossible!");}
    
    for (int k = n3; k > n1; k--) {
      special[i2][k] = special[i2][k-1];
    }
    special[i2][n1] = tag1;
    nspecial[i2][0]++;
    nspecial[i2][1]++;
    nspecial[i2][2]++;
  }

  // Forward comm all specials so ghost atoms have updated bond info
  comm->forward_comm(this);

  // Update bonded special neighbours beyond 1-2 neighs consistently across processors
  // Using rebuild_special_one directly from fix_bond_create.cpp
  rebuild_special_one(i1);
  rebuild_special_one(i2);
  
  // Safely update global bond count via MPI reduction
  int ncreate_global = 0;
  MPI_Allreduce(&ncreate, &ncreate_global, 1, MPI_INT, MPI_SUM, world);
  if (ncreate_global > 1) {error->warning(FLERR, "Created more than one bond in fix treadmilling! Revise implementation!!");}
  atom->nbonds += ncreate_global;

  // Trigger reneigbouring if any bonds were formed
  if (ncreate_global > 0) next_reneighbor = update->ntimestep;
  
  // TODO: Optional topology rebuild
  // If angles are enabled and atype is set, rebuild special lists for i1 and i2
  // and create angles involving the new bond
  // This would follow the pattern from FixBondCreate::update_topology()
  // For now, skip if treadmilling doesn't use angles
}

/* ---------------------------------------------------------------------- */

int FixTreadmilling::delete_bonds(int idx)
{
  // Delete all bonds involving particle idx
  // This requires managing the bond list
  if (!atom->avec->bonds_allow) return -1;
  if (idx < 0 || idx >= atom->nlocal) return -1;

  // Simple approach: clear all bonds for this atom
  atom->num_bond[idx] = 0;

  // Also remove bonds from other atoms pointing to this one
  tagint deleted_tag = atom->tag[idx];
  int nlocal = atom->nlocal;
  int stidx = -1;
  
  for (int i = 0; i < nlocal; i++) {
    if (i == idx) continue;
    
    // Search through this atom's bonds for the one pointing to idx
    int m = 0;
    while (m < atom->num_bond[i]) {
      if (atom->bond_atom[i][m] == deleted_tag) {
        // Found it - remove by shifting remaining bonds
        for (int n = m; n < atom->num_bond[i] - 1; n++) {
          atom->bond_type[i][n] = atom->bond_type[i][n + 1];
          atom->bond_atom[i][n] = atom->bond_atom[i][n + 1];
        }
        atom->num_bond[i]--;
        // Store index for subtail retagging as tail
        stidx = i;
      } else {
        m++;
      }
    }
  }
  return stidx;
}

/* ---------------------------------------------------------------------- */

double FixTreadmilling::compute_shrinkage_rate(double lifetime)
{
  if (r_hyd <= 0.0) return 0.0;
  return r_off_base * (1.0 - exp(-r_hyd * lifetime));
}

/* ---------------------------------------------------------------------- */
int FixTreadmilling::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,ns;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    ns = (int) ubuf(buf[m++]).i;
    nspecial[i][0] = ns;
    for (j = 0; j < ns; j++)
      special[i][j] = (tagint) ubuf(buf[m++]).i;
  }
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::rebuild_special_one(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR, "Fix treadmilling needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR, "Special list size exceeded in fix treadmilling");

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR, "Fix treadmilling needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR, "Special list size exceeded in fix treadmilling");

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ---------------------------------------------------------------------- */

double FixTreadmilling::compute_vector(int n)
{
    if (n == 0) return (double) natoms;
    if (n == 1) return (double) nbonds;
    if (n == 2) return (double) nangles;
    if (n == 3) return (double) natomstotal;
    if (n == 4) return (double) nbondstotal;
    if (n == 5) return (double) nanglestotal;
    return 0.0;
}

/* ---------------------------------------------------------------------- */

double FixTreadmilling::memory_usage()
{
  double bytes = 0.0;
  bytes += (double)(atom->nmax * 2) * sizeof(int);  // nlocalkeep, nghostlykeep
  return bytes;
}