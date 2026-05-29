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
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"

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
  copy(nullptr), random(nullptr), list(nullptr)
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
  temp = 1.0;                           // temperature for velocity initialization during creation -- default: 1.0
  
  seed = 12345;
  nlevels_respa = 0;                    // from fix_bond_create.cpp / .h

  // Special per-atom property handlers
  birth_time_index = -1;
  has_creation_time = false;

  // Parse arguments
  int iarg = 4;
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
    } else if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix treadmilling temp");
      temp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
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
  fprintf(lmp->screen, "  ptype       = %d\n", ptype);
  fprintf(lmp->screen, "  btype       = %d\n", btype);
  fprintf(lmp->screen, "  atype       = %d\n", atype);
  fprintf(lmp->screen, "  seed        = %d\n", seed);
  fprintf(lmp->screen, "  temp        = %.2f\n", temp);
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
  natomsloc = 0;
  nbondsloc = 0;
  nanglesloc = 0;
  natomstotal = 0;
  nbondstotal = 0;
  nanglestotal = 0;
}

/* ---------------------------------------------------------------------- */

FixTreadmilling::~FixTreadmilling()
{
  delete random;
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

  // Set comm sizes needed by this fix
  // special list: 1 (count) + maxspecial (entries)
  // filpos: 1
  // birth_time: 1 (if present)
  comm_forward = 2 + atom->maxspecial + (has_creation_time ? 1 : 0);
  comm_reverse = 0;   // no reverse communication needed yet

  // Check and validate per-atom property "filpos" exists - get its index
  // User must define it in input script before calling this fix:
  //  fix prop2 all property/atom i_filpos
  int flagFP, colsFP;
  filpos_index = atom->find_custom("filpos", flagFP, colsFP);
  if (filpos_index < 0 || flagFP != 0) {error->all(FLERR, "Fix treadmilling: i_filpos property not found. ");}

  // Check for pair style
  if (force->pair == nullptr)
    error->all(FLERR,"Fix treadmilling requires a pair style");

  // Check for bond style
  if (force->bond == nullptr)
    error->all(FLERR, "Fix treadmilling requires a bond style");

  // Check for angle style
  if (force->angle == nullptr)
    error->all(FLERR, "Fix treadmilling requires an angle style");

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

  if (!has_creation_time) 
    error->all(FLERR, Error::NOLASTLINE,
               "Fix {} could not find any creation/birth time property. Aborting... Please revise!", style);

  // Ensure neighbor list is current for ghost atoms
  if (lastcheck <= neighbor->lastcall) check_ghosts();

  fprintf(lmp->screen, "FIX TREADMILLING RUN: performing post_integrate operations at timestep %ld\n", update->ntimestep);

  // Force remap of atom positions to avoid PBC issues after integration drift
  domain->pbc();           // remap all atoms back into box

  // Acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm
  comm->forward_comm();

  fprintf(lmp->screen, "  Processor %d: Atom positions remapped. Neighbor list updated for ghost atoms. Now identifying filaments and performing reactions...\n", comm->me);

  // Zero global creation counts for this timestep
  natoms = 0;
  nbonds = 0;
  nangles = 0;
  natomsloc = 0;
  nbondsloc = 0;
  nanglesloc = 0;

  double dt = update->dt;
  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  // Get unique molecule IDs to iterate over filaments
  int *tag = atom->tag;
  int *molecule = atom->molecule;
  int *type = atom->type;
  double *birth_step = atom->dvector[birth_time_index];
  int *filpos = atom->ivector[filpos_index];
  double **x = atom->x;

  // Let's look for the last created particle and get its local index
  // Now scan over nlocal to find new tag and corresponding index
  int new_idx = -1;
  int own_proc = -1;
  for (int i = 0; i < nlocal; i++) {
    if (tag[i] == MAX_TAG) {
      new_idx = i;
      own_proc = comm->me;
      break;
    }
  }
  fprintf(lmp->screen, "  Last created particle has tag %d and local index %d on processor %d - filpos = %d, birth_step = %f\n", MAX_TAG, new_idx, own_proc, filpos[new_idx], birth_step[new_idx]);
  
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
      fprintf(lmp->screen, "      Filament with molecule ID %d found on processor %d for the first time. Adding to map with head_idx = tail_idx = subhead_idx = -1.\n", m, comm->me);
      it = filament_bounds.emplace(m, filbounds{-1, -1, -1}).first;
    }
    // Set index when special non-bulk particle found
    // Non-bulk particle filpos legend:
    //   - Head: 1 always
    //   - Subhead: 2 if pure subhead, 4 if subhead of dimer (also tail)
    //   - Tail: 3 if pure tail, 4 if tail of dimer (also subhead)
    if (k == 1) it->second.head_idx  = i;                   // head index
    if (k == 1) fprintf(lmp->screen, "        Found head particle with tag %d (idx %d) for filament with molecule ID %d on processor %d\n", tag[i], i, m, comm->me);
    if (k == 2 || k == 4) it->second.subhead_idx = i;       // subhead index
    if (k == 2 || k == 4) fprintf(lmp->screen, "        Found subhead particle with tag %d (idx %d) for filament with molecule ID %d on processor %d\n", tag[i], i, m, comm->me);
    if (k == 3 || k == 4) it->second.tail_idx = i;          // tail index
    if (k == 3 || k == 4) fprintf(lmp->screen, "        Found tail particle with tag %d (idx %d) for filament with molecule ID %d on processor %d\n", tag[i], i, m, comm->me);
  }

  // For each filament (molecule ID), attempt growth and shrinkage events 
  //   at head and tail respectively with probabilities based on rates and timestep
  for (auto& [mol_id, bounds] : filament_bounds) {
    auto hid = bounds.head_idx;
    auto tid = bounds.tail_idx;
    auto shid = bounds.subhead_idx;

    fprintf(lmp->screen, "Evaluating filament with molecule ID %d on processor %d with head idx %d (tag %d), tail idx %d (tag %d) and subhead idx %d (tag %d)\n", mol_id, comm->me, hid, tag[hid], tid, tag[tid], shid, tag[shid]);
    
    // Growth event
    if (r_on > 0) {
      fprintf(lmp->screen, "Evaluating growth for filament %d (head tag %d (idx %d), tail tag %d (idx %d), subhead tag %d (idx %d)) at timestep %ld -- p = %g\n", mol_id, tag[hid], hid, tag[tid], tid, tag[shid], shid, update->ntimestep, r_on * dt);
      if (should_happen(r_on)) {
        // Grow filament - takes care of everything: sampling new position, checking overlaps, creating particle, bonds and angles, and updating head and subhead flags
        fprintf(lmp->screen, "  Attempting growth for filament %d (head tag %d (idx %d), tail tag %d (idx %d), subhead tag %d (idx %d)) at timestep %ld\n", mol_id, tag[hid], hid, tag[tid], tid, tag[shid], shid, update->ntimestep);
        grow_filament(mol_id, hid, shid);
      }
    }

    // Shrinkage event
    if (r_off_base > 0.0) {
      double tage = (update->ntimestep - birth_step[tid]) * dt;
      double shrink_rate = compute_shrinkage_rate(tage);
      if (should_happen(shrink_rate)) {
        // Shrink filament -- takes care of everything: deleting particles, bonds and angles and updating flags
        shrink_filament(tid);
      }
    }
  } // end of filaments loop

  // Nucleation of new filaments
  if (r_nuc > 0.0) {
    if (should_happen(r_nuc)) {
      if (nuc_mode == 1) {
        // Nucleate a new filament with random position and orientation
        // takes care of everything: creating particles and bond and setting flags
        nucleate_filament();
      } else if (nuc_mode == 2 && nucleator_type > 0) {
        // TO-DO: Implement branching
        // placeholder:
        error->all(FLERR, "Fix treadmilling: branching not implemented yet but called for! Please revise! ");
      }
    }
  } // end of nucleation if

  // Increment cumulative counters
  natomstotal += natoms;
  nbondstotal += nbonds;
  nanglestotal += nangles;
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ----------------------------------------------------------------------
   grow filament by creating new particle at head and updating bonds and angles accordingly
     - find head position (from head index)
     - sample new position near head
      - displacement from head to subhead + noise (gaussian with magnitude noise_sigma)
     - check overlaps and try again if necessary up to max_trials
     - if successful, create new particle at that position and create bond and angle with previous head
       - update head and subhead flags: new head is filpos 1, old head becomes subhead (filpos 2 if not tail, filpos 4 if also tail)
       - set new particle properties (type, molecule ID, birth time, filpos) and tags (new tag = max tag + 1)
    needs to be called by all processors - handles rank locality and communications internally
------------------------------------------------------------------------- */

void FixTreadmilling::grow_filament(tagint mol_id, int hidx, int shidx)
{
  // zero local count of created particles for this growth event
  natomsloc = 0;

  int nlocal = atom->nlocal;
  int *molecule = atom->molecule;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *filpos = atom->ivector[filpos_index];

  // Head tag
  tagint head_tag = tag[hidx];
  if (head_tag == (tagint) 140) {
    fprintf(lmp->screen, "      Head %d - hidx = %d, shid = %d\n", (int)head_tag, hidx, shidx);
  }

  // Sample new position near head
  double head_pos[3], shead_pos[3], new_pos[3], wrapped[3];
  int hproc, shproc;
  imageint image = ((imageint) IMGMAX << IMG2BITS) |
                     ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // If current head owned by this processor define head position
  int hproc_loc = -1;
  if (hidx != -1) {
    if (is_local(x[hidx])) {
      head_pos[0] = x[hidx][0]; head_pos[1] = x[hidx][1]; head_pos[2] = x[hidx][2];
      hproc_loc = comm->me;
      fprintf(lmp->screen, "    Filament %d head found on processor %d at index %d with position (%.3f, %.3f, %.3f)\n", mol_id, comm->me, hidx, head_pos[0], head_pos[1], head_pos[2]);
    }
  }
  // Broadcast to all other processors
  MPI_Allreduce(&hproc_loc, &hproc, 1, MPI_INT, MPI_MAX, world);
  if (hproc == -1) {error->all(FLERR, "Fix treadmilling: head particle not found on any processor! Aborting! ");}
  MPI_Bcast(head_pos, 3, MPI_DOUBLE, hproc, world);

  // If current subhead owned by this processor define subhead position
  int shproc_loc = -1;
  if (shidx != -1) {
    if (is_local(x[shidx])) {
      shead_pos[0] = x[shidx][0]; shead_pos[1] = x[shidx][1]; shead_pos[2] = x[shidx][2];
      shproc_loc = comm->me;
      fprintf(lmp->screen, "    Filament %d subhead found on processor %d at index %d with position (%.3f, %.3f, %.3f)\n", mol_id, comm->me, shidx, shead_pos[0], shead_pos[1], shead_pos[2]);
    }
  }
  // Broadcast to all other processors
  MPI_Allreduce(&shproc_loc, &shproc, 1, MPI_INT, MPI_MAX, world);
  if (shproc == -1) {error->all(FLERR, "Fix treadmilling: subhead particle not found on any processor! Aborting! ");}
  MPI_Bcast(shead_pos, 3, MPI_DOUBLE, shproc, world);

  // Sample new position
  int trials = 0;
  bool placed = false;
  fprintf(lmp->screen, "  Sampling new position for growth of filament %d at timestep %ld -- head pos (%.3f, %.3f, %.3f), subhead pos (%.3f, %.3f, %.3f)\n", mol_id, update->ntimestep, head_pos[0], head_pos[1], head_pos[2], shead_pos[0], shead_pos[1], shead_pos[2]);
  // Displacement
  double dx = head_pos[0] - shead_pos[0];
  double dy = head_pos[1] - shead_pos[1];
  double dz = head_pos[2] - shead_pos[2];
  if (domain->dimension == 2) dz = 0.0;
  // Displacement PBCs
  domain->minimum_image(FLERR, dx, dy, dz);
  // if (dx > 0.5*(domain->boxhi[0]-domain->boxlo[0])) dx -= domain->boxhi[0]-domain->boxlo[0];
  // if (dx < -0.5*(domain->boxhi[0]-domain->boxlo[0])) dx += domain->boxhi[0]-domain->boxlo[0];
  // if (dy > 0.5*(domain->boxhi[1]-domain->boxlo[1])) dy -= domain->boxhi[1]-domain->boxlo[1];
  // if (dy < -0.5*(domain->boxhi[1]-domain->boxlo[1])) dy += domain->boxhi[1]-domain->boxlo[1];
  // if (dz > 0.5*(domain->boxhi[2]-domain->boxlo[2])) dz -= domain->boxhi[2]-domain->boxlo[2];
  // if (dz < -0.5*(domain->boxhi[2]-domain->boxlo[2])) dz += domain->boxhi[2]-domain->boxlo[2];
  // Normalize and scale by sigma
  double len = sqrt(dx*dx+dy*dy+dz*dz);
  if (len > 1e-10) {dx /= len; dy /= len; dz /= len;}
  else {
      // fallback: random unit vector if head and shead coincide
      dx = random->gaussian();
      dy = random->gaussian();
      dz = random->gaussian();
      if (domain->dimension == 2) dz = 0.0;
      len = sqrt(dx*dx+dy*dy+dz*dz);
      dx /= len; dy /= len; dz /= len;
  }
  dx *= sigma;
  dy *= sigma;
  dz *= sigma;
  fprintf(lmp->screen, "  Displacement for new head position for growth of filament %d at timestep %ld: (%.3f, %.3f, %.3f)\n", mol_id, update->ntimestep, dx, dy, dz);
  fprintf(lmp->screen, "  No noise position: (%.3f, %.3f, %.3f)\n", head_pos[0]+dx, head_pos[1]+dy, head_pos[2]+dz);
  while (!placed && trials < max_trials) {
    // Only sample new position on main thread (rank 0)
    if (comm->me == 0) {
      // New position
      new_pos[0] = head_pos[0] + dx + noise_sigma * random->gaussian();
      new_pos[1] = head_pos[1] + dy + noise_sigma * random->gaussian();
      if (domain->dimension == 2) new_pos[2] = head_pos[2];
      else {
        new_pos[2] = head_pos[2] + dz + noise_sigma * random->gaussian();
      }
      // CHANGING TO NO MANUAL WRAPPING HERE!
      // Check PBCs
      // domain->remap(new_pos);
      // if (new_pos[0] < domain->boxlo[0]) new_pos[0] += domain->boxhi[0] - domain->boxlo[0];
      // if (new_pos[0] >= domain->boxhi[0]) new_pos[0] -= domain->boxhi[0] - domain->boxlo[0];
      // if (new_pos[1] < domain->boxlo[1]) new_pos[1] += domain->boxhi[1] - domain->boxlo[1];
      // if (new_pos[1] >= domain->boxhi[1]) new_pos[1] -= domain->boxhi[1] - domain->boxlo[1];
      // if (new_pos[2] < domain->boxlo[2]) new_pos[2] += domain->boxhi[2] - domain->boxlo[2];
      // if (new_pos[2] >= domain->boxhi[2]) new_pos[2] -= domain->boxhi[2] - domain->boxlo[2];
    }
    // Broadcast new position to all threads
    MPI_Bcast(new_pos, 3, MPI_DOUBLE, 0, world);
    // WRAP NEW POSITION AND UPDATE PBC IMAGES BEFORE CHECKING OVERLAPS
    // Compute image flags and wrapped position before creating atom
    wrapped[0] = new_pos[0]; wrapped[1] = new_pos[1]; wrapped[2] = new_pos[2];
    domain->remap(wrapped, image);  // wrapped gets the in-box coords, image gets the offset
    // // Check overlaps
    // if (!check_overlap(new_pos)) {placed=true;}
    // Check overlaps WITH WRAPPED POSITION (i.e. where atom will actually be created after PBC handling) -- this is important to avoid creating overlapping atoms across PBC boundaries
    if (!check_overlap(wrapped)) {placed=true;}
    trials++;
  }
  fprintf(lmp->screen, "  Sampled new position (%.3f, %.3f, %.3f) for growth of filament %d at timestep %ld after %d trials\n", new_pos[0], new_pos[1], new_pos[2], mol_id, update->ntimestep, trials);
  fprintf(lmp->screen, "  Sampled wrapped new position (%.3f, %.3f, %.3f) (image: %ld) for growth of filament %d at timestep %ld after %d trials\n", wrapped[0], wrapped[1], wrapped[2], image, mol_id, update->ntimestep, trials);
  
  if (!placed && comm->me == 0) {      // return early if failed placement
    error->warning(FLERR, "Fix treadmilling: failed particle placement in filament growth. ");
    return;
  }

  // Update head and subhead flags
    // Non-bulk particle filpos legend:
    //   - Head: 1 always
    //   - Subhead: 2 if pure subhead, 4 if subhead of dimer (also tail)
    //   - Tail: 3 if pure tail, 4 if tail of dimer (also subhead)
  // Head becomes subhead: 1 (always) -> 2 (becomes pure subhead)
  if (filpos[hidx] != 1) {error->all(FLERR, "Fix treadmilling: head particle filpos value is not 1! Aborting! ");}
  filpos[hidx] = 2;
  // Subhead becomes either bulk (pure subhead case) - 2->0 - or pure tail (if was subhead of dimer, also tail) - 4->3
  if (filpos[shidx] == 2) {
    // Pure subhead becomes bulk
    filpos[shidx] = 0;
  } else if (filpos[shidx] == 4) {
    // Subhead of dimer becomes tail of dimer
    filpos[shidx] = 3;
  }
  else {
    error->all(FLERR, "Fix treadmilling: subhead particle filpos value is not 2 or 4! Aborting! ");
  }
  comm->forward_comm();  // propagate filpos to ghosts before proceeding

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
  // if (is_local(new_pos)) {
  // USE WRAPPED POSITION FOR LOCALITY CHECK AND PARTICLE CREATION
  if (is_local(wrapped)) {
    fprintf(lmp->screen, "    Creating new head for filament %d at timestep %ld with tag %d at position (%.3f, %.3f, %.3f) on processor %d\n", mol_id, update->ntimestep, new_tag, wrapped[0], wrapped[1], wrapped[2], comm->me);
    create_particle(wrapped, new_tag, ptype, mol_id, 1, image);  // use wrapped position for creation, filpos = 1 for new head, pass image flags
    // imageint image_local = ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;
    // create_particle(wrapped, new_tag, ptype, mol_id, 1, image_local);  // use wrapped position for creation, filpos = 1 for new head, pass image flags
    // Increment local count of created particles
    natomsloc++;
  }
  // MPI reduce created particle count, check correct number created and reset local count for next event
  MPI_Allreduce(&natomsloc, &natoms, 1, MPI_INT, MPI_SUM, world);
  fprintf(lmp->screen, "  Created %d new particles for growth of filament %d at timestep %ld\n", natoms, mol_id, update->ntimestep);
  if (natoms > 1 && comm->me == 0) {error->warning(FLERR, "Created more than one particle in fix treadmilling grow_filament! Revise implementation!!");}
  atom->natoms += natoms;         // update global atom count in atom class -- necessary for correct neighbor list rebuild and communication handling
  fprintf(lmp->screen, "  Updated global atom count to %d after growth of filament %d at timestep %ld\n", atom->natoms, mol_id, update->ntimestep);
  natomsloc = 0;

  fprintf(lmp->screen, "    Created new particle with tag %d at position (%.3f, %.3f, %.3f) on processor %d with filpos = %d and birth time = %f\n", atom->tag[nlocal], atom->x[nlocal][0], atom->x[nlocal][1], atom->x[nlocal][2], comm->me, atom->ivector[filpos_index][nlocal], has_creation_time ? atom->dvector[birth_time_index][nlocal] : -1.0);

  // Forward communicate new particle position for bond creation
  comm->forward_comm(this);

  fprintf(lmp->screen, "  Comm forwarded for growth of filament %d at timestep %ld\n", mol_id, update->ntimestep);

  fprintf(lmp->screen, "    Created new particle with tag %d at position (%.3f, %.3f, %.3f) on processor %d with filpos = %d and birth time = %f\n", atom->tag[nlocal], atom->x[nlocal][0], atom->x[nlocal][1], atom->x[nlocal][2], comm->me, atom->ivector[filpos_index][nlocal], has_creation_time ? atom->dvector[birth_time_index][nlocal] : -1.0);

  // Rebuild neighbor list for bond creation
  neighbor->build_one(list);

  fprintf(lmp->screen, "  Rebuilt neighbor list for growth of filament %d at timestep %ld\n", mol_id, update->ntimestep);

  fprintf(lmp->screen, "    Created new particle with tag %d at position (%.3f, %.3f, %.3f) on processor %d with filpos = %d and birth time = %f\n", atom->tag[nlocal], atom->x[nlocal][0], atom->x[nlocal][1], atom->x[nlocal][2], comm->me, atom->ivector[filpos_index][nlocal], has_creation_time ? atom->dvector[birth_time_index][nlocal] : -1.0);

  // Let's look for the new particle and get its local index
  // First reset new nlocl and tags
  nlocal = atom->nlocal;
  tag = atom->tag;
  filpos = atom->ivector[filpos_index];
  double *birth_step = atom->dvector[birth_time_index];
  // Now scan over nlocal to find new tag and corresponding index
  int new_idx = -1;
  int own_proc = -1;
  for (int i = 0; i < nlocal; i++) {
    if (tag[i] == new_tag) {
      new_idx = i;
      own_proc = comm->me;
      break;
    }
  }
  fprintf(lmp->screen, "  New particle for growth of filament %d at timestep %ld has tag %d and local index %d on processor %d -- filpos = %d, birth_step = %f\n", mol_id, update->ntimestep, new_tag, new_idx, own_proc, filpos[new_idx], birth_step[new_idx]);

  // Create bonds (handles creation of angles too)
  create_bond(new_tag, head_tag, btype);
}

/* ---------------------------------------------------------------------- */

void FixTreadmilling::shrink_filament(int tail_idx)
{
  int *filpos = atom->ivector[filpos_index];
  // Delete bonds involving tail_idx (should be a single one) and return the index of the bound particle (new tail)
  int stidx = delete_bonds(tail_idx);
  if (stidx == -1) {error->all(FLERR, "Fix treadmilling: invalid bound particle index upon tail depolymerisation. ");}
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

/* ----------------------------------------------------------------------
   nucleate filament by creating two new particles at head and updating bonds and angles accordingly
     - sample new random tail position, random orientation and corresponding head position
      - uniform sampling of the box
     - check overlaps and try again if necessary up to max_trials
     - if successful, create new particles at those positions and create bond between them
    needs to be called by all processors - handles rank locality and communications internally
------------------------------------------------------------------------- */

void FixTreadmilling::nucleate_filament()
{
  // Relevant variables
  double pos1[3], pos2[3];
  int trials = 0;
  int placed = 0;
  int nlocal = atom->nlocal;
  tagint new_mol_id = 0;

  imageint image1 = ((imageint) IMGMAX << IMG2BITS) |
                     ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  imageint image2 = ((imageint) IMGMAX << IMG2BITS) |
                     ((imageint) IMGMAX << IMGBITS) | IMGMAX;

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
      domain->remap(pos1, image1);
      domain->remap(pos2, image2);
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
  if (placed!=1) {
    if (comm->me == 0) {
      error->warning(FLERR, "Fix treadmilling: failed particle placement in filament nucleation. ");
    }
    return;
  }

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
  if (is_local(pos1)) {
    // If rank owns position of tail it creates tail
    int tail_fp = 4;                                                    // tail has filpos = 4 in dimer
    create_particle(pos1, new_tag_1, ptype, new_mol_id, tail_fp, image1);
    natomsloc++;                                                        // increment local count of created particles
  }
  if (is_local(pos2)) {
    // If rank owns position of head it creates head
    int head_fp = 1;                                                    // head has filpos = 1 always, including in dimer
    create_particle(pos2, new_tag_2, ptype, new_mol_id, head_fp, image2);
    natomsloc++;                                                        // increment local count of created particles
  }

  // MPI reduce created particle count, check correct number created and reset local count for next event
  MPI_Allreduce(&natomsloc, &natoms, 1, MPI_INT, MPI_SUM, world);
  if (natoms != 2 && comm->me == 0) {error->warning(FLERR, "Did not create exactly two particles in fix treadmilling nucleate_filament! Revise implementation!!");}
  natomsloc = 0;

  // Forward communicate new particle position for bond creation
  comm->forward_comm(this);

  // Rebuild neighbor list for bond creation
  neighbor->build_one(list);

  // Create dimer bond (handles creation of angles too, although there are no angles in a dimer)
  create_bond(new_tag_1, new_tag_2, btype);
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
      if (!check_overlap(pos1) && !check_overlap(pos2)) {
        placed = true;
      }
    }
    trials++;
  }

  if (placed) {
    int nlocal = atom->nlocal;
    tagint mol_id = ++mol_counter;  // Simple molecule ID counter
    
    // create_particle(pos1, mol_id, ptype, mol_id, 1);
    // create_particle(pos2, mol_id, ptype, mol_id, 1);
    // create_bond(nlocal, nlocal + 1, btype);
  }
}

/* ---------------------------------------------------------------------- 
  check if a proposed position overlaps with any existing particle based on overlap_cut distance
    each processor checks against its local and ghost particles and returns a flag
    local flags are reduced with MPI_MAX to determine if any processor detected an overlap
    returns true if overlap detected, false if safe to place -- global result returned to all processors
------------------------------------------------------------------------- */

bool FixTreadmilling::check_overlap(double *pos)
{
  double **x = atom->x;
  // int nlocal = atom->nlocal;
  int ntotal = atom->nlocal + atom->nghost;
  int flag = 0;
  double delx, dely, delz, rsq;
  int i;
  // for (i = 0; i < nlocal; i++) {
  for (i = 0; i < ntotal; i++) {
    delx = pos[0] - x[i][0];
    dely = pos[1] - x[i][1];
    delz = pos[2] - x[i][2];
    domain->minimum_image(FLERR, delx, dely, delz);
    rsq = delx*delx + dely*dely + delz*delz;
    if (rsq < overlap_sq) flag = 1;
  }
  int flagall = 0;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);
  return flagall > 0;  // true = overlap detected, false = safe to place
}

/* ---------------------------------------------------------------------- 
  check if a rate-controlled event occurs based on the rate and timestep
    returns true if event occurs, false otherwise
    uses random number generator to determine outcome
    only run RNG on one processor (rank 0) 
    broadcast result to ensure consistency across ranks
------------------------------------------------------------------------- */

bool FixTreadmilling::should_happen(double rate)
{
  int should = 0;
  if (comm->me == 0) {
    double prob = rate * update->dt;
    if (random->uniform() < prob) {
      should = 1;
    }
  }
  MPI_Bcast(&should, 1, MPI_INT, 0, world);
  return should == 1;
}

/* ----------------------------------------------------------------------
   insert a new particle at position pos with given tag, type, molecule id, and filpos
    uses atom->avec->create_atom to insert
    sets tag, mol_id and mask
    sets velocity from Boltzmann distribution
    sets forces to zero
    sets the image flag to zero (remapped, standard procedure)
    sets special properties:
      fp given when calling function
      birth time = current timestep (update->ntimestep)
    updates atom map at the end so new atom is findable immediately
------------------------------------------------------------------------- */

void FixTreadmilling::create_particle(double *pos, tagint tag, int type_id, tagint mol_id, int fp, imageint image)
{
  int nlocal = atom->nlocal;
  
  atom->avec->create_atom(type_id, pos);
  
  // Set atom properties
  atom->tag[nlocal]       = tag;
  atom->molecule[nlocal]  = mol_id; // Set molecule id - same as filament
  atom->mask[nlocal]      = groupbit;
  atom->image[nlocal]     = image;
  
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

  // Zero the force on the new atom
  atom->f[nlocal][0] = 0.0;
  atom->f[nlocal][1] = 0.0;
  atom->f[nlocal][2] = 0.0;

  // Per-atom custom properties
  if (has_creation_time) {atom->dvector[birth_time_index][nlocal] = update->ntimestep;}
  atom->ivector[filpos_index][nlocal] = fp;

  fprintf(lmp->screen, "    Created new particle with tag %d at position (%.3f, %.3f, %.3f) on processor %d with filpos = %d and birth time = %f\n", tag, pos[0], pos[1], pos[2], comm->me, atom->ivector[filpos_index][nlocal], has_creation_time ? atom->dvector[birth_time_index][nlocal] : -1.0);

  // Update atom map so new tag is findable immediately
  if (atom->map_style != Atom::MAP_NONE) {atom->map_one(tag, nlocal);}

  fprintf(lmp->screen, "    Created new particle with tag %d at position (%.3f, %.3f, %.3f) on processor %d with filpos = %d and birth time = %f\n", tag, pos[0], pos[1], pos[2], comm->me, atom->ivector[filpos_index][nlocal], has_creation_time ? atom->dvector[birth_time_index][nlocal] : -1.0);

  // // Set neighbour list flag to indicate new atom added (forces rebuild of neighbor list before next force calculation)
  // neighbor->set_dirty(ACTION_NEIGH);
  // neighbor->set_dirty(ACTION_SPECIAL);
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

/* ----------------------------------------------------------------------
   create a new bond between atoms with given tags and bond type
    sets up bond_atom and bond_type arrays for the two atoms involved, consistent with newton_bond setting
    updates special neighbor lists for the two atoms involved to add 1-2 neighbor relationship
    forward comm all specials to ensure ghost atoms have updated bond info
    updates topology (like in fix_bond_create) to add new bond and any resulting angles
    updates new bond counts and checks for consistency
------------------------------------------------------------------------- */

void FixTreadmilling::create_bond(tagint tagi, tagint tagj, int btype)
{
  // reset local bond count for this creation event
  nbondsloc = 0;

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
  int *num_bond = atom->num_bond;
  int newton_bond = force->newton_bond;
  
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
    nbondsloc++;                                    // only count bond creation on owning processor to avoid double counting when newton_bond=0
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
    if (n3 >= atom->maxspecial) {error->one(FLERR, "Special neighbor list overflow in fix treadmilling");}
    
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
    if (n3 >= atom->maxspecial) {error->one(FLERR, "Special neighbor list overflow in fix treadmilling");}
    
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

  // Update topology (like in fix_bond_create but for a single bond)
  // this will handle any necessary updates to angles as well as special neighbor lists beyond 1-2
  update_topology(tag1, tag2); 
  
  // Safely update global bond count via MPI reduction, then reset local count for next event
  MPI_Allreduce(&nbondsloc, &nbonds, 1, MPI_INT, MPI_SUM, world);
  if (nbonds > 1) {error->warning(FLERR, "Created more than one bond in fix treadmilling create_bond! Revise implementation!!");}
  atom->nbonds += nbonds;
  nbondsloc = 0;

  // Trigger reneigbouring if any bonds were formed
  if (nbonds > 0) next_reneighbor = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

int FixTreadmilling::delete_bonds(int idx)
{
  // Delete all bonds involving particle idx
  // This requires managing the bond list - CAREFUL!
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

/* ----------------------------------------------------------------------
  compute the lifetime-dependent shrinking rate of a filament
    r_off = r_off_base * (1 - exp(-rhyd * lifetime))
      where lifetime is the lifetime of the tail monomer: (current step - birth step) * dt
    returns 0 if invalid rhyd (<=0)
------------------------------------------------------------------------- */

double FixTreadmilling::compute_shrinkage_rate(double lifetime)
{
  if (r_hyd <= 0.0) return 0.0;
  return r_off_base * (1.0 - exp(-r_hyd * lifetime));
}

/* ----------------------------------------------------------------------
   ensure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly created
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is added
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
  STRAIGHT FROM fix_bond_create.cpp
------------------------------------------------------------------------- */

void FixTreadmilling::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall)
    error->all(FLERR, Error::NOLASTLINE, "Fix {} needs ghost atoms from further away", style);
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
   loop over my atoms and compare with created bond endpoints to determine if special list needs to be rebuilt
   influenced = 1 if atom's topology is affected by the created bond between id1 and id2, which means:
     yes if is one of 2 atoms in bond (id1 or id2)
     yes if either atom ID appears in as 1-2 or 1-3 in atom's special list
     else no
   if influenced, then:
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
     check for angles to create due modified special list
   this is done on all processors for all atoms, but only the influenced atoms will have their topology updated
   nangles is updated locally and then reduced globally at the end to update global angle count
     if newton_bond=0, each angle is stored on 3 atoms instead of 1, so divide by 3 at the end to avoid double counting
   also check for overflow of special list and angle list and set overflow flag if necessary

  ADAPTED FROM fix_bond_create.cpp
------------------------------------------------------------------------- */

void FixTreadmilling::update_topology(tagint id1, tagint id2)         // by convention in create_bond, id1 < id2 -- so id1 is central atom for angle creation and id2 is the new neighbor being added
{
  int i,j,k,n,influence,influenced;                                   // influence = does this atom "touch" a newly created bond?; influenced = does this atom touch any newly created bond?
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  nangles = 0;
  overflow = 0;

  for (i = 0; i < nlocal; i++) {
    influenced = 0;                                                   // reset per atom i, before looping over all new bonds
    slist = special[i];

    influence = 0;
    // compute influence of bond j on atom i
    if (tag[i] == id1 || tag[i] == id2) influenced = 1;               // atom i IS one of the bond endpoints (id1 or id2)
    else {
      n = nspecial[i][1];                                             // number of 1-3 neighbors (angle partners)
      for (k = 0; k < n; k++)
        if (slist[k] == id1 || slist[k] == id2) {                     // atom i is a 1-3 neighbor of a bond endpoint (id1 or id2) -> it's influenced by the bond creation
          influenced = 1;
          break;
        }
    }

    // rebuild_special_one() first, since used by create_angles, etc

    if (influenced) {
      rebuild_special_one(i);                                         // update 1-2/1-3/1-4 neighbor list
      create_angle(i, id1, id2);                                      // create new angles involving atom i, knowing that id1 is the central atom and id2 is the new neighbor being added
    }
  }

  int overflowall;
  MPI_Allreduce(&overflow,&overflowall,1,MPI_INT,MPI_SUM,world);
  if (overflowall)
    error->all(FLERR, Error::NOLASTLINE,
               "Fix {} induced too many angles per atom", style);

  // Safely update global angle count via MPI reduction and reset local count for next event
  MPI_Allreduce(&nanglesloc,&nangles,1,MPI_INT,MPI_SUM,world);
  if (!force->newton_bond) nangles /= 3;
  atom->nangles += nangles;
  nanglesloc = 0;
}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs

  ADAPTED FROM fix_bond_create.cpp
------------------------------------------------------------------------- */

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
      error->one(FLERR, Error::NOLASTLINE, "Fix {} needs ghost atoms from further away", style);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR, Error::NOLASTLINE, "Special list size exceeded in fix {}", style);

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR, Error::NOLASTLINE, "Fix {} needs ghost atoms from further away", style);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR, Error::NOLASTLINE, "Special list size exceeded in fix {}", style);

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates

  ADAPTED FROM fix_bond_create.cpp
------------------------------------------------------------------------- */

int FixTreadmilling::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}

/* ----------------------------------------------------------------------
   create any angles owned by atom M induced by newly created bonds
   walk special list to find all possible angles to create
   only add an angle if a new bond is one of its 2 bonds (I-J,J-K) -- IDENTIFIED BY ID1 and ID3
   only add an angle if atom IDs match given IDs (id1 and id3)
   for newton_bond on, atom M is central atom
   for newton_bond off, atom M is any of 3 atoms in angle

  ADAPTED FROM fix_bond_create.cpp
------------------------------------------------------------------------- */

void FixTreadmilling::create_angle(int m, tagint id1, tagint id2)
{
  // id1 and id2 are two bond atom tags -- id1 is central angle atom and id2 is new neighbor being added 
  //   so only create angles where id1 is central atom and id2 is one of the outer atoms, 
  //   which ensures we only create angles that are actually new due to the new bond creation, 
  //   and we don't double count angles when looping over both bond endpoints in update_topology

  int i,j,n,i2local,n1,n2;
  tagint i1,i2,i3;
  tagint *s1list,*s2list;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int num_angle = atom->num_angle[m];
  int *angle_type = atom->angle_type[m];
  tagint *angle_atom1 = atom->angle_atom1[m];
  tagint *angle_atom2 = atom->angle_atom2[m];
  tagint *angle_atom3 = atom->angle_atom3[m];

  // atom M is central atom in angle -- means its tag should match id1, and one of its 1-2 neighbors should match id2
  // double loop over 1-2 neighs
  // avoid double counting by 2nd loop as j = i+1,N not j = 1,N
  // consider all angles, only add if:
  //   IDs match given IDs (id1 and id3)

  i2 = tag[m];
  n2 = nspecial[m][0];
  s2list = special[m];

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];
    for (j = i+1; j < n2; j++) {
      i3 = s2list[j];

      // angle = i1-i2-i3
      // check i2 = id1 and i3 = id2
      if (i2 != id1 || i3 != id2) continue;

      // NOTE: this is place to check atom types of i1,i2,i3 - but we don't care about atom types so we don't check

      if (num_angle < atom->angle_per_atom) {
        angle_type[num_angle] = atype;
        angle_atom1[num_angle] = i1;
        angle_atom2[num_angle] = i2;
        angle_atom3[num_angle] = i3;
        num_angle++;
        nanglesloc++;                               // increment local angles count (function is called locally)
      } else overflow = 1;
    }
  }

  atom->num_angle[m] = num_angle;
  if (force->newton_bond) return;

  // for newton_bond off, also consider atom M as atom 1 in angle
  // then its two 1-2 neighbors are atoms 2 and 3 in angle, and we check if either of them is id1 and the other is id2

  i1 = tag[m];
  n1 = nspecial[m][0];
  s1list = special[m];

  for (i = 0; i < n1; i++) {
    i2 = s1list[i];
    i2local = atom->map(i2);
    if (i2local < 0)
      error->one(FLERR, Error::NOLASTLINE, "Fix {} needs ghost atoms from further away", style);
    s2list = special[i2local];
    n2 = nspecial[i2local][0];

    for (j = 0; j < n2; j++) {
      i3 = s2list[j];
      if (i3 == i1) continue;

      // angle = i1-i2-i3
      // check i2 = id1 and i3 = id2
      if (i2 != id1 || i3 != id2) continue;

      // NOTE: this is place to check atom types of i1,i2,i3 - but we don't care about atom types so we don't check

      if (num_angle < atom->angle_per_atom) {
        angle_type[num_angle] = atype;
        angle_atom1[num_angle] = i1;
        angle_atom2[num_angle] = i2;
        angle_atom3[num_angle] = i3;
        num_angle++;
        nanglesloc++;                               // increment local angles count (function is called locally)
      } else overflow = 1;
    }
  }

  atom->num_angle[m] = num_angle;
}

/* ---------------------------------------------------------------------- */

bool FixTreadmilling::is_local(double *pos)
{
    return (pos[0] >= domain->sublo[0] && pos[0] < domain->subhi[0] &&
            pos[1] >= domain->sublo[1] && pos[1] < domain->subhi[1] &&
            pos[2] >= domain->sublo[2] && pos[2] < domain->subhi[2]);
}

/* ---------------------------------------------------------------------- */

// Adapted from fix_bond_create.cpp -- forward comm all specials so ghost atoms have updated bond info
int FixTreadmilling::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,ns;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int *filpos      = atom->ivector[filpos_index];
  double *birth    = has_creation_time ? atom->dvector[birth_time_index] : nullptr;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
    // buf[m++] = ubuf(filpos[j]).d;
    // if (has_creation_time) buf[m++] = ubuf(birth[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

// Adapted from fix_bond_create.cpp -- forward comm all specials so ghost atoms have updated bond info
void FixTreadmilling::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int *filpos      = atom->ivector[filpos_index];
  double *birth    = has_creation_time ? atom->dvector[birth_time_index] : nullptr;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    ns = (int) ubuf(buf[m++]).i;
    nspecial[i][0] = ns;
    for (j = 0; j < ns; j++)
      special[i][j] = (tagint) ubuf(buf[m++]).i;
    // filpos[i] = (int) ubuf(buf[m++]).i;
    // if (has_creation_time) birth[i] = ubuf(buf[m++]).d;
  }
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
  // bytes += (double)(atom->nmax * 2) * sizeof(int);  // nlocalkeep, nghostlykeep
  return bytes;
}