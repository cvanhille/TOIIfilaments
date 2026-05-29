/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef FIX_CLASS
// clang-format off
FixStyle(treadmilling,FixTreadmilling);
// clang-format on
#else

#ifndef LMP_FIX_TREADMILLING_H
#define LMP_FIX_TREADMILLING_H

#include "fix.h"

namespace LAMMPS_NS {

class FixTreadmilling : public Fix {
 public:
  FixTreadmilling(class LAMMPS *, int, char **);
  ~FixTreadmilling() override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void post_integrate() override;
  void post_integrate_respa(int, int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override; // From FixBondCreate -- keep?
  double memory_usage() override;

 protected:
  int ptype, btype, atype;                          // particle type, bond type and angle type to create

  int seed;                                        // RNG seed

  int overflow;                                    // flag set if new bonds/angles overflow the per-atom storage
  tagint lastcheck;                                // timestep of last ghost-distance check; used by check_ghosts()

  tagint *copy;                                    // temporary buffer used when rebuilding an atom’s special neighbor list

  class RanMars *random;                           // RNG object
  class NeighList *list;                           // pointer to the neighbor list assigned via init_list() - needed for neighbor-list callbacks

  int nlevels_respa;                               // number of RESPA levels if integrator is RESPA
  
  int natoms, nbonds, nangles;                     // number of atoms, bonds and angles created in this timestep
  int natomsloc, nbondsloc, nanglesloc;            // number of atoms, bonds and angles created in this timestep -- local count for this processor
  int natomstotal, nbondstotal, nanglestotal;      // cumulative number of atoms, bonds and angles created

  void check_ghosts();                             // ensure ghost atoms have correct partner info after communication -- adapted from fix_bond_create.cpp
  void update_topology(tagint, tagint);            // check local atoms influence by new bond creation and update special lists accordingly -- adapted from fix_bond_create.cpp 
  void rebuild_special_one(int);                   // re-build special list of atom i -- called by update_topology -- adapted from fix_bond_create.cpp
  void create_angle(int, tagint, tagint);          // create any angles owned by atom i induced by newly created bonds (involving i1 and i3) -- adapted from fix_bond_create.cpp
  int dedup(int, int, tagint *);                   // remove all ID duplicates in copy -- called by rebuild_special_one -- adapted from fix_bond_create.cpp

 private:
  // Rates and parameters
  double r_on;                                     // growth rate (particles/time)
  double r_off_base;                               // base shrinkage rate r_0 (1/time)
  double r_hyd;                                    // hydrolysis rate (1/time)
  double r_nuc;                                    // nucleation rate for new filaments (filaments/time)
  double sigma;                                    // bond length / monomer size
  double noise_sigma;                              // standard deviation of noise in placement
  int max_trials;                                  // max attempts to find non-overlapping position
  double overlap_cut;                              // cutoff distance for overlap checking
  double overlap_sq;                               // squared overlap distance

  // Filament nucleation modes
  int nuc_mode;                                    // 0=none, 1=free nucleation, 2=branched
  int nucleator_type;                              // atom type that can nucleate branches (for mode 2)

  // Per-atom properties
  int birth_time_index;                            // birth time property index -- to access birth times as `double *birth_time = atom->dvector[birth_time_index];`
  bool has_creation_time;                          // flag if creation_time property exists
  int filpos_index;                                // filament position property index -- to access filament positions as `int *filpos = atom->dvector[filpos_index];`

  // Trackers for particle and molecule IDs
  tagint MAX_TAG;
  tagint MAX_MOL;

  // Temperature
  double temp;                                                             // system temperature, for velocity initialization during creation - assumed constant and uniform for now, but could be made more complex in the future by accessing compute_temp or similar

  // ---- Growth operation entry ----
  // When a filament grows, we need:
  //   - filament_id (molecule id)
  //   - head_tag
  //   - subhead_tag
  struct GrowOp {
    tagint filament_id;
    tagint head_tag;
    tagint subhead_tag;
    GrowOp(tagint f, tagint h, tagint s)
       : filament_id(f), head_tag(h), subhead_tag(s) {}
  };

  // ---- Shrink operation entry ----
  // When a filament shrinks, we need:
  //   - filament_id (molecule id)
  //   - tail_tag
  struct ShrinkOp {
    tagint filament_id;
    tagint tail_tag;

    ShrinkOp(tagint f, tagint t)
     : filament_id(f), tail_tag(t) {}
  };

  // ---- Nucleation operation entry ----
  // When a filament nucleates, we need:
  //   - new filament_id (molecule id)
  struct NucleateOp {
    tagint filament_id;

    NucleateOp(tagint f)
      : filament_id(f) {}
  };

  std::vector<GrowOp>      grow_list;
  std::vector<ShrinkOp>    shrink_list;
  std::vector<NucleateOp>  nucleate_list;

  // Helper methods
  void grow_filament(tagint, int, int);                                    // add particle at head of filament i
  void shrink_filament(int);                                               // remove particle from tail of filament i
  void nucleate_filament();                                                // nucleate new free filament
  void nucleate_branch(int);                                               // nucleate branch from atom i
  
  bool check_overlap(double *);                                            // check if position overlaps with others (globally)
  bool should_happen(double);                                              // check if a rate-controlled event occurs based on the rate and timestep
  
  void create_particle(double *, tagint, int, tagint, int, imageint);      // create particle locally and update atom map
  void delete_particle(int);                                               // delete particle and update filament
  void create_bond(tagint, tagint, int);                                   // create bond of given type (int) between given atoms (tags)
  int delete_bonds(int);                                                   // delete bonds involving particle i, return index for bound particle (index for subtail which become new tail)
  
  double compute_shrinkage_rate(double);                                   // compute r_off(lifetime) = r_0(1-exp(-r_hyd*lifetime))

  bool is_local(double *);                                                 // check if position is local to this processor
};

}  // namespace LAMMPS_NS

#endif
#endif
