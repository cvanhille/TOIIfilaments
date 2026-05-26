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
  int ptype, btype, atype                          // particle type, bond type and angle type to create

  int seed;                                        // RNG seed

  int overflow;                                    // flag set if new bonds/angles overflow the per-atom storage
  tagint lastcheck;                                // timestep of last ghost-distance check; used by check_ghosts()

//   tagint **created;  // THIS I NEED TO KEEP IF I KEEP CREATED IN .cpp
  tagint **created_bonds;                          // array of created bond pairs ([tag_i, tag_j]) - used to update special lists and angle creation -- INSTEAD of created from fix bond/create
  int ncreated_bonds, maxcreated_bonds;            // created bonds counter (for created_bonds array) and allocated size capacity

  tagint *copy;                                    // temporary buffer used when rebuilding an atom’s special neighbor list

  class RanMars *random;                           // RNG object
  class NeighList *list;                           // pointer to the neighbor list assigned via init_list() - needed for neighbor-list callbacks

  int commflag;                                    // selects which data is packed/unpacked during communication
  int nlevels_respa;                               // number of RESPA levels if integrator is RESPA
  
  int natoms, nbonds, nangles;                     // number of atoms, bonds and angles created in this timestep
  int natomstotal, nbondstotal, nanglestotal;      // cumulative number of atoms, bonds and angles created

  void check_ghosts();                             // ensure ghost atoms have correct partner info after communication -- adapted from fix_bond_create.cpp
  void update_topology();                          // 
  void rebuild_special_one(int);
  void create_angles(int);
  void create_dihedrals(int);
  void create_impropers(int);
  int dedup(int, int, tagint *);

 private:
  // Rates and parameters
  double r_on;            // growth rate (particles/time)
  double r_off_base;      // base shrinkage rate r_0 (1/time)
  double r_hyd;           // hydrolysis rate (1/time)
  double r_nuc;           // nucleation rate for new filaments
  double sigma;           // bond length / monomer size
  double noise_sigma;     // standard deviation of noise in placement
  int max_trials;         // max attempts to find non-overlapping position
  double overlap_cut;     // cutoff distance for overlap checking
  double overlap_sq;      // squared overlap distance

  // Filament nucleation modes
  int nuc_mode;           // 0=none, 1=free nucleation, 2=branched
  int nucleator_type;     // atom type that can nucleate branches (for mode 2)

  // Per-atom properties
  int birth_time_index;            // birth time property index -- to access birth times as `double *birth_time = atom->dvector[birth_time_index];`
  bool has_creation_time;          // flag if creation_time property exists
  int filpos_index;                // filamwent position property index -- to access filament positions as `int *filpos = atom->dvector[filpos_index];`

  // Thermodynamic integration
  int respa_level;
  int nlevels_respa;

  // Trackers for particle and molecule IDs
  tagint MAX_TAG;
  tagint MAX_MOL;

  // Bool flags for event completion
  int did_nucleate;

  // Helper methods
  void grow_filament(tagint, int, int);                           // add particle at head of filament i
  void shrink_filament(int);                                      // remove particle from tail of filament i
  void nucleate_filament();                                       // nucleate new free filament
  void nucleate_branch(int);                                      // nucleate branch from atom i
  
  bool check_overlap(double *);                                   // check if position overlaps with others (globally)
  
  void create_particle(double *, tagint, int, tagint, int);       // create particle locally and update atom map
  void delete_particle(int);                                      // delete particle and update filament
  void create_bonds(int, int);                                    // create bonds between consecutive filament atoms
  int delete_bonds(int);                                          // delete bonds involving particle i, return index for bound particle (index for subtail which become new tail)
  
  double compute_shrinkage_rate(double);                          // compute r_off(lifetime) = r_0(1-exp(-r_hyd*lifetime))
};

}  // namespace LAMMPS_NS

#endif
#endif
