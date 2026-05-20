# Turnover + Intermittent Interactions (TOII) Filaments

TOII Filaments are introduced here as a minimal "active nematic" system that captures the two essential activity modes of the cell cytoskeleton: turnover of components and intermittent crosslinking by partner molecules.

We consider bead-spring polymers, a classical minimal model for filaments that exhibit the following important features:
- They have well-defined constituent subunits (monomers): the beads. This in turn allows to easily implement two essential features:
	- Transient *local* interactions: single monomers can switch between attractive and purely repulsive interactions, capturing the local cross-linking of filaments in the cytoskeleton, rather than the whole filament being attractive or repulsive.
	- Well-defined turnover kinetics: single monomers can be added or removed from an individual filament, thus enabling the implementation of particular polymerisation kinetics (such as treadmilling), rather than general filament turnover where complete filaments are removed and nucleated.
- They have well-defined stretching stiffness, implemented via the harmonic springs keeping the beads together, which has tunable strength.
- They have well-defined bending stiffness, implemented via the harmonic angle potentials acting on every connected triad of beads, which has tunable strength.

...
