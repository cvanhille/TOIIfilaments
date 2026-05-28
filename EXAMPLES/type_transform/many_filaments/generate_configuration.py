import numpy as np

# -----------------------------
# Parameters
# -----------------------------
n_filaments = 500
n_beads = 9
spacing = 1.0          # bond length
box = [-100, 100]
Lbox = box[1]-box[0]
z_box = [-0.25, 0.25]
min_dist = 1.12246204831         # non-overlap check

np.random.seed(12345)

atoms = []
bonds = []
angles = []

atom_id = 1
bond_id = 1
angle_id = 1

positions = []

def random_unit_vector():
    v = np.random.normal(size=2)
    v /= np.linalg.norm(v)
    return v

def is_too_close(new_pos):
    for p in positions:
        if np.linalg.norm(new_pos - p) < min_dist:
            return True
    return False

filament_count = 0
attempts = 0

while filament_count < n_filaments and attempts < 100000:
    attempts += 1

    # random center
    center = np.array([
        np.random.uniform(*box),
        np.random.uniform(*box)
    ])

    direction = random_unit_vector()

    filament_positions = []
    ok = True

    # build filament
    for i in range(n_beads):
        pos = center + (i - n_beads//2) * spacing * direction

        # PBCs
        if pos[0] < box[0]: pos[0] += Lbox
        if pos[0] >= box[1]: pos[0] -= Lbox
        if pos[1] < box[0]: pos[1] += Lbox
        if pos[1] >= box[1]: pos[1] -= Lbox

        if is_too_close(pos):
            ok = False
            break

        filament_positions.append(pos)

    if not ok:
        continue

    # accept filament
    start_atom_id = atom_id
    type_id = 1 + (filament_count % 2)

    for i, pos in enumerate(filament_positions):
        atoms.append((atom_id, type_id, 1, *pos, 0.0))
        positions.append(pos)
        atom_id += 1

    # bonds
    for i in range(n_beads - 1):
        bonds.append((bond_id, 1, start_atom_id + i, start_atom_id + i + 1))
        bond_id += 1

    # angles
    for i in range(n_beads - 2):
        angles.append((angle_id, 1,
                       start_atom_id + i,
                       start_atom_id + i + 1,
                       start_atom_id + i + 2))
        angle_id += 1

    filament_count += 1

# -----------------------------
# Write LAMMPS data file
# -----------------------------
with open("config.in", "w") as f:
    f.write("LAMMPS data file: %d random filaments\n\n"%(n_filaments))

    f.write(f"{len(atoms)} atoms\n")
    f.write(f"{len(bonds)} bonds\n")
    f.write(f"{len(angles)} angles\n\n")

    f.write("2 atom types\n1 bond types\n1 angle types\n\n")

    f.write(f"{box[0]} {box[1]} xlo xhi\n")
    f.write(f"{box[0]} {box[1]} ylo yhi\n")
    f.write(f"{z_box[0]} {z_box[1]} zlo zhi\n\n")

    f.write("Masses\n\n")
    f.write("1 1.0\n2 1.0\n\n")

    f.write("Atoms\n\n")
    for a in atoms:
        f.write(" ".join(map(str, a)) + "\n")

    f.write("\nBonds\n\n")
    for b in bonds:
        f.write(" ".join(map(str, b)) + "\n")

    f.write("\nAngles\n\n")
    for ang in angles:
        f.write(" ".join(map(str, ang)) + "\n")

print("Generated config.in")