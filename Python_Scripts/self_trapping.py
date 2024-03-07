from lammps import lammps
import numpy as np
from scipy.spatial import cKDTree
import json

def WS_analysis(kd_ref, xyz_query):
    dist, nn_idx = kd_ref.query(xyz, k=1)
    ws = np.zeros(len(xyz_query))

    for idx in nn_idx:
        ws[int(idx)] += 1

    return ws

with open('formations.json', 'r') as file:
    cluster = json.load(file)

size = 10

alattice = 3.144221296574379

dump_folder = '../Self_Trapping'

n_helium = int(1e-1*2*size**3)

unique_tet_sites = np.array([ 
                              [0.25, 0.5, 0],
                              [0.75, 0.5, 0],
                              [0.5, 0.25, 0],
                              [0.5, 0.75, 0],
                              [0.25, 0, 0.5],
                              [0.75, 0, 0.5],
                              [0.5, 0, 0.25],
                              [0.5, 0, 0.75],                              
                              [0, 0.25, 0.5],
                              [0, 0.75, 0.5],
                              [0, 0.5, 0.25],
                              [0, 0.5, 0.75],
                            ])

tet_sites = np.array([[0.25, 0.5 ,0]])

for i in range(size):
    for j in range(size):
        for k in range(size):

            tet_sites = np.vstack([tet_sites, unique_tet_sites + np.array([i, j, k])])


tet_sites = tet_sites[1:]

potfile = 'Potentials/test.0.eam.alloy'

temp = 800

lmp = lammps()

lmp.command('# Lammps input file')

lmp.command('units metal')

lmp.command('atom_style atomic')

lmp.command('atom_modify map array sort 0 0.0')

lmp.command('boundary p p p')

lmp.command('lattice bcc %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % 
            (alattice,
            1, 0, 0,
            0, 1, 0, 
            0, 0, 1
            ) 
            )

lmp.command('region r_simbox block %f %f %f %f %f %f units lattice' % (

    -1e-9, size + 1e-9, -1e-9, size + 1e-9, -1e-9, size + 1e-9
))

lmp.command('region r_atombox block %f %f %f %f %f %f units lattice' % (

    -1e-4, size + 1e-4, -1e-4, size + 1e-4, -1e-4, size + 1e-4
))

lmp.command('create_box 3 r_simbox')

lmp.command('create_atoms 1 region r_atombox')

lmp.command('mass 1 183.84')

lmp.command('mass 2 1.00784')

lmp.command('mass 3 4.002602')

lmp.command('pair_style eam/alloy' )

lmp.command('pair_coeff * * %s W H He' % potfile)

xyz_ref = np.array(lmp.gather_atoms('x', 1, 3))

xyz_ref = xyz_ref.reshape(len(xyz_ref)//3, 3)

kd_ref = cKDTree(xyz_ref, boxsize=alattice*size)

lmp.command('timestep 2e-3')

lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

lmp.command('thermo 100')

lmp.command('fix 1 all nvt temp %f %f %f' % (temp, temp, temp/3))


test_cluster = cluster['V0H0He6']['xyz_opt']

for site in test_cluster[-1]:
    # rng_int = np.random.randint(0, len(tet_sites))
    # site = tet_sites[rng_int]
    lmp.command('create_atoms %d single %f %f %f units lattice' % (3, site[0], site[1], site[2]))
    # tet_sites = np.delete(tet_sites, rng_int, axis=0)

rng_seed = np.random.randint(1,10000)

lmp.command('velocity all create %f %d mom yes rot no dist gaussian units box' % (2*temp, rng_seed))

lmp.command('run 0')

lmp.command('dump myDump all custom 500 %s/test.*.atom id type x y z' % dump_folder)

lmp.command('run 15000')

xyz = np.array(lmp.gather_atoms('x', 1, 3))

xyz = xyz.reshape(len(xyz)//3, 3)

ws = WS_analysis(kd_ref, xyz)

print(np.sum(ws))