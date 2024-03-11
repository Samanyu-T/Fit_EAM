from lammps import lammps
import numpy as np
import itertools
import statistics
import os
from mpi4py import MPI
import time
def get_tetrahedral_sites(R):

    tet_sites_0 = np.zeros((12,3))
    k = 0

    for [i,j] in itertools.combinations([0, 1, 2],2):
        tet_sites_0[4*k:4*k+4,[i,j]] = np.array( [[0.5 , 0.25],
                                            [0.25, 0.5],
                                            [0.5 , 0.75],
                                            [0.75, 0.5] ])

        k += 1

    tet_sites_1 = np.ones((12,3))
    k = 0

    for [i,j] in itertools.combinations([0, 1, 2],2):
        tet_sites_1[4*k:4*k+4,[i,j]] = np.array( [[0.5 , 0.25],
                                            [0.25, 0.5],
                                            [0.5 , 0.75],
                                            [0.75, 0.5] ])

        k += 1

    tet_sites_unit = np.vstack([tet_sites_0, tet_sites_1])

    tet_sites = tet_sites_unit @ R

    tet_sites = (tet_sites + 2) % 1

    tet_sites = np.unique(tet_sites % 1, axis = 0)
    
    mode = statistics.mode(tet_sites[:,2])

    tet_sites = tet_sites[tet_sites[:,2] == mode]

    return tet_sites

    
def H_surface_energy(size, alattice, orientx, orienty, orientz, h_conc, temp=800, proc=0, machine=''):
    
    surface = 100

    R_inv = np.vstack([orientx, orienty, orientz]).T

    R = np.linalg.inv(R_inv)

    dump_folder = '../H_Surface_Dump'

    unique_tet_sites = get_tetrahedral_sites(R)

    tet_sites = np.array([[0.25, 0.5 ,0]])

    k = -1

    for i in range(size):
        for j in range(size):
                tet_sites = np.vstack([tet_sites, unique_tet_sites + np.array([i, j, k])])


    tet_sites = tet_sites[1:]

    potfile = 'Potentials/WHHe_test.eam.alloy'

    lmp = lammps(name = machine, cmdargs=['-m', str(proc),'-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('boundary p p p')

    lmp.command('lattice bcc %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % 
                (alattice,
                orientx[0], orientx[1], orientx[2],
                orienty[0], orienty[1], orienty[2], 
                orientz[0], orientz[1], orientz[2]
                ) 
                )

    lmp.command('region r_simbox block %f %f %f %f %f %f units lattice' % (

        -1e-9, size + 1e-9, -1e-9, size + 1e-9, -1e-9 - 0.5*surface, size + 1e-9 + 0.5*surface
    ))

    lmp.command('region r_atombox block %f %f %f %f %f %f units lattice' % (

        -1e-4, size + 1e-4, -1e-4, size + 1e-4, -1e-4, size + 1e-4
    ))

    lmp.command('create_box 3 r_simbox')

    lmp.command('create_atoms 1 region r_atombox')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    ref = np.array(lmp.gather_atoms('x', 1, 3))

    ref = ref.reshape(len(ref)//3, 3)

    surface = ref[(-1 < ref[:, 2]) & (ref[:, 2] < 1)]

    N_ref = len(ref)

    n_h = int(h_conc*len(surface)*1e-2)

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('minimize 1e-9 1e-12 10 10')

    lmp.command('minimize 1e-9 1e-12 100 100')

    lmp.command('minimize 1e-9 1e-12 10000 10000')
    
    pe_ref = lmp.get_thermo('pe')

    lmp.command('timestep 1e-3')

    lmp.command('fix 1 all nvt temp %f %f %f' % (temp, temp, temp/3))

    for i in range(n_h):
        rng_int = np.random.randint(0, len(tet_sites))
        site = tet_sites[rng_int]
        lmp.command('create_atoms %d single %f %f %f units lattice' % (2, site[0], site[1], site[2]))
        tet_sites = np.delete(tet_sites, rng_int, axis=0)

    rng_seed = np.random.randint(1,10000)

    lmp.command('velocity all create %f %d mom yes rot no dist gaussian units box' % (2*temp, rng_seed))

    lmp.command('run 0')

    # lmp.command('dump myDump all custom 10 %s/test.*.atom id type x y z' % dump_folder)

    zlat = lmp.get_thermo('zlat')

    for i in range(100):

        lmp.command('run 10')

        lmp.command('variable v_low atom "z < %f"'  % (-3*zlat))
        lmp.command('group g_low variable v_low')
        lmp.command('delete_atoms group g_low')
        
        lmp.command('variable v_high atom "z > %f"'  % ((size+3)*zlat))
        lmp.command('group g_high variable v_high')
        lmp.command('delete_atoms group g_high')

    lmp.command('velocity all zero linear')

    lmp.command('minimize 1e-9 1e-12 100 100')

    lmp.command('minimize 1e-12 1e-15 100 100')

    lmp.command('minimize 1e-13 1e-16 %d %d' % (10000, 10000))

    pe_final = lmp.get_thermo('pe')

    N_final = lmp.get_natoms()

    N_h = (N_final-N_ref)

    c_final = 100*N_h/len(surface)
    
    binding = 0

    if N_final-N_ref>0:

        binding = (pe_ref -2.121*N_h - pe_final)/N_h


    return binding, c_final, N_h


def worker(proc, data_folder):
    machine=''

    size = 10

    # ''' Use for 100 surface '''
    # orientx = [1, 0, 0]
    # orienty = [0, 1, 0]
    # orientz = [0 ,0, 1]

    # ''' Use for 110 surface '''
    # orientx = [1, 1, 0]
    # orienty = [0, 0,-1]
    # orientz = [-1,1, 0]

    ''' Use for 111 surface '''
    orientx = [1, 1, 1]
    orienty = [-1,2,-1]
    orientz = [-1,0, 1]

    alattice = 3.144221

    N = 500

    init_conc = np.linspace(15, 100, N)

    with open(os.path.join(data_folder, 'data_%d.txt' % proc), 'w') as file:
        file.write('')


    for i in range(N):

        binding, conc, n_h = H_surface_energy(size, alattice, orientx, orienty, orientz, init_conc[i], 800, proc, machine)

        with open(os.path.join(data_folder, 'data_%d.txt' % proc), 'a') as file:
            file.write('%d %.4f %.6f \n' % (n_h, conc, binding))


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    me = comm.Get_rank()

    nprocs = comm.Get_size() 

    data_folder = '../H_Surface_Data'

    if me == 0:
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        
        print('Start on %d Procs' % nprocs)

    comm.Barrier()

    worker(me, data_folder)