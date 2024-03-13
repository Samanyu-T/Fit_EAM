from re import T
from lammps import lammps
import numpy as np
import itertools
import statistics
import os
from mpi4py import MPI
import ctypes
import time
import matplotlib.pyplot as plt

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

    
def H_surface_energy(comm, rank, size, alattice, orientx, orienty, orientz, h_conc, temp=800, machine=''):
    
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

    lmp = lammps(name = machine, comm=comm)

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

    lmp.command('minimize 1e-9 1e-12 100 100')

    lmp.command('minimize 1e-12 1e-15 100 100')

    lmp.command('minimize 1e-13 1e-16 %d %d' % (10000, 10000))


    pe_curr = lmp.get_thermo('pe')

    pe_test = 0

    kb = 8.6173303e-5

    beta = 1/(kb*temp)

    pe_unique = []

    pe_explored = []

    for mcmc_iter in range(2):

        type = np.array(lmp.gather_atoms("type", 0, 1))
        
        all_h_idx = np.where(type==2)[0]
        
        n_displace = np.clip(len(all_h_idx), a_min=0, a_max=5)

        rng_int = np.random.randint(0, len(all_h_idx), size=(n_displace,))

        h_idx = all_h_idx[rng_int]
        
        xyz_curr = np.array(lmp.gather_atoms("x", 1, 3))

        xyz_curr = xyz_curr.reshape(len(xyz_curr)//3, 3)

        xyz_test = np.copy(xyz_curr)

        move = np.random.normal(loc=0, scale=1, size=(len(h_idx), 3))

        xyz_test[h_idx] = xyz_test[h_idx] + move

        xyz_testc = xyz_test.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        lmp.scatter_atoms('x', 1, 3, xyz_testc)
            
        lmp.command('minimize 1e-9 1e-12 100 100')

        lmp.command('minimize 1e-12 1e-15 100 100')

        lmp.command('minimize 1e-13 1e-16 %d %d' % (10000, 10000))

        pe_test = lmp.get_thermo('pe')

        acceptance = min(1, np.exp(-beta*(pe_test - pe_curr)))

        rand = np.random.rand()

        pe_explored.append((pe_ref -2.121*n_h - pe_test)/n_h)
        pe_unique.append((pe_ref -2.121*n_h - pe_curr)/n_h)

        if rand <= acceptance:
            print('True')

            pe_curr = pe_test

        else:

            xyz_currc = xyz_curr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            lmp.scatter_atoms('x', 1, 3, xyz_currc)


    pe_final = lmp.get_thermo('pe')

    N_final = lmp.get_natoms()

    N_h = (N_final-N_ref)

    c_final = 100*N_h/len(surface)
    
    binding = 0

    if N_final-N_ref>0:

        binding = (pe_ref -2.121*N_h - pe_final)/N_h

    # if rank == 0:

    #     plt.plot(pe_unique)

    #     plt.plot(pe_explored)
    #     plt.show()

    return binding, c_final, N_h


def worker(proc, data_folder):
    machine=''

    size = 10

    # ''' Use for 100 surface '''
    # orientx = [1, 0, 0]
    # orienty = [0, 1, 0]
    # orientz = [0 ,0, 1]

    ''' Use for 110 surface '''
    orientx = [1, 1, 0]
    orienty = [0, 0,-1]
    orientz = [-1,1, 0]

    # ''' Use for 111 surface '''
    # orientx = [1, 1, 1]
    # orienty = [-1,2,-1]
    # orientz = [-1,0, 1]

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

    rank = comm.Get_rank()

    size = comm.Get_size() 

    # Number of desired processes per group
    procs_per_group = 2

    # Calculate the number of groups
    num_groups = size // procs_per_group

    # Split the processes into groups
    color = rank // procs_per_group
    group_comm = comm.Split(color, rank)

    # Now, each group should contain the desired number of processes
    group_rank = group_comm.Get_rank()
    group_size = group_comm.Get_size()

    print("Group:", color, "Group Rank:", group_rank, "out of", group_size)
    
    data_folder = '../H_Surface_Data'

    if rank == 0:
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        
        print('Start on %d Procs' % size)

    comm.Barrier()
    size = 10

    orientx = [1, 1, 1]
    orienty = [-1,2,-1]
    orientz = [-1,0, 1]

    alattice = 3.144221

    H_surface_energy(group_comm, group_rank, size, alattice, orientx, orienty, orientz, 10, 800)
    # worker(me, data_folder)
    
    MPI.Finalize()
