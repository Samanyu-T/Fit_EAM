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

    
def H_surface_energy(size, alattice, orientx, orienty, orientz, h_conc, temp=800, machine='', proc = 0):

    max_h = 10

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

    for i in range(n_h):
        rng_int = np.random.randint(0, len(tet_sites))
        site = tet_sites[rng_int]
        lmp.command('create_atoms %d single %f %f %f units lattice' % (2, site[0], site[1], site[2]))
        tet_sites = np.delete(tet_sites, rng_int, axis=0)

    lmp.command('minimize 1e-9 1e-12 100 100')

    lmp.command('minimize 1e-12 1e-15 100 100')

    lmp.command('minimize 1e-13 1e-16 %d %d' % (10000, 10000))

    pbc = lmp.get_thermo('lx')

    pe_curr = lmp.get_thermo('pe')

    pe_test = 0

    kb = 8.6173303e-5

    beta = 1/(kb*temp)

    pe_unique = []

    pe_explored = []

    type = np.array( lmp.gather_atoms('type', 0 , 1) )

    all_h_idx = np.where(type == 2)[0]

    N_h = len(all_h_idx)
    
    n_accept = 0
    
    N_iterations = int(1e5)

    for mcmc_iter in range(N_iterations):
        
        if mcmc_iter % 1000 == 0:
            print(proc, mcmc_iter)

        xyz = np.array(lmp.gather_atoms('x', 1, 3))

        xyz = xyz.reshape(len(xyz)//3 , 3)

        np.random.shuffle(all_h_idx)

        slct_h = np.clip(len(all_h_idx), a_min=0, a_max=max_h)

        h_idx = np.copy(all_h_idx[:slct_h])
        
        displace = np.random.normal(loc=0, scale=0.5, size=(len(h_idx),3))

        xyz[h_idx] += displace

        xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        lmp.scatter_atoms('x', 1, 3, xyz_c)

        lmp.command('minimize 1e-9 1e-12 10 10')

        lmp.command('minimize 1e-9 1e-12 100 100')

        lmp.command('minimize 1e-9 1e-12 10000 10000')

        pe_test = lmp.get_thermo('pe')

        acceptance = np.min([1, np.exp(-beta*(pe_test - pe_curr))])

        rng = np.random.rand()

        pe_explored.append((pe_ref -2.121*N_h - pe_test)/N_h)

        if rng <= acceptance:
            pe_unique.append((pe_ref -2.121*N_h - pe_curr)/N_h)

            n_accept += 1

            pe_curr = pe_test

        else:
            
            xyz[h_idx] -= displace

            xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            lmp.scatter_atoms('x', 1, 3, xyz_c)

    N_final = lmp.get_natoms()

    pe_final = lmp.get_thermo('pe')

    lmp.close()

    print(n_accept/N_iterations)

    N_h = (N_final-N_ref)

    c_final = 100*N_h/len(surface)
    
    binding = 0

    if N_final-N_ref>0:

        binding = (pe_ref -2.121*N_h - pe_final)/N_h
    
    pe_explored = np.array(pe_explored)
    pe_unique = np.array(pe_unique)

        
    plt.plot(pe_explored)
    plt.plot(pe_unique)
    plt.show()

    if not os.path.exists('../MCMC_Data'):
        os.mkdir('../MCMC_Data')

    np.savetxt('../MCMC_Data/mcmc_explore_%d.txt' % proc, pe_explored)
    np.savetxt('../MCMC_Data/mcmc_unique_%d.txt' % proc, pe_unique)
    

    return binding, c_final, N_h


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    size = comm.Get_size() 

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

    init_conc = np.linspace(0.25, 100, size)

    H_surface_energy(size, alattice, orientx, orienty, orientz, init_conc[rank], 800, '',rank)

