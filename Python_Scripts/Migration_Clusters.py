from lammps import lammps
import numpy as np
import itertools
import statistics
import os
from mpi4py import MPI
import ctypes
import time
import matplotlib.pyplot as plt
from scipy import stats
from sympy import N

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

    max_h = 2

    surface = 100

    R_inv = np.vstack([orientx, orienty, orientz]).T

    R = np.linalg.inv(R_inv)

    dump_folder = '../H_Surface_Dump'

    unique_tet_sites = get_tetrahedral_sites(R)

    tet_sites = np.array([[0.25, 0.5 ,0]])

    k = -0.5

    for i in range(size):
        for j in range(size):
                tet_sites = np.vstack([tet_sites, unique_tet_sites + np.array([i, j, k])])


    tet_sites = tet_sites[1:]

    potfile = 'Potentials/test.0.eam.alloy'

    lmp = lammps(name = machine, cmdargs=['-m', str(proc),'-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data ../HeH_Clusters/V0H4He4.data')

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    
    lmp.command('run 0')

    pe_ref = lmp.get_thermo('pe')

    pe_curr = pe_ref

    pe_test = 0

    kb = 8.6173303e-5

    beta = 1/(kb*temp)

    pe_explored = []

    type = np.array( lmp.gather_atoms('type', 0 , 1) )

    all_h_idx = np.where(type != 1)[0]
    
    N_h = len(all_h_idx)
    
    n_ensemple = int(1000)

    n_samples = int(50)

    converged = False
    
    converge_thresh = 0.95

    canonical = np.zeros((n_ensemple, ))

    pos_h = np.zeros((n_ensemple,N_h,3))


    n_accept = 0

    counter = 0

    while n_accept < n_ensemple:

        xyz = np.array(lmp.gather_atoms('x', 1, 3))

        xyz = xyz.reshape(len(xyz)//3 , 3)

        np.random.shuffle(all_h_idx)

        slct_h = np.clip(len(all_h_idx), a_min=0, a_max=max_h)

        h_idx = np.copy(all_h_idx[:slct_h])
        
        displace = np.random.normal(loc=0, scale=0.5, size=(len(h_idx),3))

        xyz[h_idx] += displace

        # xyz = xyz - np.floor(xyz/pbc)*pbc

        xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        lmp.scatter_atoms('x', 1, 3, xyz_c)

        lmp.command('minimize 1e-9 1e-12 10 10')

        lmp.command('minimize 1e-9 1e-12 100 100')

        lmp.command('minimize 1e-9 1e-12 10000 10000')

        pe_test = lmp.get_thermo('pe')

        acceptance = np.min([1, np.exp(-beta*(pe_test - pe_curr))])

        rng = np.random.rand()

        # print(pe_test - pe_curr)

        n_h_surface = sum( (-3*alattice < xyz[all_h_idx, 2]) & (xyz[all_h_idx, 2] < 3*alattice) )

        pe_explored.append((pe_test - pe_ref))

        if rng <= acceptance:
            
            pe_curr = pe_test

            lmp.command('write_dump all atom ../MCMC_Dump/data_%d.atom' % counter)

            print(pe_curr - pe_ref)

            canonical[n_accept] = (pe_curr - pe_ref)

            pos_h[n_accept] = xyz[all_h_idx]

            n_accept += 1

            counter += 1
            
        else:
            
            xyz[h_idx] -= displace

            xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            lmp.scatter_atoms('x', 1, 3, xyz_c)


    plt.plot(canonical)
    plt.show()

    if not os.path.exists('../MCMC_Data'):
        os.mkdir('../MCMC_Data')

    np.savetxt('../MCMC_Data/mcmc_explore_%d.txt' % proc, pe_explored)

    np.savetxt('../MCMC_Data/mcmc_unique_%d.txt' % proc, canonical)
    


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

    H_surface_energy(size, alattice, orientx, orienty, orientz, init_conc[rank], 1000, '',rank)

