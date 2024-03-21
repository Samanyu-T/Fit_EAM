from lammps import lammps
import numpy as np
import itertools
import statistics
import os
from mpi4py import MPI
import ctypes
import time
from scipy import stats
import glob

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

    
def mcmc_optim_cluster(filepath, temp=800, machine=''):

    max_h = 2

    potfile = 'Potentials/test.0.eam.alloy'

    lmp = lammps(name = machine, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data %s' % filepath)

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    
    lmp.command('run 0')

    pe_ref = lmp.get_thermo('pe')

    pe_curr = pe_ref

    pe_test = 0

    kb = 8.6173303e-5

    pe_explored = []

    type = np.array( lmp.gather_atoms('type', 0 , 1) )

    all_h_idx = np.where(type != 1)[0]
    
    N_h = len(all_h_idx)
    
    n_ensemple = int(50)

    canonical = np.zeros((n_ensemple, ))

    pos_h = np.zeros((n_ensemple,N_h,3))

    n_accept = 0

    counter = 0

    xyz = np.array(lmp.gather_atoms('x', 1, 3))

    xyz = xyz.reshape(len(xyz)//3 , 3)
    
    beta = 1/(kb*temp)

    while n_accept < n_ensemple and counter < 5*n_ensemple:
        
        counter += 1

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

        pe_explored.append((pe_test - pe_ref))

        if rng <= acceptance:
                
            temp = temp*np.exp(-n_accept/n_ensemple)

            beta = 1/(kb*temp)

            pe_curr = pe_test

            # lmp.command('write_dump all atom ../MCMC_Dump/data_%d.atom' % n_accept)

            print(pe_curr - pe_ref)

            canonical[n_accept] = (pe_curr - pe_ref)

            pos_h[n_accept] = xyz[all_h_idx]

            n_accept += 1

            
        else:
            
            xyz[h_idx] -= displace

            xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            lmp.scatter_atoms('x', 1, 3, xyz_c)


    min_energy = canonical.argmin()

    xyz[all_h_idx] = pos_h[min_energy]

    xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    lmp.scatter_atoms('x', 1, 3, xyz_c)

    lmp.command('minimize 1e-9 1e-12 10 10')

    lmp.command('minimize 1e-9 1e-12 100 100')

    lmp.command('minimize 1e-9 1e-12 10000 10000')

    pe_final = lmp.get_thermo('pe')

    print('final energy: %f' % pe_final)

    filename = os.path.basename(filepath).split('.')[0]

    lmp.command('write_data ../HeH_Clusters_New/%s_new.data' % filename)

    lmp.close()

    # plt.plot(canonical)
    # plt.show()

    
    # lmp.command('timestep 1e-3')
    

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

    alattice = 3.144221

    for filename in glob.glob('../HeH_Clusters/V0H0He*.data'):
        print(filename)
        mcmc_optim_cluster(filename, 1000, '')

