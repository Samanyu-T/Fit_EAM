from lammps import lammps
import numpy as np
import os
from mpi4py import MPI
import ctypes
import time
from scipy import stats
import sys

    
def H_surface_energy(size, alattice, orientx, orienty, orientz, N_h, temp=800, machine='', proc = 0):

    N_h = int(N_h)

    max_h = 10

    surface = 100

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

    xyz = np.array(lmp.gather_atoms('x', 1, 3))
    
    xyz = xyz.reshape(len(xyz)//3, 3)

    surface = xyz[np.logical_and(-1 < xyz[:, -1], xyz[:, -1] < 1)]

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    
    xlo = lmp.get_thermo('xlo')
    
    xhi = lmp.get_thermo('xhi')

    ylo = lmp.get_thermo('ylo')
    
    yhi = lmp.get_thermo('yhi')

    # Define the ranges for x and y
    x = np.linspace(xlo + 0.1, xhi - 0.1, 100)  # Start, End, Number of points
    y = np.linspace(ylo + 0.1, yhi - 0.1, 100)   # Start, End, Number of points

    # Create the meshgrid
    X, Y = np.meshgrid(x, y)

    implantation_depth = 0

    # Flatten the meshgrid arrays
    points = np.column_stack(( X.ravel(), Y.ravel(), implantation_depth*np.ones( ( len(X.ravel()) , ) )   ))

    sites = []

    for _p in points:
        add = True
        for _r in surface:
            if np.linalg.norm(_p - _r) < 1:
                add = False
                break
        if add: 
            sites.append(_p)
    
    lmp.command('minimize 1e-9 1e-12 10 10')

    lmp.command('minimize 1e-9 1e-12 100 100')

    lmp.command('minimize 1e-9 1e-12 10000 10000')


    pe_ref = lmp.get_thermo('pe')

    N_atoms = lmp.get_natoms()

    lx = lmp.get_thermo('lx')

    ly = lmp.get_thermo('ly')


    # lmp.command('write_dump all atom ../MCMC_Dump/init.atom')

    # lmp.command('timestep 1e-3')
    
    for i in range(N_h):
        rng_int = np.random.randint(0, len(sites))
        site = sites[rng_int]
        lmp.command('create_atoms %d single %f %f %f units box' % (2, site[0], site[1], site[2]))
        sites = np.delete(sites, rng_int, axis=0)
    
    lmp.command('minimize 1e-9 1e-12 100 100')

    lmp.command('minimize 1e-12 1e-15 100 100')

    lmp.command('minimize 1e-13 1e-16 %d %d' % (10000, 10000))

    # lmp.command('write_dump all atom ../MCMC_Dump/init.atom')

    pe_curr = lmp.get_thermo('pe')

    pe_test = 0

    kb = 8.6173303e-5

    beta = 1/(kb*temp)

    pe_explored = []

    type = np.array( lmp.gather_atoms('type', 0 , 1) )

    all_h_idx = np.where(type != 1)[0]
        
    n_ensemple = int(50)

    n_samples = int(50)

    converged = False
    
    converge_thresh = 0.95

    canonical = np.zeros((n_ensemple, ))

    samples = np.zeros((n_samples,))

    surface_retention = np.zeros((n_ensemple,))

    counter = 0
    
    acceptance_ratio = 0

    dump_image = 0
    
    pbc = np.array([lx, ly, 1000])

    while True: 
        
        n_accept = 0

        while n_accept < n_ensemple:

            xyz = np.array(lmp.gather_atoms('x', 1, 3))

            xyz = xyz.reshape(len(xyz)//3 , 3)

            prev_hxyz = xyz[all_h_idx]

            np.random.shuffle(all_h_idx)

            slct_h = np.clip(len(all_h_idx), a_min=0, a_max=max_h)

            h_idx = np.copy(all_h_idx[:slct_h])
            
            displace = alattice*np.random.normal(loc=0, scale=0.5, size=(len(h_idx),3))

            xyz[h_idx] += displace

            # xyz = xyz - np.floor(xyz/pbc)*pbc

            xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            lmp.scatter_atoms('x', 1, 3, xyz_c)

            lmp.command('minimize 1e-9 1e-12 10 10')

            lmp.command('minimize 1e-9 1e-12 100 100')

            lmp.command('minimize 1e-9 1e-12 10000 10000')

            xyz = np.array(lmp.gather_atoms('x', 1, 3))

            xyz = xyz.reshape(len(xyz)//3 , 3)

            hxyz = xyz[all_h_idx]

            pe_test = lmp.get_thermo('pe')

            acceptance = np.min([1, np.exp(-beta*(pe_test - pe_curr))])

            rng = np.random.rand()

            # print(pe_test - pe_curr)

            n_h_surface = sum( (-3*alattice < xyz[all_h_idx, 2]) & (xyz[all_h_idx, 2] < 3*alattice) )

            pe_explored.append((pe_ref -2.121*N_h - pe_test)/N_h)

            delta = prev_hxyz - hxyz

            delta = delta - np.round(delta/pbc)*pbc
                    
            distance = np.linalg.norm(delta, axis = 1)

            if rng <= acceptance and distance.max() > alattice*np.sqrt(2)/4:
                

                pe_curr = pe_test

                # lmp.command('write_dump all atom ../MCMC_Dump/data_%d.atom' % counter)

                canonical[n_accept] = (pe_curr - (-8.949*N_atoms) - (-2.121*N_h))/(2*lx*ly)

                # print(canonical[n_accept])

                surface_retention[n_accept] = n_h_surface

                n_accept += 1

                counter += 1
                
                acceptance_ratio += 1


                if n_accept % 15 == 10:
                    lmp.command('write_dump all atom ../MCMC_Dump/Nh_%d_image_%d.atom' % (N_h, dump_image))
                    dump_image += 1
                
            else:
                
                xyz[h_idx] -= displace

                xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

                lmp.scatter_atoms('x', 1, 3, xyz_c)

        n_accept = 0

        if converged:
            break
            
        while n_accept < n_samples:

            xyz = np.array(lmp.gather_atoms('x', 1, 3))

            xyz = xyz.reshape(len(xyz)//3 , 3)

            prev_hxyz = xyz[all_h_idx]

            np.random.shuffle(all_h_idx)

            slct_h = np.clip(len(all_h_idx), a_min=0, a_max=max_h)

            h_idx = np.copy(all_h_idx[:slct_h])
            
            displace = np.random.normal(loc=0, scale=1, size=(len(h_idx),3))

            xyz[h_idx] += displace

            # xyz = xyz - np.floor(xyz/pbc)*pbc

            xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            lmp.scatter_atoms('x', 1, 3, xyz_c)

            lmp.command('minimize 1e-9 1e-12 10 10')

            lmp.command('minimize 1e-9 1e-12 100 100')

            lmp.command('minimize 1e-9 1e-12 10000 10000')

            xyz = np.array(lmp.gather_atoms('x', 1, 3))

            xyz = xyz.reshape(len(xyz)//3 , 3)

            hxyz = xyz[all_h_idx]

            pe_test = lmp.get_thermo('pe')

            acceptance = np.min([1, np.exp(-beta*(pe_test - pe_curr))])

            rng = np.random.rand()

            # print(pe_test - pe_curr)

            n_h_surface = sum( (alattice < xyz[all_h_idx, 2]) & (xyz[all_h_idx, 2] < alattice) )

            pe_explored.append((pe_ref -2.121*N_h - pe_test)/N_h)


            delta = prev_hxyz - hxyz

            delta = delta - np.round(delta/pbc)*pbc
                    
            distance = np.linalg.norm(delta, axis = 1)

            if rng <= acceptance and distance.max() > alattice*np.sqrt(2)/4:

                pe_curr = pe_test
                
                # lmp.command('write_dump all atom ../MCMC_Dump/data_%d.atom' % counter)

                samples[n_accept] = (pe_curr - (-8.949*N_atoms) - (-2.121*N_h))/(2*lx*ly)
                
                surface_retention[n_accept] = n_h_surface

                counter += 1
                
                n_accept += 1

                acceptance_ratio += 1

            else:
                
                xyz[h_idx] -= displace

                xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

                lmp.scatter_atoms('x', 1, 3, xyz_c)

        res = stats.ttest_ind(canonical, samples, equal_var=False)

        print(res)
        sys.stdout.flush()

        if res.pvalue > (1-converge_thresh):
            converged = True
        
    if not os.path.exists('../MCMC_Data'):
        os.mkdir('../MCMC_Data')

    acceptance_ratio /= len(pe_explored)
    
    print('Converged %d, Acceptance Ratio %f, Max Pe %f, Min Pe %f ,N_retain %d' % (N_h, acceptance_ratio, canonical.max(), canonical.min(), surface_retention[-1])) 

    np.savetxt('../MCMC_Data/mcmc_explore_%d.txt' % proc, pe_explored)

    np.savetxt('../MCMC_Data/mcmc_unique_%d.txt' % proc, canonical)
    
    np.savetxt('../MCMC_Data/mcmc_nh_%d.txt' % proc, surface_retention)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    nprocs = comm.Get_size() 

    data_folder = '../H_Surface_Data'

    if rank == 0:
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        
        print('Start on %d Procs' % nprocs)

    comm.Barrier()
    
    ''' Use for 100 surface '''
    # orientx = [1, 0, 0]
    # orienty = [0, 1, 0]
    # orientz = [0 ,0, 1]

    ''' Use for 110 surface '''
    # orientx = [1, 1, 0]
    # orienty = [0, 0,-1]
    # orientz = [-1,1, 0] 

    ''' Use for 111 surface '''
    orientx = [1, 1, 1]
    orienty = [-1,2,-1]
    orientz = [-1,0, 1]

    alattice = 3.144221

    boxsize = 10
    ''' For 100 use 1, 110 use 2, 111 use 4'''

    n_surface = 4*boxsize**2

    init_conc = np.ceil(n_surface*1e-2*np.hstack([np.linspace(0.25, 10, nprocs//3), np.logspace(1, 3, nprocs - nprocs//3)]))

    H_surface_energy(boxsize, alattice, orientx, orienty, orientz, init_conc[rank], 800, '', rank)

