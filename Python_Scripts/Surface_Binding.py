from Lammps_PDefect_Classes import Lammps_Point_Defect
import numpy as np
import os
from mpi4py import MPI
import matplotlib.pyplot as plt
import sys

try:

    comm = MPI.COMM_WORLD

    me = comm.Get_rank()

    mode = 'MPI'

except:

    me = 0

    mode = 'Serial'

def climb_sites(alattice, tet_arr, R, R_inv, xy_offset):

    tet = R @ tet_arr.T
    
    tet = (tet + 2) % 1

    idx = np.argsort(tet[-1,:])

    tet = tet[:,idx]

    tet = tet.T

    tet[:, :2] += xy_offset

    tet = alattice * np.linalg.norm(R_inv, axis = 0) * tet

    climb = [tet[0]]

    for _t in tet:

        if _t[2] == climb[-1][2]:

            if len(climb) > 1:

                if np.linalg.norm(climb[-2] - climb[-1]) > np.linalg.norm(climb[-2] - _t):
                    climb[-1] = _t
        
        else:

            climb.append(_t)
        
    return np.array(climb)

def edit_dump(potfile, orient, i):

    save_folder = 'Lammps_Dump/Surface/%s/Depth_%d.atom' % (orient, i)

    with open(save_folder, 'r') as file:
        lines = file.readlines()

    with open(save_folder % (orient, i), 'w') as file:
        file.write(lines[3])
        file.writelines(lines[9:])
                
    txt = '''
units metal 

atom_style atomic

atom_modify map array sort 0 0.0

read_data Lammps_Dump/Surface/%s/Depth_%d.data

mass 1 183.84

mass 2 1.00784

mass 3 4.002602

pair_style eam/alloy

pair_coeff * * %s W H He

thermo 10

run 0

fix 1 all neb 1e-4

timestep 1e-3

min_style quickmin

thermo 100 

variable i equal part

neb 10e-8 10e-10 5000 5000 100 final Lammps_Dump/Surface/%s/Depth_%d.atom

write_dump all custom Lammps_Dump/Neb/%s/%d/neb.$i.dump id type x y z ''' % (orient, i-1, potfile, orient, i, orient, i)

    with open('Lammps_Scripts/surface%d_%s.neb' % (i-1, orient), 'w') as file:
        file.write(txt)


def surface_profile(size, potfile, orientx, orienty, orientz, N = 5, alattice = 3.144221296574379):
    

    lmp = Lammps_Point_Defect(size = size, n_vac=0, potfile=potfile, surface = True , depth = 0,
                            orientx=orientx, orienty=orienty, orientz=orientz, conv = 10000)
    
    init_pos = size//2

    R_inv = np.vstack([orientx, orienty, orientz]).T
    R = np.linalg.inv(R_inv)

    tet_arr = lmp.get_tetrahedral_sites()

    if me == 0:
        climb = climb_sites(alattice, tet_arr, R, R_inv, init_pos)
    else:
        climb = None

    comm.barrier()
    climb = comm.bcast(climb, root = 0)

    depth = np.arange(4)

    test_sites = np.vstack([climb +  np.array([0,0,1])* z * alattice * np.linalg.norm(R_inv, axis = 0)[-1] for z in depth])
    
    pe_arr = np.zeros((len(test_sites, )))
    z_final = np.zeros((len(test_sites, )))

    for i, site in enumerate(test_sites):
        
        orient_str = '%d%d%d' % (orientx[0], orientx[1], orientx[2])

        dump_name = '../Lammps_Dump/Surface/%s/Depth_%d' % (orient_str, i)

        pe, pos = lmp.Build_Defect([[],[],[site]], dump_name=dump_name)

        pe_arr[i] = pe
        z_final[i] = pos[-1][0][-1]

        if i > 0 and me == 0:
            
            edit_dump(potfile, orient_str, i)

    comm.barrier()


def main(potfile, N_images):

    N_images = int(N_images)
    
    alattice = 3.144221296574379
    size = 12

    lmp = Lammps_Point_Defect(size = 12, n_vac=0, potfile=potfile, surface = False , depth = 0,
                            orientx=[1,0,0], orienty=[0,1,0], orientz=[0,0,1], conv = 1000)

    xyz = alattice*np.hstack([6.25, 6.5, 6])
    pe_t, pos = lmp.Build_Defect([[],[],[xyz]], dump_name='Bulk/tet_1')

    with open('Test_Data/Bulk_tet.txt','w') as file:
        file.write('%f' % pe_t)

    xyz = alattice*np.hstack([6.5,6.25, 6])
    pe_t, pos = lmp.Build_Defect([[],[],[xyz]], dump_name='Bulk/tet_2')

    xyz = alattice*np.hstack([6.5,6.5, 6])
    pe_o, pos = lmp.Build_Defect([[],[],[xyz]], dump_name='Bulk/oct')


    ''' Use for 100 surface '''
    orientx = [1, 0, 0]
    orienty = [0, 1, 0]
    orientz = [0 ,0, 1]

    surface_profile(size,potfile, orientx, orienty, orientz, N = N_images)

    ''' Use for 110 surface '''
    orientx = [1, 1, 0]
    orienty = [0, 0,-1]
    orientz = [-1,1, 0]

    surface_profile(size,potfile, orientx, orienty, orientz, N = N_images)

    ''' Use for 111 surface '''
    orientx = [1, 1, 1]
    orienty = [-1,2,-1]
    orientz = [-1,0, 1]

    surface_profile(size,potfile, orientx, orienty, orientz, N = N_images)

    if me == 0:
        with open('Lammps_Dump/Bulk/tet_2.atom', 'r') as file:
            lines = file.readlines()

        with open('Lammps_Dump/Bulk/tet_2.atom', 'w') as file:
            file.write(lines[3])
            file.writelines(lines[9:])

        with open('Lammps_Dump/Bulk/oct.atom', 'r') as file:
            lines = file.readlines()

        with open('Lammps_Dump/Bulk/oct.atom', 'w') as file:
            file.write(lines[3])
            file.writelines(lines[9:])

        
        for pos_type in ['tet_2', 'oct']:
            txt = '''
units metal 

atom_style atomic

atom_modify map array sort 0 0.0

read_data Lammps_Dump/Bulk/tet_1.data

mass 1 183.84

mass 2 1.00784

mass 3 4.002602

pair_style eam/alloy

pair_coeff * * %s W H He

thermo 10

run 0

fix 1 all neb 1e-4

timestep 1e-3

min_style quickmin

thermo 100 

variable i equal part

neb 1e-8 1e-10 5000 5000 100 final Lammps_Dump/Bulk/%s.atom

write_dump all custom Lammps_Dump/Neb/%s/neb.$i.dump id type x y z ''' % (potfile, pos_type, pos_type)

            with open('Lammps_Scripts/tet_%s.neb' % pos_type, 'w') as file:
                file.write(txt)

    comm.barrier()

    if mode == 'MPI':
        MPI.Finalize()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

# else:
    # main('Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy')