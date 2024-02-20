from importlib import machinery
from Lammps_PDefect_Classes import Lammps_Point_Defect
import numpy as np
import os
from mpi4py import MPI
import sys

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

def edit_dump(init_path, final_path):

    orient_path = os.path.dirname(init_path)
    orient_str = os.path.basename(orient_path)

    neb_image_folder = os.path.join(orient_path, 'Neb_Images')
    if not os.path.exists(neb_image_folder):
        os.mkdir(neb_image_folder)

    neb_script_folder = '../Neb_Scripts/Surface/%s' % orient_str
    if not os.path.exists(neb_script_folder):
        os.makedirs(neb_script_folder, exist_ok=True)

    with open(final_path, 'r') as file:
        lines = file.readlines()

    with open(final_path, 'w') as file:
        file.write(lines[3])
        file.writelines(lines[9:])
            
    txt = '''
units metal 

atom_style atomic

atom_modify map array sort 0 0.0

read_data %s

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

neb 10e-8 10e-10 5000 5000 100 final %s

write_dump all custom %s/neb.$i.atom id type x y z ''' % (init_path, potfile, final_path, neb_image_folder)


    with open('%s/simple.neb' % neb_script_folder, 'w') as file:
        file.write(txt)


def surface_profile(size, potfile, orientx, orienty, orientz, N = 5, alattice = 3.144221296574379, machine = ''):

    lmp = Lammps_Point_Defect(size = size, n_vac=0, potfile=potfile, surface = True , depth = 0,
                            orientx=orientx, orienty=orienty, orientz=orientz, conv = 100000, machine = machine)
    
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
    
    threshold = 10

    idx = np.where(test_sites[:, -1] > threshold)[0]

    orient_str = '%d%d%d' % (orientx[0], orientx[1], orientx[2])

    if not os.path.exists('../Neb_Dump/Surface/%s' % orient_str):
        os.makedirs('../Neb_Dump/Surface/%s' % orient_str, exist_ok=True)

    pe_init, pos_init = lmp.Build_Defect([[],[],[test_sites[0]]], dump_name='../Neb_Dump/Surface/%s/init_simple' % (orient_str))
    pe_final, pos_final = lmp.Build_Defect([[],[],[test_sites[0] + np.array([0, 0, 3*alattice*np.linalg.norm(orientz)])]],
                                            dump_name='../Neb_Dump/Surface/%s/final_simple' % (orient_str))

    if me == 0:
        edit_dump('../Neb_Dump/Surface/%s/init_simple.data' % orient_str, '../Neb_Dump/Surface/%s/final_simple.atom' % orient_str)
 
    comm.barrier()


def main(potfile, machine=''):

    ''' Use for 100 surface '''
    orientx = [1, 0, 0]
    orienty = [0, 1, 0]
    orientz = [0 ,0, 1]

    bulk = Lammps_Point_Defect(size = 12, n_vac=0, potfile=potfile, surface=False, depth=0,
                              orientx=orientx, orienty=orienty, orientz=orientz, conv=100000, machine='')

    alattice = 3.144221296574379

    ''' Use for 100 surface '''
    orientx = [1, 0, 0]
    orienty = [0, 1, 0]
    orientz = [0 ,0, 1]
    surface_profile(size=12, potfile=potfile, orientx=orientx, orienty=orienty, orientz=orientz,
                    N = 1, alattice = 3.144221296574379, machine = machine)

    ''' Use for 110 surface '''
    orientx = [1, 1, 0]
    orienty = [0, 0,-1]
    orientz = [-1,1, 0]
    surface_profile(size=12, potfile=potfile, orientx=orientx, orienty=orienty, orientz=orientz,
                    N = 1, alattice = 3.144221296574379, machine = machine)

    ''' Use for 111 surface '''
    orientx = [1, 1, 1]
    orienty = [-1,2,-1]
    orientz = [-1,0, 1]
    surface_profile(size=12, potfile=potfile, orientx=orientx, orienty=orienty, orientz=orientz,
                    N = 1, alattice = 3.144221296574379, machine = machine)

    comm.Barrier()
if __name__ == '__main__':

    global comm
    global me
    global nprocs
    global potfile
    
    try:
        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        nprocs = comm.Get_size() 
    except:
        me = 0

        nprocs = 1


    if me == 0:
        print('Start on %d Procs, cwd: %s' % (nprocs, os.getcwd()))
        sys.stdout.flush()  
    comm.Barrier()

    potfile = 'Potentials/WHHe_test.eam.alloy'
    machine = ''

    if len(sys.argv) > 1:
        potfile = sys.argv[1]

    if len(sys.argv) > 2:
        machine = sys.argv[2]

    if me == 0:
        print(potfile)
    comm.Barrier()

    main(potfile, machine)

    MPI.Finalize()