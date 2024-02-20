from Lammps_PDefect_Classes import Lammps_Point_Defect
import numpy as np
import os
from mpi4py import MPI
import sys

def edit_dump(init_path, final_path):

    orient_path = os.path.dirname(init_path)
    orient_str = os.path.basename(orient_path)

    neb_image_folder = os.path.join(orient_path, 'Neb_Images')
    if not os.path.exists(neb_image_folder):
        os.mkdir(neb_image_folder)

    neb_script_folder = '../Neb_Scripts/Bulk/%s' % orient_str
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
        
def main(potfile, machine=''):

    orientx = [1, 0, 0]
    orienty = [0, 1, 0]
    orientz = [0, 0, 1]

    lmp = Lammps_Point_Defect(size = 8, n_vac = 0, potfile=potfile, surface=False, depth=0,
                              orientx=orientx, orienty=orienty, orientz=orientz, conv=10000, machine=machine)

    if not os.path.exists('../Neb_Dump/Bulk/Tet_Tet'):
        os.makedirs('../Neb_Dump/Bulk/Tet_Tet', exist_ok=True)

    if not os.path.exists('../Neb_Dump/Bulk/Tet_Oct'):
        os.makedirs('../Neb_Dump/Bulk/Tet_Oct', exist_ok=True)

    tet_0 = lmp.alattice*np.array([3,3.5,3.25])
    tet_1 = lmp.alattice*np.array([3.25,3.5,3])
    oct_0 = lmp.alattice*np.array([3,3.5,3.5])

    _, _ = lmp.Build_Defect([[], [], [tet_0]], dump_name='../Neb_Dump/Bulk/Tet_Tet/tet_0' )

    _, _ = lmp.Build_Defect([[], [], [tet_1]], dump_name='../Neb_Dump/Bulk/Tet_Tet/tet_1' )

    _, _ = lmp.Build_Defect([[], [], [tet_0]], dump_name='../Neb_Dump/Bulk/Tet_Oct/tet' )

    _, _ = lmp.Build_Defect([[], [], [oct_0]], dump_name='../Neb_Dump/Bulk/Tet_Oct/oct' )

    if me == 0:
        edit_dump('../Neb_Dump/Bulk/Tet_Tet/tet_0.data', '../Neb_Dump/Bulk/Tet_Tet/tet_1.atom')
        edit_dump('../Neb_Dump/Bulk/Tet_Oct/tet.data', '../Neb_Dump/Bulk/Tet_Oct/oct.atom' )
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

    if len(sys.argv) > 1:
        potfile = sys.argv[1]

    machine = ''

    if len(sys.argv) > 2:
        machine = sys.argv[2]

    if me == 0:
        print(potfile, machine)

    comm.Barrier()

    main(potfile, machine)

    MPI.Finalize()