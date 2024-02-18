from Lammps_PDefect_Classes import Lammps_Point_Defect
import numpy as np
import os
from mpi4py import MPI
import sys

def main(potfile, machine=''):

    orientx = [1, 0, 0]
    orienty = [0, 1, 0]
    orientz = [0, 0, 1]

    lmp = Lammps_Point_Defect(size = 8, n_vac = 0, potfile=potfile, surface=False, depth=0,
                              orientx=orientx, orienty=orienty, orientz=orientz, conv=10000, machine=machine)

    sites = lmp.get_all_sites()

    if not os.path.exists('../Neb_Dump/Bulk'):
        os.makedirs('../Neb_Dump/Bulk', exist_ok=True)

    tet_0, _ = lmp.Build_Defect([[], [], [sites['tet'][0] + 3]], dump_name='../Neb_Dump/Bulk/tet_0' )

    tet_1, _ = lmp.Build_Defect([[], [], [sites['tet'][0] + 3]], dump_name='../Neb_Dump/Bulk/tet_1' )

    oct_0, _ = lmp.Build_Defect([[], [], [sites['oct'][0] + 3]], dump_name='../Neb_Dump/Bulk/oct_0' )

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

    if me == 0:
        print(potfile)

    comm.Barrier()

    main(potfile)

    MPI.Finalize()