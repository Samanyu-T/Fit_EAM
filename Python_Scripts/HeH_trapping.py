import json
from Lammps_PDefect_Classes import Lammps_Point_Defect
from Handle_Dictionaries import find_binding
import numpy as np
import os 
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD

me = comm.Get_rank()

nprocs = comm.Get_size() 

if me == 0:
    print('Start on %d Procs' % nprocs)
    sys.stdout.flush()  
comm.Barrier()

lmp = Lammps_Point_Defect(size=7, potfile='Potentials/WHHe_test.eam.alloy', n_vac=0, conv = 10000, depth=3)

tet_pos = lmp.alattice*np.array([3.25, 3.5, 3])

data = {}

dump_folder = '../HeH_Trapping_Dump'

for n_vac in range(3):

    lmp.n_vac = n_vac

    pe, rvol, pos = lmp.Build_Defect([[], [], []], dump_name=os.path.join(dump_folder, 'V%dH0He0' % n_vac))

    key = 'V%dH%dHe%d' % (n_vac, 0, 0)

    data[key] = {}

    data[key]['val'] = pe

    data[key]['rvol'] = rvol

    data[key]['xyz_opt'] = pos

    pe, rvol, pos = lmp.Find_Min_Config(init_config=os.path.join(dump_folder,'V%dH0He0' % n_vac), atom_to_add=2)

    key = 'V%dH%dHe%d' % (n_vac, 1, 0)

    data[key] = {}

    data[key]['val'] = pe

    data[key]['rvol'] = rvol

    data[key]['xyz_opt'] = pos

    pe, rvol, pos = lmp.Find_Min_Config(init_config=os.path.join(dump_folder,'V%dH0He0' % n_vac), atom_to_add=3)

    key = 'V%dH%dHe%d' % (n_vac, 0, 1)

    data[key] = {}

    data[key]['val'] = pe

    data[key]['rvol'] = rvol

    data[key]['xyz_opt'] = pos


for n_vac in range(3):
    
    lmp.n_vac = n_vac

    for n_he in range(2, 8):
        
        key = 'V%dH%dHe%d' % (n_vac, 0, n_he-1)

        pe, rvol, pos = lmp.Find_Min_Config(init_config=os.path.join(dump_folder,key), atom_to_add=3)

        key = 'V%dH%dHe%d' % (n_vac, 0, n_he)

        data[key] = {}

        data[key]['val'] = pe

        data[key]['rvol'] = rvol

        data[key]['xyz_opt'] = pos


for n_vac in range(3):
    
    lmp.n_vac = n_vac

    for n_h in range(2, 8):
        
        key = 'V%dH%dHe%d' % (n_vac, n_h - 1, 0)

        pe, rvol, pos = lmp.Find_Min_Config(init_config=os.path.join(dump_folder,key), atom_to_add=2)

        key = 'V%dH%dHe%d' % (n_vac, n_h, 0)

        data[key] = {}

        data[key]['val'] = pe

        data[key]['rvol'] = rvol
        
        data[key]['xyz_opt'] = pos


for n_vac in range(3):
    
    lmp.n_vac = n_vac

    for n_h in range(1, 8):

        for n_he in range(1, 8):

            key = 'V%dH%dHe%d' % (n_vac, n_h - 1, n_he)

            pe, rvol, pos = lmp.Find_Min_Config(init_config=os.path.join(dump_folder,key), atom_to_add=2)

            key = 'V%dH%dHe%d' % (n_vac, n_h, n_he)

            data[key] = {}

            data[key]['val'] = pe

            data[key]['rvol'] = rvol
            
            data[key]['xyz_opt'] = pos


if me == 0:

    he_vac_binding = []

    for i in range(3):
        he_vac_binding.append(find_binding(data, [i, 0, 1], [0, 0, 1], [0,0,1]))

    with open('formations.json' , 'w') as file:
        json.dump(data, file, indent=4)

    for i in range(len(he_vac_binding)):
        plt.plot(he_vac_binding[i], label='n_vac: %d' % i)
    
    plt.show()
    plt.legend()
comm.Barrier()
MPI.Finalize()
