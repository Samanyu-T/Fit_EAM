import numpy as np
from lammps import lammps
from mpi4py import MPI
import itertools
import copy 
import os 

def lmp_minimize(init_file, read_file, potfile, machine = ''):

    lmp = lammps(name = machine) #,cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data %s' % init_file)

    with open(read_file, 'r') as file:

        file.readline()
        timestep = int(file.readline())

    if mode == 'MPI':
        comm.barrier()


    lmp.command('read_dump %s %d x y z' % (read_file, timestep) )

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    N = lmp.get_natoms()

    id = lmp.numpy.extract_atom('type')
    xyz = lmp.numpy.extract_atom('x')

    he_idx= np.where(id == 3)[0]

    print(xyz[he_idx][0])

if __name__ == '__main__':

    global comm
    global me
    global mode

    try:
        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        nprocs = comm.Get_size()

        mode = 'MPI' 
    except:
        comm = None
        me = 0
        mode = 'serial'

    for orient_folder in os.listdir('../Lammps_Dump/Surface'):
        init_file = os.path.join(orient_folder, 'Depth_0.data')
        
    lmp_minimize('../Lammps_Dump/Surface/100/Depth_0.data', '../Lammps_Dump/Surface/100/0_1/neb.3.dump', 'Potentials/WHHe_test.eam.alloy')

    if mode == 'MPI':
        MPI.Finalize()