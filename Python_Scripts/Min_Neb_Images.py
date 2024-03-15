import numpy as np
from lammps import lammps
from mpi4py import MPI
import os 
import glob
import sys
import shutil

def min_image(init_file, read_file, potfile, machine = ''):

    lmp = lammps(name = machine,cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data %s' % init_file)

    with open(read_file, 'r') as file:

        file.readline()
        timestep = int(file.readline())

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

    lmp.command('minimize 1e-15 1e-18 10000 10000')

    folder = os.path.join(os.path.dirname(init_file), 'Min_Neb_Images')

    sep = '.'

    filepath = os.path.join(folder, sep.join(os.path.basename(read_file).split('.')[:-1]))

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    lmp.command('write_data %s.data' % filepath)

    lmp.command('write_dump all custom %s.atom id x y z' % filepath)

    lmp.close()

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
         
    potfile = 'Potentials/WHHe_test.eam.alloy'

    if len(sys.argv) > 1:
        potfile = sys.argv[1]

    if me == 0:
        print(potfile)
    comm.Barrier()

    for orient_folder in ['100', '110', '111']:
        init_file = '../Neb_Dump/Surface/%s/init_simple.data' % orient_folder
        for read_file in sorted(glob.glob('../Neb_Dump/Surface/%s/Neb_Images/neb.*.atom' % orient_folder)):
            min_image(init_file, read_file, potfile=potfile, machine = '')
