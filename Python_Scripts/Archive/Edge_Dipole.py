from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt 
from mpi4py import MPI
import sys

def main():

    alattice = 3.144221296574379
    size = 24

    ''' Use for 111 surface '''
    orienty = [-1, -1, -1]
    orientx = [-1,2,-1]
    orientz = [-1,0, 1]

    lmp = lammps()

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

        -1e-9, size + 1e-9, -1e-9, size + 1e-9, -1e-9, size + 1e-9 
    ))

    lmp.command('region r_atombox block %f %f %f %f %f %f units lattice' % (

        -1e-4, size + 1e-4, -1e-4, size + 1e-4, -1e-4, size + 1e-4
    ))

    lmp.command('create_box 3 r_simbox')

    lmp.command('create_atoms 1 region r_atombox')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    potfile = 'Potentials/WHHe_test.eam.alloy'

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('run 0')

    lmp.command('region r_remove block %f %f %f %f %f %f' % (0, size, 0, size//4, size//2 + 0.4, size//2+0.6))
    lmp.command('group g_remove region r_remove')
    lmp.command('delete_atoms group g_remove')

    lmp.command('region r_remove2 block %f %f %f %f %f %f' % (0, size, 3*size//4, size, size//2 + 0.4, size//2+0.6))
    lmp.command('group g_remove2 region r_remove2')
    lmp.command('delete_atoms group g_remove2')

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    # lmp.command('fix 1 all box/relax aniso 0.0')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 10000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Edge.atom id type x y z')

    lmp.command('write_data Lammps_Dump/Dislocations/Edge.data')

    edge = lmp.get_thermo('pe')
    
    lmp.command('create_atoms 3 single %f %f %f units box' % (75.439 ,alattice*size//2 + 0.5, 56.5))

    # lmp.command('minimize 1e-15 1e-18 10 10')

    # lmp.command('minimize 1e-15 1e-18 10 100')

    # lmp.command('minimize 1e-15 1e-18 10000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Edge_He.atom id type x y z')

    he = lmp.get_thermo('pe')


    try:

        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        mode = 'MPI'

    except:

        me = 0
        mode = 'Serial'
        
    binding = 6.16 + edge - he

    if me == 0:
        with open('Test_Data/Screw_Binding.txt', 'w') as file:
            file.write('%f' % binding)

    if mode =='MPI':
        MPI.Finalize()

if __name__ == '__main__':

    main()