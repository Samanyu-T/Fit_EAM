from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt 
from mpi4py import MPI
import sys
import ctypes

def main():

    alattice = 3.144221296574379
    size = 12

    try:

        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        mode = 'MPI'

    except:

        me = 0
        mode = 'Serial'

    lmp = lammps()

    # lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('boundary p p p')

    ''' Use for 100 surface '''
    orientx = [1, 0, 0]
    orienty = [0, 1, 0]
    orientz = [0 ,0, 1]

    ''' Use for 110 surface '''
    # orientx = [1, 1, 0]
    # orienty = [0, 0,-1]
    # orientz = [-1,1, 0]

    ''' Use for 111 surface '''
    # orientx = [1, 1, 1]
    # orienty = [-1,2,-1]
    # orientz = [-1,0, 1]

    lmp.command('lattice bcc %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % 
                (alattice,
                orientx[0], orientx[1], orientx[2],
                orienty[0], orienty[1], orienty[2], 
                orientz[0], orientz[1], orientz[2]
                ) 
                )
    
    lmp.command('region r_simbox block %f %f %f %f %f %f units lattice' % (

        -1e-9, size + 1e-9, -1e-9, size + 1e-9, -1e-9 , size + 1e-9 
    ))

    lmp.command('region r_atombox1 block %f %f %f %f %f %f units lattice' % (

        -1e-4, size + 1e-4, -1e-4, size + 1e-4, -1e-4, size + 1e-4
    ))

    lmp.command('create_box 3 r_simbox')

    lmp.command('create_atoms 1 region r_atombox1')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    potfile = 'Potentials/WHHe_test.eam.alloy'

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('compute patom all pe/atom')

    lmp.command('run 0') 

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
        
    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 10000 10000')

    N = lmp.get_natoms()


    # lmp.command('group remove id %d' % (N//2))

    # lmp.command('delete_atoms group remove')

    for i in range(5):

        id = lmp.extract_atom('id')

        pe = lmp.extract_compute('patom', 1, 1)

        pe_atom = np.ctypeslib.as_array(pe, shape=(N,))

        id_atom = np.ctypeslib.as_array(id, shape=(N,))
        
        idx = pe_atom.argmax()

        print(pe_atom[idx], id_atom[idx])

        lmp.command('group remove id %d' % (id_atom[idx]))

        lmp.command('delete_atoms group remove')

        lmp.command('minimize 1e-15 1e-18 10 10')

        lmp.command('minimize 1e-15 1e-18 10 100')

        lmp.command('minimize 1e-15 1e-18 10000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Loop.atom id type x y z c_patom')


main()