from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt 
from mpi4py import MPI
import sys

def create_dislocation():

    try:

        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        mode = 'MPI'

    except:

        me = 0
        mode = 'Serial'

    alattice = 3.144221296574379
    size = 21

    ''' Use for 111 surface '''
    orientx = [1, 1, 1]
    orienty = [-1,2,-1]
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

        -1e-9, size + 1e-9, -1e-9, size + 1e-9, -1e-9 , size + 1e-9 +10
    ))

    lmp.command('region r_atombox1 block %f %f %f %f %f %f units lattice' % (

        -1e-4, size + 1e-4, -1e-4, size + 1e-4, -1e-4, size//2 
    ))


    lmp.command('region r_atombox2 block %f %f %f %f %f %f units lattice' % (

        -1e-4, size - 1 + 1e-4 , -1e-4, size + 1e-4, size//2 + 1e-4, size + 1e-4
    ))

    lmp.command('create_box 3 r_simbox')

    lmp.command('create_atoms 1 region r_atombox1')

    lmp.command('create_atoms 1 region r_atombox2')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    potfile = 'Potentials/WHHe_test.eam.alloy'

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('run 0')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Edge/Edge_Single.atom id type x y z')

    lmp.command('write_data Lammps_Dump/Dislocations/Edge/Edge_Single.data')

    data = np.loadtxt('Lammps_Dump/Dislocations/Edge/Edge_Single.atom', skiprows=9)


    atom_pos = data[:,-3:]

    z_lat = alattice*np.linalg.norm( np.array(orientz))
    
    idx = np.where( atom_pos[:, -1] > z_lat*(size//2 + 0.01))[0]

    ratio = (3*sum(np.abs(orientx))*size)/(3*sum(np.abs(orientx))*size - 10.5)

    atom_pos[idx, 0] *= ratio

    starting_lines = ''

    with open('Lammps_Dump/Dislocations/Edge/Edge_Single.atom', 'r') as file:
        for i in range(9):

            if i == 1:
                timestep = file.readline()
                starting_lines += timestep
                timestep = int(timestep)
            else:
                starting_lines += file.readline()

    if me == 0:
        with open('Lammps_Dump/Dislocations/Edge/Edge_Single.atom', 'w') as file:
            file.write(starting_lines)
            for i, pos in enumerate(atom_pos):
                file.write('%d %d ' % (data[i,0], data[i,1]))
                np.savetxt(file, pos, fmt = '%.5f', newline=' ')
                file.write('\n')

    lmp.command('read_dump Lammps_Dump/Dislocations/Edge/Edge_Single.atom %d x y z' % timestep)

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
        
    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 10000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Edge/Edge_Single.atom id type x y z')

    lmp.command('write_data Lammps_Dump/Dislocations/Edge/Edge_Single.data')

    if mode =='MPI':
        MPI.Finalize()

def binding(potfile):

    try:

        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        mode = 'MPI'

    except:

        me = 0
        mode = 'Serial'

    N = 8

    z = np.linspace(0, 16, N)
    
    d_pos = np.array([64.0439, 77.8412/2, 45.7768])

    pe_arr = np.zeros((N,))
    z_arr = np.zeros((N,))

    for i, _z in enumerate(z):
        lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data Lammps_Dump/Dislocations/Edge/Edge_Single.data')

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % potfile)

        lmp.command('thermo 50')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        lmp.command('run 0')

        pe0 = lmp.get_thermo('pe')
            
        lmp.command('create_atoms 3 single %f %f %f units box' % (d_pos[0], d_pos[1], d_pos[2] + _z)) 

        lmp.command('minimize 1e-15 1e-18 10 10')

        lmp.command('minimize 1e-15 1e-18 10 100')

        lmp.command('minimize 1e-15 1e-18 1000 1000')

        lmp.command('write_dump all custom Lammps_Dump/Dislocations/Edge/Edge_He%d.atom id type x y z' % i)

        pe1 = lmp.get_thermo('pe')

        xyz_system = np.array(lmp.gather_atoms('x',1,3))

        xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

        pos = xyz_system[-1]

        z_arr[i] = np.linalg.norm(pos - d_pos)

        pe_arr[i] = pe0 + - pe1

    if me == 0:
        save = np.hstack([z_arr.reshape(N, 1), pe_arr.reshape(N,1)])
        np.savetxt('Test_Data/Edge_Binding.txt', save)

    if mode =='MPI':
        MPI.Finalize()

if __name__ == '__main__':

    # create_dislocation()
        
    binding(sys.argv[1])