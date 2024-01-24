from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt 
from mpi4py import MPI
import sys

def init_screw(alattice, size):

    try:

        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        mode = 'MPI'

    except:

        me = 0
        mode = 'Serial'

    lmp = lammps()

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('boundary p p p')


    ''' Use for 100 surface '''
    # orientx = [1, 0, 0]
    # orienty = [0, 1, 0]
    # orientz = [0 ,0, 1]

    ''' Use for 111 surface '''
    orientz = [1, 1, 1]
    orienty = [-1,-1,2]
    orientx = [1,-1, 0]

    lmp.command('lattice bcc %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % 
                (alattice,
                orientx[0], orientx[1], orientx[2],
                orienty[0], orienty[1], orienty[2], 
                orientz[0], orientz[1], orientz[2]
                ) 
                )
    
    lmp.command('region r_simbox block %d %d %d %d %d %f units lattice' % (-4*size, 4*size, -4*size, 4*size,  0, size))

    lmp.command('region r_atombox cylinder z 0 0 %d 0 %d units lattice' % (size, size))

    lmp.command('create_box 3 r_simbox')

    lmp.command('create_atoms 1 region r_atombox')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    potfile = 'Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy'

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('run 0')

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('fix 1 all box/relax aniso 0.0')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Screw/Cylinder.atom id type x y z')

    data = np.loadtxt('Lammps_Dump/Dislocations/Screw/Cylinder.atom', skiprows=9)

    atom_pos = data[:,-3:]

    screw_centres = alattice*np.array([[0, 0, size//2]])

    screw_idx = []
    screw_pos = []

    for i, pos in enumerate(atom_pos):

        for centre in screw_centres:
            if np.linalg.norm(pos - centre) < 0.1:
                screw_idx.append(i)
                screw_pos.append(pos)

    b = alattice

    for i, pos in enumerate(atom_pos):

        theta = np.arctan2(pos[1] - screw_centres[0][1], pos[0] - screw_centres[0][0])
        # Ensure the angle is between 0 and 2*pi
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        data[i,-1] += (b/(2*np.pi))*(theta)

    starting_lines = ''

    with open('Lammps_Dump/Dislocations/Screw/Cylinder.atom', 'r') as file:
            for i in range(9):

                if i == 1:
                    timestep = file.readline()
                    starting_lines += timestep
                    timestep = int(timestep)
                else:
                    starting_lines += file.readline()

    if me == 0:
        with open('Lammps_Dump/Dislocations/Screw/Screw_Cylinder_Init.atom', 'w') as file:
            file.write(starting_lines)
            for i, pos in enumerate(atom_pos):
                file.write('%d %d ' % (data[i,0], data[i,1]))
                np.savetxt(file, pos, fmt = '%.5f', newline=' ')
                file.write('\n')

    lmp.command('read_dump Lammps_Dump/Dislocations/Screw/Screw_Cylinder_Init.atom %d x y z' % timestep)

    lmp.command('fix 1 all box/relax aniso 0.0')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Screw/Screw_Cylinder_Relaxed.atom id type x y z')

    lmp.command('write_data Lammps_Dump/Dislocations/Screw/Screw_Cylinder_Relaxed.data')

    pe_screw = lmp.get_thermo('pe')

    lmp.close()

    if mode =='MPI':
        MPI.Finalize()

    return pe_screw

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
    
    d_pos = np.array([0.8938, 0.9664, 62.7384/2])

    pe_arr = np.zeros((N,))

    z_arr = np.zeros((N,))

    for i, _z in enumerate(z):
        lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data Lammps_Dump/Dislocations/Screw/Screw_Cylinder_Relaxed.data')

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % potfile)

        lmp.command('thermo 50')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        lmp.command('run 0')

        pe0 = lmp.get_thermo('pe')
            
        lmp.command('create_atoms 3 single %f %f %f units box' % (d_pos[0] + _z/np.sqrt(2), d_pos[1] + _z/np.sqrt(2), d_pos[2])) 

        lmp.command('minimize 1e-15 1e-18 10 10')

        lmp.command('minimize 1e-15 1e-18 10 100')

        lmp.command('minimize 1e-15 1e-18 1000 1000')

        lmp.command('write_dump all custom Lammps_Dump/Dislocations/Screw/Screw_He%d.atom id type x y z' % i)

        xyz_system = np.array(lmp.gather_atoms('x',1,3))

        xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

        pos = xyz_system[-1]

        z_arr[i] = np.linalg.norm(pos - d_pos)

        pe1 = lmp.get_thermo('pe')

        pe_arr[i] = pe0 + - pe1

    if me == 0:
        save = np.hstack([z_arr.reshape(N, 1), pe_arr.reshape(N,1)])
        np.savetxt('Test_Data/Screw_Binding.txt', save)

    # if me == 0:
    #     plt.plot(z_arr, pe_arr)
    #     plt.ylabel('Binding Energy/eV')
    #     plt.xlabel('Distance in 110 direction/ A')
    #     plt.show()
    if mode =='MPI':
        MPI.Finalize()

if __name__ == '__main__':

    alattice = 3.144221296574379
    size = 24
    init_screw(alattice, size)

    # binding(sys.argv[1])