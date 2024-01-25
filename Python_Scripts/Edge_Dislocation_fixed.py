from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt 
from mpi4py import MPI
import sys

def init_screw(alattice, size):
    
    C11 = 3.229
    C12 = 1.224
    C44 = 0.888

    C_voigt = np.array([[C11, C12, C12, 0, 0, 0],
                        [C12, C11, C12, 0, 0, 0],
                        [C12, C12, C11, 0, 0, 0],
                        [0, 0, 0, C44, 0, 0],
                        [0, 0, 0, 0, C44, 0],
                        [0, 0, 0, 0, 0, C44]])

    Cinv = np.linalg.inv(C_voigt)

    shear = C44
    v =  -Cinv[0,1]/Cinv[0,0]
    youngs = 1/Cinv[0,0]

    lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

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
    orientx = [1, 1, 1]
    orienty = [-1,-1,2]
    orientz = [1,-1, 0]

    lmp.command('lattice bcc %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % 
                (alattice,
                orientx[0], orientx[1], orientx[2],
                orienty[0], orienty[1], orienty[2], 
                orientz[0], orientz[1], orientz[2]
                ) 
                )
    
    lmp.command('region r_simbox block %f %f %f %f %f %f units lattice' % (

        -1e-9 - size/2 - 10, size/2 + 1e-9 + 10, -1e-9 - size/2 - 10, size/2 + 1e-9 + 10, -1e-9, size + 1e-9
    ))

    lmp.command('region r_atombox block %f %f %f %f %f %f units lattice' % (

        -1e-4 - size/2, size/2 + 1e-4, -1e-4 - size/2, size/2 + 1e-4, -1e-4, size + 1e-4
    ))


    lmp.command('create_box 3 r_simbox')

    lmp.command('create_atoms 1 region r_atombox')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    potfile = 'Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy'

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('variable radius atom (x^%d+y^%d)^(1/%d)' % (norm, norm, norm))

    lmp.command('variable select atom "v_radius  > %f" ' % (alattice*(size - k)))

    lmp.command('group fixpos variable select')

    lmp.command('fix freeze fixpos setforce 0.0 0.0 0.0')
    
    lmp.command('run 0')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Edge/Box.atom id type x y z')


    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    pe0 = lmp.get_thermo('pe')

    b = alattice

    if me == 0:
        data = np.loadtxt('Lammps_Dump/Dislocations/Edge/Box.atom', skiprows=9)

        atom_pos = data[:,-3:]

        for i, pos in enumerate(atom_pos):

            x = pos[0] + 1e-2
            y = pos[1] + 1e-2

            theta = np.arctan2(y, x)

            # Ensure the angle is between 0 and 2*pi
            # theta = (theta + 2 * np.pi) % (2 * np.pi)

            ux = (b/(2*np.pi)) * (theta + (x*y)/(2*(1 - v)*(x**2 + y**2)))

            uy = -(b/(2*np.pi)) * ( (1 - 2*v)/(4 - 4*v) * np.log(x**2 + y**2) + (x**2 - y**2)/(4*(1 - v)*(x**2 + y**2)))

            data[i,-3] += ux
            
            data[i,-2] += uy

        starting_lines = ''

        with open('Lammps_Dump/Dislocations/Edge/Box.atom', 'r') as file:
                for i in range(9):

                    if i == 1:
                        timestep = file.readline()
                        starting_lines += timestep
                        timestep = int(timestep)
                    else:
                        starting_lines += file.readline()

        with open('Lammps_Dump/Dislocations/Edge/Box_Init.atom', 'w') as file:
            file.write(starting_lines)
            for i, pos in enumerate(atom_pos):
                file.write('%d %d ' % (data[i,0], data[i,1]))
                np.savetxt(file, pos, fmt = '%.5f', newline=' ')
                file.write('\n')
    else:
        timestep = None

    timestep = comm.bcast(timestep, root=0)

    lmp.command('read_dump Lammps_Dump/Dislocations/Edge/Box_Init.atom %d x y z' % timestep)

    # lmp.command('fix 1 all box/relax aniso 0.0')
    
    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Edge/Box_Relaxed.atom id type x y z')

    lmp.command('write_data Lammps_Dump/Dislocations/Edge/Box_Relaxed.data')

    pe1 = lmp.get_thermo('pe')

    print( (pe1 - pe0)/size )

    lmp.close()

def binding(potfile, alattice, size):

    tet = np.array([0.25, 0.5, 0])

    orientx = [1, 1, 1]
    orienty = [-1,-1,2]
    orientz = [1,-1, 0]

    R_inv = np.vstack([orientx, orienty, orientz]).T
    R = np.linalg.inv(R_inv)

    tet_new = R @ tet

    tet_new = (tet_new + 2) % 1

    N = 8

    z = np.linspace(0,16, N)
    
    d_pos = np.array([5.2082, -1.2843, tet_new[-1]*alattice*np.linalg.norm(orientz)])

    # d_pos = alattice*tet_new*np.array([np.linalg.norm(orientx),np.linalg.norm(orienty),np.linalg.norm(orientz)])

    pe_arr = np.zeros((N,))

    z_arr = np.zeros((N,))

    for i, _z in enumerate(z):
        # lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])
        lmp = lammps()
        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data Lammps_Dump/Dislocations/Edge/Box_Relaxed.data')

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % potfile)

        lmp.command('variable radius atom (x^%d+y^%d)^(1/%d)' % (norm, norm, norm))

        lmp.command('variable select atom "v_radius  > %f" ' % (alattice*(size - k)))

        lmp.command('group fixpos variable select')

        lmp.command('fix freeze fixpos setforce 0.0 0.0 0.0')
        
        lmp.command('thermo 50')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        lmp.command('run 0')

        pe0 = lmp.get_thermo('pe')
            
        lmp.command('create_atoms 3 single %f %f %f units box' % (d_pos[0] + _z, d_pos[1], d_pos[2])) 

        lmp.command('minimize 1e-15 1e-18 10 10')

        lmp.command('minimize 1e-15 1e-18 10 100')

        lmp.command('minimize 1e-15 1e-18 1000 1000')

        lmp.command('write_dump all custom Lammps_Dump/Dislocations/Edge/Edge_He%d.atom id type x y z' % i)

        xyz_system = np.array(lmp.gather_atoms('x',1,3))

        xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

        pos = xyz_system[-1]

        z_arr[i] = np.linalg.norm(pos - d_pos)

        pe1 = lmp.get_thermo('pe')

        pe_arr[i] = pe0 - pe1

    if me == 0:
        save = np.hstack([z_arr.reshape(N, 1), pe_arr.reshape(N,1)])
        np.savetxt('Test_Data/Edge_Binding.txt', save)

    # if me == 0:
    #     plt.plot(z_arr, pe_arr)
    #     plt.ylabel('Binding Energy/eV')
    #     plt.xlabel('Distance in 110 direction/ A')
    #     plt.show()

    return pe_arr[0]
if __name__ == '__main__':
    try:
        global comm
        global me
        global k 
        global norm
        norm = 10*2
        k = 5
        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        mode = 'MPI'

    except:

        me = 0
        mode = 'Serial'

    alattice = 3.144221296574379
    size = 21

    # freeze = np.arange(3,10)
    # e = []
    # for i, _f in enumerate(freeze):

    #     k = _f
    # init_screw(alattice, size)

        # e.append(binding(sys.argv[1], alattice, size))
    binding(sys.argv[1], alattice, size)
    # if me == 0:
    #     plt.plot(freeze, np.array(e))
    #     plt.show()
    # comm.barrier()

    if mode =='MPI':
        MPI.Finalize()
