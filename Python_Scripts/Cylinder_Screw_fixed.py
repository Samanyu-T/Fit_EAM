from lammps import lammps
import numpy as np
from mpi4py import MPI
import sys

def init_screw(alattice, size):

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

    # ''' Use for 111 surface '''
    orientz = [1, 1, 1]
    orientx = [-1,-1,2]
    orienty = [1,-1, 0]

    lmp.command('lattice bcc %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % 
                (alattice,
                orientx[0], orientx[1], orientx[2],
                orienty[0], orienty[1], orienty[2], 
                orientz[0], orientz[1], orientz[2]
                ) 
                )
    
    lmp.command('region r_simbox block %d %d %d %d %d %f units lattice' % (-4*size, 4*size, -4*size, 4*size,  -1e-9, size//2 + 1e-9))

    lmp.command('region r_atombox cylinder z 0 0 %d %f %f units lattice' % (size, -1e-4, size//2 + 1e-4))

    lmp.command('create_box 3 r_simbox')

    lmp.command('create_atoms 1 region r_atombox')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    potfile = 'Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy'

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('variable radius atom sqrt(x^2+y^2)')

    lmp.command('variable select atom "v_radius  > %f" ' % (0.75*size*alattice))

    lmp.command('group fixpos variable select')

    lmp.command('fix freeze fixpos setforce 0.0 0.0 0.0')
    
    lmp.command('run 0')

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Screw/Cylinder.atom id type x y z')

    b = alattice
    
    if me == 0:
        data = np.loadtxt('Lammps_Dump/Dislocations/Screw/Cylinder.atom', skiprows=9)

        atom_pos = data[:,-3:]

        for i, pos in enumerate(atom_pos):

            x = pos[0] + 1e-2
            y = pos[1] + 1e-2

            theta = np.arctan2(y, x)

            uz = (b/(2*np.pi)) * theta
            
            data[i,-1] += uz

        starting_lines = ''

        with open('Lammps_Dump/Dislocations/Screw/Cylinder.atom', 'r') as file:
                for i in range(9):

                    if i == 1:
                        timestep = file.readline()
                        starting_lines += timestep
                        timestep = int(timestep)
                    else:
                        starting_lines += file.readline()

        with open('Lammps_Dump/Dislocations/Screw/Cylinder_Init.atom', 'w') as file:
            file.write(starting_lines)
            for i, pos in enumerate(atom_pos):
                file.write('%d %d ' % (data[i,0], data[i,1]))
                np.savetxt(file, pos, fmt = '%.5f', newline=' ')
                file.write('\n')
    else:
        timestep = None

    timestep = comm.bcast(timestep, root=0)

    lmp.command('read_dump Lammps_Dump/Dislocations/Screw/Cylinder_Init.atom %d x y z' % timestep)

    lmp.command('variable radius atom sqrt(x^2+y^2)')

    lmp.command('variable select atom "v_radius  > %f" ' % (0.75*size*alattice))

    lmp.command('group fixpos variable select')

    lmp.command('fix freeze fixpos setforce 0.0 0.0 0.0')

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Screw/Cylinder_Relaxed.atom id type x y z')

    lmp.command('write_data Lammps_Dump/Dislocations/Screw/Cylinder_Relaxed.data')

    pe_screw = lmp.get_thermo('pe')

    lmp.close()

    return pe_screw

def test(potfile, alattice):
    d_pos = np.array([0, 0, 0])

    lmp = lammps()
    
    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data Lammps_Dump/Dislocations/Screw/Cylinder_Relaxed.data')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('variable radius atom sqrt(x^2+y^2)')

    lmp.command('variable select atom "v_radius  > %f" ' % (0.5*size*alattice))

    lmp.command('group fixpos variable select')

    lmp.command('fix freeze fixpos setforce 0.0 0.0 0.0')

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('run 0')

    pe0 = lmp.get_thermo('pe')
        
    lmp.command('create_atoms 3 single %f %f %f units box' % (d_pos[0] , d_pos[1], d_pos[2])) 

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 10000 10000')

    pe1 = lmp.get_thermo('pe')

    lmp.command('write_dump all custom Lammps_Dump/Dislocations/Screw/test.atom id type x y z')

    print(pe1 - pe0, pe0 + 6.16 - pe1)

def binding(potfile, alattice, size):


    N = 8

    z = np.linspace(0, 16, N)
    
    
    d_pos = np.array([-0.5430, -1.1331, 13.7956])

    pe_arr = np.zeros((N,))

    z_arr = np.zeros((N,))

    for i, _z in enumerate(z):
        # lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp = lammps()
        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data Lammps_Dump/Dislocations/Screw/Cylinder_Relaxed.data')

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % potfile)

        lmp.command('variable radius atom sqrt(x^2+y^2)')

        lmp.command('variable select atom "v_radius  > %f" ' % (0.5*size*alattice))

        lmp.command('group fixpos variable select')

        lmp.command('fix freeze fixpos setforce 0.0 0.0 0.0')

        lmp.command('thermo 50')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        lmp.command('run 0')

        pe0 = lmp.get_thermo('pe')
            
        lmp.command('create_atoms 3 single %f %f %f units box' % (d_pos[0] , d_pos[1] + _z, d_pos[2])) 

        lmp.command('minimize 1e-15 1e-18 10 10')

        lmp.command('minimize 1e-15 1e-18 10 100')

        lmp.command('minimize 1e-15 1e-18 1000 1000')

        lmp.command('write_dump all custom Lammps_Dump/Dislocations/Screw/Screw_He%d.atom id type x y z' % i)

        xyz_system = np.array(lmp.gather_atoms('x',1,3))

        xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

        pos = xyz_system[-1]

        z_arr[i] = np.linalg.norm(pos - d_pos)

        pe1 = lmp.get_thermo('pe')

        pe_arr[i] = pe0 - pe1

    if me == 0:
        save = np.hstack([z_arr.reshape(N, 1), pe_arr.reshape(N,1)])
        np.savetxt('Test_Data/Screw_Binding.txt', save)

    # if me == 0:
    #     plt.plot(z_arr, pe_arr)
    #     plt.ylabel('Binding Energy/eV')
    #     plt.xlabel('Distance in 110 direction/ A')
    #     plt.show()

if __name__ == '__main__':
    try:
        global comm
        global me
        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        mode = 'MPI'

    except:

        me = 0
        mode = 'Serial'
    alattice = 3.144221296574379
    size = 21
    # init_screw(alattice, size) 
    test('Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy', alattice)
    # binding(sys.argv[1], alattice, size)


    if mode =='MPI':
        MPI.Finalize()
