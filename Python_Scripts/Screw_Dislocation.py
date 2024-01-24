from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt 
from mpi4py import MPI
alattice = 3.144221296574379
size = 25

def init_screw(alattice, size):
    lmp = lammps()

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('boundary p p p')

    lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % alattice)

    lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (size, size, size))

    lmp.command('create_box 3 r_simbox')

    lmp.command('create_atoms 1 region r_simbox')

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    potfile = 'Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy'

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('run 0')

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Perfect.atom id type x y z')

    pe_init = lmp.get_thermo('pe')

    data = np.loadtxt('Lammps_Dump/Perfect.atom', skiprows=9)

    atom_pos = data[:,-3:]

    lspace = np.linspace(0, size, 4)

    screw_centres = alattice*np.array([[int(lspace[1]), size//2, size//2], [int(lspace[2]), size//2, size//2]])

    screw_idx = []
    screw_pos = []

    for i, pos in enumerate(atom_pos):

        for centre in screw_centres:
            if np.linalg.norm(pos - centre) < 0.1:
                screw_idx.append(i)
                screw_pos.append(pos)

    b = 20

    for i, pos in enumerate(atom_pos):

        theta1 = np.arctan2(pos[1] - screw_pos[0][1], pos[0] - screw_pos[0][0])
        # Ensure the angle is between 0 and 2*pi
        theta1 = (theta1 + 2 * np.pi) % (2 * np.pi)


        theta2 = np.arctan2(pos[1] - screw_pos[1][1], pos[0] - screw_pos[1][0])
        # Ensure the angle is between 0 and 2*pi
        theta2 = (theta2 + 2 * np.pi) % (2 * np.pi)

        data[i,-1] += (b/(2*np.pi))*(theta1 - theta2)

    starting_lines = ''

    with open('Lammps_Dump/Perfect.atom', 'r') as file:
            for i in range(9):
                starting_lines += file.readline()

    with open('Lammps_Dump/Screw_Dipole_Init.atom', 'w') as file:
        file.write(starting_lines)
        for i, pos in enumerate(atom_pos):
            file.write('%d %d ' % (data[i,0], data[i,1]))
            np.savetxt(file, pos, fmt = '%.5f', newline=' ')
            file.write('\n')

    lmp.command('read_dump Lammps_Dump/Screw_Dipole_Init.atom 3 x y z')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    lmp.command('write_dump all custom Lammps_Dump/Screw_Dipole_Relaxed.atom id type x y z')

    lmp.command('write_data Lammps_Dump/Screw_Dipole_Relaxed.data')

    pe_screw = lmp.get_thermo('pe')

    lmp.close()

    return pe_screw

def bind_he_screw(alattice,start_file, dist_to_screw):

    lmp = lammps()

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data %s' % start_file)

    lmp.command('pair_style eam/alloy' )

    potfile = 'Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy'

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('run 0')

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    he_pos = np.array([65.3568 + dist_to_screw, 55.2753, 75.0894])

    lmp.command('create_atoms 3 single %f %f %f units box' % (he_pos[0], he_pos[1], he_pos[2]) )

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    pe_he = lmp.get_thermo('pe')

    lmp.command('write_dump all custom Lammps_Dump/Screw_Dipole_Helium.atom id type x y z')
    
    lmp.close()

    return pe_he

try:

    comm = MPI.COMM_WORLD

    me = comm.Get_rank()

    mode = 'MPI'

except:

    me = 0
    mode = 'Serial'

pe_screw = init_screw(alattice, size)
dist = np.linspace(0.25, 6, 10)
pe_he = np.zeros((10,))

for i, z in enumerate(dist):
    pe_he[i] = bind_he_screw(alattice, 'Lammps_Dump/Screw_Dipole_Relaxed.data', z)
    
binding = 6.16 + pe_screw - pe_he

if me == 0:
    print(binding)
    plt.plot(dist, binding)

if mode == 'MPI':
    MPI.Finalize()