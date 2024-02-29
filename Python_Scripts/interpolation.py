from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt

alattice = 3.144221
size = 7
potfile = 'Potentials/WHHe_test.eam.alloy'

lmp = lammps(name = '', cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])


lmp.command('# Lammps input file')

lmp.command('units metal')

lmp.command('atom_style atomic')

lmp.command('atom_modify map array sort 0 0.0')

lmp.command('boundary p p p')

lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % alattice)

lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (size, size, size))

lmp.command('region r_atombox block 0 %d 0 %d 0 %d units lattice' % (size, size, size))
            
lmp.command('create_box 3 r_simbox')

lmp.command('create_atoms 1 region r_atombox')

lmp.command('mass 1 183.84')

lmp.command('mass 2 1.00784')

lmp.command('mass 3 4.002602')

lmp.command('pair_style eam/alloy' )

lmp.command('pair_coeff * * %s W H He' % potfile)

# lmp.command('fix 3 all box/relax  aniso 0.0')

lmp.command('run 0')

lmp.command('thermo 5')

lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

# lmp.command('minimize 1e-12 1e-16 10000 100000')

N_atoms = lmp.get_natoms()

atom_to_add = 3

tet_1 = np.array([0.25, 0.5, 0]) + 3
tet_2 = np.array([0.5, 0.25, 0]) + 3
tet_3 = np.array([0,0.5, 0.25]) + 3

oct = np.array([0.5, 0.5, 0]) + 3
inter = np.array([0.375, 0.375, 0]) + 3

diag1 = np.array([0.0, 0.5, 0.0]) + 3
diag2 = np.array([0.15, 0.5, 0.15]) + 3

N = 15

sites = np.hstack( [
    np.linspace(tet_1[0], tet_2[0], N).reshape(-1, 1),
    np.linspace(tet_1[1], tet_2[1], N).reshape(-1, 1),
    np.linspace(tet_1[2], tet_2[2], N).reshape(-1, 1)

])

print( sites )

pe_lst = []


for site in sites:
    
    site *= alattice

    lmp.command('create_atoms %d single %f %f %f units box' % (atom_to_add, site[0], site[1], site[2]))

    lmp.command('run 0')

    lmp.command('minimize 1e-12 1e-16 10 10')
    lmp.command('minimize 1e-12 1e-16 10 10')
    lmp.command('minimize 1e-12 1e-16 100 1000')

    pe_lst.append(lmp.get_thermo('pe'))

    print(lmp.get_thermo('pe'))

    xyz_system = np.array(lmp.gather_atoms('x',1,3))

    xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

    # print(xyz_system[-1]/alattice)

    lmp.command('group delete id %d' % (N_atoms + 1) )

    lmp.command('delete_atoms group delete')

pe = np.array(pe_lst)
pe -= pe.min()
plt.plot(pe)
plt.show()