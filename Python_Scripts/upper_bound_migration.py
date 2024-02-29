from lammps import lammps
import numpy as np

def eval_lammps(site, write_name):

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

    lmp.command('run 0')

    lmp.command('thermo 5')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    atom_to_add = 3

    site *= alattice

    lmp.command('create_atoms %d single %f %f %f units box' % (atom_to_add, site[0], site[1], site[2]))

    lmp.command('run 0')

    lmp.command('minimize 1e-12 1e-16 10 10')
    lmp.command('minimize 1e-12 1e-16 10 10')
    lmp.command('minimize 1e-12 1e-16 1000 10000')

    pe = lmp.get_thermo('pe')

    lmp.command('write_dump all custom %s id type xu yu zu' % write_name)
    
    print(pe)

    xyz_system = np.array(lmp.gather_atoms('x',1,3))

    xyz_system = xyz_system.reshape(len(xyz_system)//3,3)
    
    lmp.close()

    return pe, xyz_system

def eval_lammps_given_dump():

    lmp = lammps(name = '', cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])
    # lmp =lammps()

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

    lmp.command('run 0')

    lmp.command('thermo 5')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('create_atoms %d single %f %f %f units lattice' % (3, 3.25, 3.5, 3))

    lmp.command('read_dump neb_upper.atom 1 x y z')

    lmp.command('run 0')

    pe = lmp.get_thermo('pe')

    print(pe)

    xyz_system = np.array(lmp.gather_atoms('x',1,3))

    xyz_system = xyz_system.reshape(len(xyz_system)//3,3)
    
    lmp.close()

    return pe, xyz_system


alattice = 3.144221
size = 7
potfile = 'Potentials/WHHe_test.eam.alloy'


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

optim_pos = []
name = ['tet1.atom', 'tet2.atom', 'inter.atom']
for i, site in enumerate([tet_1, tet_2, inter]):
    pe, xyz = eval_lammps(site, name[i])

    optim_pos.append(xyz)
    pe_lst.append(pe)

optim_pos[0] = np.loadtxt('tet1.atom', skiprows=9, usecols=[2, 3, 4])
optim_pos[1] = np.loadtxt('tet2.atom', skiprows=9, usecols=[2, 3, 4])

print(optim_pos[0][-1], optim_pos[1][-1], optim_pos[2][-1])

print( (optim_pos[0][-1] + optim_pos[1][-1]) /2)

with open('neb_upper.atom', 'w') as file:
    file.write('''ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
687
ITEM: BOX BOUNDS pp pp pp
-1.0380103268104445e-06 2.2009548038010326e+01
-1.0380103270892315e-06 2.2009548038010323e+01
-1.0380103266153252e-06 2.2009548038010326e+01
ITEM: ATOMS id type x y z
''')
    id = np.arange(1, 688).reshape(-1, 1)
    type = np.ones(id.shape)
    type[-1] = 3

    he_av = (optim_pos[0][-1] + optim_pos[1][-1]) /2
    he_int =  optim_pos[2][-1]

    xyz = (optim_pos[0] + optim_pos[1])/2 + he_int - he_av

    # xyz = optim_pos[0]
    # xyz[-1] = he_int

    data = np.hstack([id, type, xyz])

    for i in range(len(id)):
        file.write('%d %d %.4f %.4f %.4f \n' % (id[i], type[i], xyz[i, 0], xyz[i, 1], xyz[i, 2]))    

pe, xyz = eval_lammps_given_dump()

print(- pe_lst[1] + pe, -pe_lst[1] + pe_lst[-1])