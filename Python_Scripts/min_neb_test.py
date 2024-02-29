from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt

alattice = 3.144221
size = 7
potfile = 'Potentials/WHHe_test.eam.alloy'

lmp = lammps()


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

lmp.command('create_atoms %d single %f %f %f units lattice' % (3, 3.25, 3.5, 3))

lmp.command('run 0')

lmp.command('minimize 1e-12 1e-16 10 10')

lmp.command('minimize 1e-12 1e-16 10 10')

lmp.command('minimize 1e-12 1e-16 100 1000')

pe1 = lmp.get_thermo('pe')

lmp.command('read_dump ../Neb_Dump/Bulk/Tet_Tet/Neb_Images/neb.3.atom 39 x y z ')

lmp.command('run 0')

lmp.command('thermo 5')

lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

lmp.command('minimize 1e-12 1e-16 10 10')
lmp.command('minimize 1e-12 1e-16 10 10')
lmp.command('minimize 1e-12 1e-16 100 1000')

pe2 = lmp.get_thermo('pe')

print(pe2 - pe1)