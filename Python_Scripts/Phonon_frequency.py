from lammps import lammps
import numpy as np

def phonon_frequency(xyz_inter, size = 4 ,machine=''):

    potfile = 'Potentials/test.0.eam.alloy' 

    lmp = lammps(name = machine, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('boundary p p p')

    lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % 3.144221)

    lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (size, size, size))

    lmp.command('region r_atombox block 0 %d 0 %d 0 %d units lattice' % (size, size, size))
                
    lmp.command('create_box 3 r_simbox')
    
    lmp.command('create_atoms 1 region r_atombox')

    for element, xyz_element in enumerate(xyz_inter):
        for xyz in xyz_element:
            lmp.command('create_atoms %d single %f %f %f units lattice' % (element + 1, xyz[0], xyz[1], xyz[2]))

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    
    lmp.command('run 0')
    # lmp.command('fix 3 all box/relax  aniso 0.0')

    lmp.command('minimize 1e-9 1e-12 10 10')
    lmp.command('minimize 1e-9 1e-12 100 100')
    lmp.command('minimize 1e-9 1e-12 10000 10000')
    pe_ref = lmp.get_thermo('pe')   

    f_ref = np.copy(lmp.numpy.extract_atom('f'))

    type = lmp.numpy.extract_atom('type')

    id = len(f_ref) - 1

    lmp.command('group hydrogen id %d' % len(f_ref))
    
    alpha = 1e-5

    lmp.command('displace_atoms hydrogen move %f %f %f' % (0, 0, alpha))

    xyz = np.copy(lmp.numpy.extract_atom('x'))

    lmp.command('run 0')
    
    pe = lmp.get_thermo('pe')   

    fz = np.copy(lmp.numpy.extract_atom('f'))


    lmp.command('displace_atoms hydrogen move %f %f %f' % (0, alpha, -alpha))

    lmp.command('run 0')
    
    pe = lmp.get_thermo('pe')   

    fy = np.copy(lmp.numpy.extract_atom('f'))

    xyz = np.copy(lmp.numpy.extract_atom('x'))

    lmp.command('displace_atoms hydrogen move %f %f %f' % (alpha, -alpha, 0))

    lmp.command('run 0')
    
    pe = lmp.get_thermo('pe')   

    fx = np.copy(lmp.numpy.extract_atom('f'))

    hess = np.vstack([fx[id] - f_ref[id], fy[id]- f_ref[id], fz[id]- f_ref[id]]).T/alpha

    xyz = np.copy(lmp.numpy.extract_atom('x'))

    eig = -np.linalg.eigvals(hess) 

    freq = np.sqrt(eig[eig > 0])

    return freq

freq = phonon_frequency([[], [], []])

print(freq)

freq_saddle = phonon_frequency([[], [[3.35, 3.35, 3]], []])

freq_min = phonon_frequency([[], [[3.25, 3.5, 3]], []])

print(freq_saddle, freq_min, np.prod(freq_min)/np.prod(freq_saddle))


freq_saddle = phonon_frequency([[], [], [[3.35, 3.35, 3]]])


freq_min = phonon_frequency([[], [], [[3.25, 3.5, 3]]])

print(freq_saddle, freq_min, np.prod(freq_min)/np.prod(freq_saddle))

