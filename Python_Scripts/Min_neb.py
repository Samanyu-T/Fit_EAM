import numpy as np
from lammps import lammps
from mpi4py import MPI
import itertools
import copy 
import os 

def lmp_minimize(init_file, read_file, potfile, he_lst, pe_lst, machine = ''):

    lmp = lammps(name = machine,cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data %s' % init_file)

    with open(read_file, 'r') as file:

        file.readline()
        timestep = int(file.readline())

    if mode == 'MPI':
        comm.barrier()


    lmp.command('read_dump %s %d x y z' % (read_file, timestep) )

    lmp.command('mass 1 183.84')

    lmp.command('mass 2 1.00784')

    lmp.command('mass 3 4.002602')

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('thermo 50')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

    lmp.command('minimize 1e-15 1e-18 10 10')

    lmp.command('minimize 1e-15 1e-18 10 100')

    lmp.command('minimize 1e-15 1e-18 1000 10000')

    N = lmp.get_natoms()

    add_bool = True

    id = lmp.numpy.extract_atom('type')
    xyz = lmp.numpy.extract_atom('x')

    he_idx= np.where(id == 3)[0]

    pe = lmp.get_thermo('pe')

    if len(he_idx) > 0:

        id = comm.bcast(id, me)
        xyz = comm.bcast(xyz, me)

        for i in range(len(he_lst)):

            # print(read_file, he_lst,he_idx)
            print(he_lst[i], xyz[he_idx[0]])

            if np.linalg.norm( he_lst[i][-1] - xyz[he_idx[0]][-1] ) < 0.5:
                add_bool = False
                add_bool = comm.bcast(add_bool, me)

    comm.barrier()
    if add_bool:
        he_lst.append(xyz[he_idx[0]])
        pe_lst.append(pe)

        folder = os.path.dirname(os.path.dirname(read_file))

        filename = 'New_Depth_%d' % (len(he_lst) - 1)
        filepath  = os.path.join(folder, filename)

        lmp.command('write_data %s.data' % filepath)
        lmp.command('write_dump all custom %s.atom id x y z' % filepath)

        if me == 0: 
            with open('%s.atom' % filepath, 'r') as file:
                lines = file.readlines()

            with open('%s.atom' % filepath, 'w') as file:
                file.write(lines[3])
                file.writelines(lines[9:])

    comm.barrier()

    lmp.close()
    return he_lst, pe_lst

if __name__ == '__main__':

    global comm
    global me
    global mode

    try:
        comm = MPI.COMM_WORLD

        me = comm.Get_rank()

        nprocs = comm.Get_size()

        mode = 'MPI' 
    except:
        comm = None
        me = 0
        mode = 'serial'

    for orient_folder in os.listdir('../Lammps_Dump/Surface'):
        init_file = os.path.join(orient_folder, 'Depth_0.data')
    
    he_lst = []
    pe_lst = []

    surface_folder = '../Lammps_Dump/Surface'

    lst_orient_folders = [os.path.join(surface_folder, folder) for folder in os.listdir(surface_folder)
                      if os.path.isdir(os.path.join(surface_folder, folder))]

    for orient_folder in lst_orient_folders:
        init_file = os.path.join(orient_folder, 'Depth_0.data')

        lst_neb_folders =  [os.path.join(orient_folder, folder) for folder in os.listdir(orient_folder)
                        if os.path.isdir(os.path.join(orient_folder, folder))]
        
        sorted_idx = np.argsort(np.array([int(os.path.basename(folder).split('_')[0]) for folder in lst_neb_folders])).astype('int')

        sorted_neb_folders = [lst_neb_folders[i] for i in sorted_idx]
        
        for neb_folder in sorted_neb_folders:

            lst_neb_files = sorted([os.path.join(neb_folder,file) for file in os.listdir(neb_folder) if file.startswith('neb')])
            for neb_file in lst_neb_files:
                he_lst, pe_lst = lmp_minimize(init_file, neb_file, 'Potentials/WHHe_test.eam.alloy', he_lst, pe_lst)

    if mode == 'MPI':
        MPI.Finalize()