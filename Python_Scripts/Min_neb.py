import numpy as np
from lammps import lammps
from mpi4py import MPI
import itertools
import copy 
import os 
import glob
import sys

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

    filepath = ''

    if len(he_idx) > 0:

        # print(he_lst, xyz[he_idx[0]])

        for i in range(len(he_lst)):

            if np.linalg.norm( he_lst[i][-1] - xyz[he_idx[0]][-1]) < 0.4:
                add_bool = False
        
        if add_bool:
            he_lst.append(xyz[he_idx[0]])
            pe_lst.append(pe)

            folder = os.path.dirname(os.path.dirname(read_file))

            filename = 'New_Depth_%d' % (len(he_lst) - 1)
            filepath  = os.path.join(folder, filename)

    comm.barrier()

    he_cores = comm.gather(len(he_idx), root=0)
    root = None
    if me == 0:
        root = he_cores.index(1)
    comm.barrier()

    root = comm.bcast(root, 0)
    add_bool = comm.bcast(add_bool, root)
    filepath = comm.bcast(filepath, root)
    he_lst = comm.bcast(he_lst, root)
    pe_lst = comm.bcast(pe_lst, root)

    if add_bool:
        lmp.command('write_data %s.data' % filepath)
        lmp.command('write_dump all custom %s.atom id x y z' % filepath)
    
        if me == root:
            with open('%s.atom' % filepath, 'r') as file:
                lines = file.readlines()

            with open('%s.atom' % filepath, 'w') as file:
                file.write(lines[3])
                file.writelines(lines[9:])

    comm.barrier()

    lmp.close()
    return he_lst, pe_lst

if __name__ == '__main__':

    if len(sys.argv) > 1:
        potfile = sys.argv[1]
    else:
        potfile = 'Potentials/WHHe_test.eam.alloy'

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
        he_lst = []
        pe_lst = []
        init_file = os.path.join(orient_folder, 'Depth_0.data')

        lst_neb_folders =  [os.path.join(orient_folder, folder) for folder in os.listdir(orient_folder)
                        if os.path.isdir(os.path.join(orient_folder, folder)) and not folder.startswith('new')]
        
        sorted_idx = np.argsort(np.array([int(os.path.basename(folder).split('_')[0]) for folder in lst_neb_folders] )).astype('int')
        
        sorted_neb_folders = [lst_neb_folders[i] for i in sorted_idx]

        # print(sorted_neb_folders)

        for neb_folder in sorted_neb_folders:

            lst_neb_files = sorted([os.path.join(neb_folder,file) for file in os.listdir(neb_folder) if file.startswith('neb')])
            for neb_file in lst_neb_files:
                he_lst, pe_lst = lmp_minimize(init_file, neb_file, potfile, he_lst, pe_lst)

        if me == 0:
            neb_script_folder = os.path.join('../Lammps_Scripts' , orient_folder.split('/')[-1])

            if not os.path.exists(neb_script_folder):
                os.makedirs(neb_script_folder)      

            new_neb_files = glob.glob(os.path.join(orient_folder,'New_Depth_*.data'))

            for i in range(len(new_neb_files)):

                if not os.path.exists(os.path.join(orient_folder, 'new_%d_%d' % (i, i + 1))):
                    os.mkdir(os.path.join(orient_folder, 'new_%d_%d' % (i, i + 1)))

                txt = '''
units metal 

atom_style atomic

atom_modify map array sort 0 0.0

read_data %s

mass 1 183.84

mass 2 1.00784

mass 3 4.002602

pair_style eam/alloy

pair_coeff * * %s W H He

thermo 10

run 0

fix 1 all neb 1e-4

timestep 1e-3

min_style quickmin

thermo 100 

variable i equal part

neb 10e-8 10e-10 5000 5000 100 final %s

write_dump all custom %s/neb.$i.dump id type x y z ''' % (new_neb_files[i], potfile, 
                                                          os.path.join(orient_folder, 'New_Depth_%d.atom' % (i +1)), 
                                                          os.path.join(orient_folder, 'new_%d_%d' % (i, i + 1)))
                
                with open(os.path.join(neb_script_folder, 'surface-new_%d_%d.neb' % (i, i + 1)), 'w') as file:
                    file.write(txt)
            

    if mode == 'MPI':
        MPI.Finalize()