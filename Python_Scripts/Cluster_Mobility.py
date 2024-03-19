from lammps import lammps
import numpy as np
import os
from mpi4py import MPI
import glob


def temp_md(filepath, temp=800, machine=''):

    potfile = 'Potentials/test.0.eam.alloy'

    lmp = lammps(name = machine, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data %s' % filepath)

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    
    lmp.command('run 0')

    pe_ref = lmp.get_thermo('pe')
    
    lmp.command('timestep 1e-3')

    lmp.command('fix nve_fix all nve') 

    rngint = np.random.randint(0 , 10000)
    
    lmp.command('velocity all create %f %d rot no dist gaussian' % (2*temp, rngint) )

    type = np.array( lmp.gather_atoms('type', 0 , 1) )  
    
    N = 1000

    tstep = 100

    light_idx = np.where(type != 1)[0]

    data_xyz = np.zeros( (N + 1, len(light_idx), 3) ) 

    data_pe = np.zeros( (N + 1,) )

    xyz = np.array(lmp.gather_atoms('x', 1, 3))

    xyz = xyz.reshape(len(xyz)//3, 3)

    data_xyz[0] = xyz[light_idx]

    pe = lmp.get_thermo('pe')

    data_pe[0] = pe

    for i in range(1,N+1):

        lmp.command('run %d' % tstep)

        xyz = np.array(lmp.gather_atoms('x', 1, 3))

        xyz = xyz.reshape(len(xyz)//3, 3)

        data_xyz[i] = xyz[light_idx]

        pe = lmp.get_thermo('pe')

        data_pe[i] = pe

    filename = os.path.basename(filepath).split('.')[0].split('_')[0]

    if not os.path.exists('../Migration_Data'):
        os.mkdir('../Migration_Data')

    np.save('../Migration_Data/%s_xyz.npy' % filename, data_xyz)
    np.save('../Migration_Data/%s_pe.npy' % filename, data_pe)



if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    size = comm.Get_size() 

    data_folder = '../H_Surface_Data'

    if rank == 0:
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        
        print('Start on %d Procs' % size)

    comm.Barrier()

    for filename in glob.glob('../HeH_Clusters_New/V0H0He4_new.data'):
        print(filename)
        temp_md(filename, 1000, '')

