from mpi4py import MPI
import Simplex
import os

comm = MPI.COMM_WORLD

me = comm.Get_rank()

nprocs = comm.Get_size() 

data_folder = '../data_102'


n_knots = [1,0,2]

bool_fit = {}

bool_fit['He_F(rho)'] = bool(n_knots[0])
bool_fit['He_rho(r)'] = bool(n_knots[1])
bool_fit['W-He'] =   bool(n_knots[2])
bool_fit['H-He'] = True
bool_fit['He-He'] = True

machine = ''

write_dir = '../Optim_Local'

simplex_folder = os.path.join(data_folder, 'Simplex/Core_%d' % me)
Simplex.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, simplex_folder=simplex_folder, write_dir=write_dir)
