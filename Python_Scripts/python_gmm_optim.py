from mpi4py import MPI
import Random_Sampling
import Gaussian_Sampling
import Simplex
import GMM
import sys
import json

def main(machine, max_time):

    comm = MPI.COMM_WORLD

    me = comm.Get_rank()

    nprocs = comm.Get_size()

    n_knots = [1,0,2]

    bool_fit = {}

    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] =   bool(n_knots[2])
    bool_fit['H-He'] = False
    bool_fit['He-He'] = False

    Random_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=max_time)
    comm.barrier()

    if me == 0:
        GMM.main()

    comm.barrier()

    Gaussian_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=max_time)
    comm.barrier()

    Simplex.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine)
    comm.barrier()

    MPI.Finalize()

if __name__ == '__main__':

    with open(sys.argv[1], 'r') as json_file:
        param_dict = json.load(json_file)

    main(param_dict['machine'], int(param_dict['max_time']) )