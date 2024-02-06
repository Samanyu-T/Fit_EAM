from mpi4py import MPI
import Random_Sampling
import Gaussian_Sampling
import Simplex
import GMM
import sys
import json
import time

def main(machine, max_time):

    n_knots = [1,0,2]

    bool_fit = {}

    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] =   bool(n_knots[2])
    bool_fit['H-He'] = False
    bool_fit['He-He'] = False

    if me == 0:
        print('Start Random Sampling')
        sys.stdout.flush()  

    t1 = time.perf_counter()

    try:
        Random_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=max_time)
    except Exception as e:
        if me == 0:
            with open('../Error/random.txt', 'w') as error_file:
                error_file.write(str(e))

    comm.barrier()
    t2 = time.perf_counter()

    if me == 0:
        print('Random Sampling took %.2f s \n Start GMM Clustering' % (t2 - t1))
        sys.stdout.flush()  

    t1 = time.perf_counter()

    try:
        if me == 0:
            GMM.main()
    except Exception as e:
        if me == 0:
            with open('../Error/gmm.txt', 'w') as error_file:
                error_file.write(str(e))

    comm.barrier()

    t2 = time.perf_counter()

    if me == 0:
        print('Clustering took %.2f s \n Start Gaussian Sampling' % (t2 - t1))
        sys.stdout.flush()  

    t1 = time.perf_counter()

    try:
        Gaussian_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=max_time)
    except Exception as e:
        if me == 0:
            with open('../Error/gaussian.txt', 'w') as error_file:
                error_file.write(str(e))
    t2 = time.perf_counter()

    comm.barrier()

    if me == 0:
        print('Gaussian Sampling took %.2f s \n Start Simplex' % (t2 - t1))
        sys.stdout.flush()  

    try:
        Simplex.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine)
    except Exception as e:
        if me == 0:
            with open('../Error/simplex.txt', 'w') as error_file:
                error_file.write(str(e))

    comm.barrier()

    MPI.Finalize()

if __name__ == '__main__':
    global comm
    global me

    comm = MPI.COMM_WORLD

    me = comm.Get_rank()

    nprocs = comm.Get_size() 

    if me == 0:
        print('Start on %d Procs' % nprocs)
        sys.stdout.flush()  
    comm.barrier()
    
    with open(sys.argv[1], 'r') as json_file:
        param_dict = json.load(json_file)

    main(param_dict['machine'], int(param_dict['max_time']) )
