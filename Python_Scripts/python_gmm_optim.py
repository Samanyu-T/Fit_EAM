from random import sample
from mpi4py import MPI
import Random_Sampling
import Gaussian_Sampling
import Simplex
import GMM
import sys
import json
import time
import glob
import numpy as np

def main(machine, max_time):

    n_knots = [1,0,2]

    bool_fit = {}

    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] =   bool(n_knots[2])
    bool_fit['H-He'] = False
    bool_fit['He-He'] = False



    ### START RANDOM SAMPLING ###
    if me == 0:
        print('Start Random Sampling \n')
        sys.stdout.flush()  

    comm.Barrier()

    t1 = time.perf_counter()

    try:
        Random_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=max_time)
    except Exception as e:
        if me == 0:
            with open('../Error/random.txt', 'w') as error_file:
                error_file.write(str(e))

    t2 = time.perf_counter()

    if me == 0:
        print('Random Sampling took %.2f s \n' % (t2 - t1))
        sys.stdout.flush()  
    ### END RANDOM SAMPLING ###

    comm.Barrier()



    ### START CLUSTERING ALGORITHM ###
    if me == 0:
        print('Start GMM Clustering \n')
        sys.stdout.flush()  

    comm.Barrier()

    t1 = time.perf_counter()

    try:
        if me == 0:
            GMM.main('../W-He_102/Random_Samples/Core_*/Filtered_Samples.txt', 0)
    except Exception as e:
        if me == 0:
            with open('../Error/gmm.txt', 'w') as error_file:
                error_file.write(str(e))

    t2 = time.perf_counter()

    if me == 0:
        print('\n Clustering took %.2f s ' % (t2 - t1))
        sys.stdout.flush()  

    ### END CLUSTERING ALGORITHM ###
        
    comm.Barrier()



    ### START GAUSSIAN SAMPLING LOOP ###
    
    N_gaussian = 5

    for i in range(N_gaussian):

        if me == 0:
            print('Start Gaussian Sampling %dth iteration' % i)
            sys.stdout.flush()  

        comm.Barrier()

        t1 = time.perf_counter()

        try:
            Gaussian_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=max_time, iter=i)
        except Exception as e:
            if me == 0:
                with open('../Error/gaussian.txt', 'w') as error_file:
                    error_file.write(str(e))

        t2 = time.perf_counter()

        if me == 0:
            print('End Gaussian Sampling %dth iteration it took %.2f' % (i, t2- t1))
            sys.stdout.flush()  

        comm.Barrier()

        t1 = time.perf_counter()

        try:
            if me == 0:
                GMM.main('../W-He_102/Gaussian_Samples_%d/Core_*/Filtered_Samples.txt' % i, i + 1)
        except Exception as e:
            if me == 0:
                with open('../Error/gmm.txt', 'w') as error_file:
                    error_file.write(str(e))

        t2 = time.perf_counter()

        if me == 0:
            print('\n Clustering took %.2f s ' % (t2 - t1))
            sys.stdout.flush()  

        comm.Barrier()
    ### END GAUSSIAN SAMPLING LOOP ###



    if me == 0:
        print('\n Gaussian Sampling took %.2f s \n Start Simplex' % (t2 - t1))
        sys.stdout.flush()  

        folders = glob.glob('../W-He_102/Gaussian_Samples_%d/Core_*' % (N_gaussian - 1))

        lst_samples = []
        lst_loss = []
        for folder in folders:
            lst_loss.append(np.loadtxt('%s/Filtered_Loss.txt' % folder))
            lst_samples.append(np.loadtxt('%s/Filtered_Samples.txt' % folder))

        loss = np.hstack(lst_loss).reshape(-1, 1)
        samples = np.vstack(lst_samples)

        N_simplex = 5 

        if nprocs >= len(loss):

            for i in range(len(loss)):
                np.savetxt('%s/Simplex_Init.txt' % folders[i], samples[i])

            for i in range(len(loss), nprocs):
                with open('%s/Simplex_Init.txt', 'w') as file:
                    file.write('')

        elif len(loss) > nprocs and N_simplex*nprocs > len(loss):
            part = len(loss) // nprocs
            idx = 0

            for i in range(nprocs - 1):
                np.savetxt('%s/Simplex_Init.txt' % folders[i], samples[idx: idx + part])
                idx += part

            np.savetxt('%s/Simplex_Init.txt' % folders[i], samples[idx:])

        else:
            part = N_simplex
            sort_idx = np.argsort(loss).flatten()
            loss = loss[sort_idx]
            samples = samples[sort_idx]

            idx = 0
            for i in range(nprocs):
                np.savetxt('%s/Simplex_Init.txt' % folders[i], samples[idx: idx + part])
                idx += part

    comm.Barrier()

    try:
        Simplex.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine)
    except Exception as e:
        if me == 0:
            with open('../Error/simplex.txt', 'w') as error_file:
                error_file.write(str(e))

if __name__ == '__main__':
    global comm
    global me
    global nprocs

    comm = MPI.COMM_WORLD

    me = comm.Get_rank()

    nprocs = comm.Get_size() 

    if me == 0:
        print('Start on %d Procs' % nprocs)
        sys.stdout.flush()  
    comm.Barrier()
    
    with open(sys.argv[1], 'r') as json_file:
        param_dict = json.load(json_file)

    main(param_dict['machine'], int(param_dict['max_time']) )
