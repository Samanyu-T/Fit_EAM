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
import os
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

    ### END GAUSSIAN SAMPLING LOOP ###




    ### OPTIMIZE FOR HE-HE POTENTIAL BY USING THE FINAL CLUSTER OF THE W-HE GMM AS A STARTING POINT ###
            
    comm.Barrier()

    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] =   bool(n_knots[2])
    bool_fit['H-He'] = False
    bool_fit['He-He'] = True
    
    # Edit a new Covariance Matrix for the He-He potential
    if me == 0:
        for cov_file in glob.glob('../W-He_102/GMM_%d/Cov*' % (N_gaussian - 1)):
            cov_0 = np.loadtxt(cov_file)
            cov_1 = np.diag([4, 8, 16, 4, 8, 16])

            cov = np.block([[cov_0, np.zeros((cov_0.shape[0], cov_1.shape[0]))], 
                           [np.zeros((cov_1.shape[0], cov_0.shape[0])), cov_1]])

            cov_name = os.path.basename(cov_file) 
            np.savetxt('../W-He_102/GMM_%d/%s' % (N_gaussian, cov_name), cov)

        for mean_file in glob.glob('../W-He_102/GMM_%d/Mean*' % (N_gaussian - 1)):
            mean_0 = np.loadtxt(mean_file)
            mean_1 = np.zeros((3,1))

            mean = np.vstack([mean_0, mean_1])

            mean_name = os.path.basename(mean_file) 
            np.savetxt('../W-He_102/GMM_%d/%s' % (N_gaussian, mean_name), mean)

    comm.Barrier()

    ### BEGIN GAUSSIAN SAMPLING FOR HE-HE POTENTIAL ###

    for i in range(N_gaussian, 2*N_gaussian):

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

    ### END GAUSSIAN SAMPLING FOR HE-HE POTENTIAL ###



    ### OPTIMIZE FOR H-HE POTENTIAL BY USING THE FINAL CLUSTER OF THE W-HE GMM AS A STARTING POINT ###
            
    comm.Barrier()

    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] =   bool(n_knots[2])
    bool_fit['H-He'] = True
    bool_fit['He-He'] = True
    
    # Edit a new Covariance Matrix for the H-He potential
    if me == 0:
        for cov_file in glob.glob('../W-He_102/GMM_%d/Cov*' % (2*N_gaussian - 1)):
            cov_0 = np.loadtxt(cov_file)
            cov_1 = np.diag([4, 8, 16, 4, 8, 16])

            cov = np.block([[cov_0, np.zeros((cov_0.shape[0], cov_1.shape[0]))], 
                           [np.zeros((cov_1.shape[0], cov_0.shape[0])), cov_1]])

            cov_name = os.path.basename(cov_file) 
            np.savetxt('../W-He_102/GMM_%d/%s' % (2*N_gaussian, cov_name), cov)

        for mean_file in glob.glob('../W-He_102/GMM_%d/Mean*' % (2*N_gaussian - 1)):
            mean_0 = np.loadtxt(mean_file)
            mean_1 = np.zeros((3,1))

            mean = np.vstack([mean_0, mean_1])

            mean_name = os.path.basename(mean_file) 
            np.savetxt('../W-He_102/GMM_%d/%s' % (N_gaussian, mean_name), mean)

    comm.Barrier()

    ### BEGIN GAUSSIAN SAMPLING FOR HE-HE POTENTIAL ###

    for i in range(2*N_gaussian, 3*N_gaussian):

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

    ### END GAUSSIAN SAMPLING FOR H-HE POTENTIAL ###
            

    comm.Barrier()

    if me == 0:
        print('\n Gaussian Sampling took %.2f s \n Start Simplex' % (t2 - t1))
        sys.stdout.flush()  

        folders = glob.glob('../W-He_102/Gaussian_Samples_%d/Core_*' % (3*N_gaussian - 1))

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

    main(param_dict['machine'], float(param_dict['max_time']) )
