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
import psutil


def main(machine, max_time, write_dir, save_dir):

    n_knots = [1,0,2]

    data_folder = ''

    if me == 0:
    # Get memory usage statistics
        memory = psutil.virtual_memory()

        # Total physical memory
        total_memory = memory.total / (1024 ** 3)  # Convert bytes to GB

        # Available physical memory
        available_memory = memory.available / (1024 ** 3)  # Convert bytes to GB

        # Used physical memory
        used_memory = memory.used / (1024 ** 3)  # Convert bytes to GB

        # Percentage of memory usage
        memory_percent = memory.percent

        print("Total Memory:", total_memory, "GB")
        print("Available Memory:", available_memory, "GB")
        print("Used Memory:", used_memory, "GB")
        print("Memory Usage Percentage:", memory_percent, "%")


        # Init Output locations
        data_folder = os.path.join(save_dir, 'data_%d%d%d' % (n_knots[0], n_knots[1], n_knots[2]))
        
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

    comm.Barrier()
    data_folder = comm.bcast(data_folder, root = 0)

    bool_fit = {}

    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] =   bool(n_knots[2])
    bool_fit['H-He'] = False
    bool_fit['He-He'] = False


    ### START RANDOM SAMPLING ###
    # rsamples_folder = ''

    # if me == 0:
    #     print('Start Random Sampling \n')
    #     sys.stdout.flush()  

    #     rsamples_folder = os.path.join(data_folder, 'Random_Samples') 

    #     if not os.path.exists(rsamples_folder):
    #         os.mkdir(rsamples_folder)

    # request = comm.Ibarrier()  # Non-blocking barrier

    # rsamples_folder = comm.bcast(rsamples_folder, root = 0)

    # t1 = time.perf_counter()

    # Random_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine,
    #                          max_time=0.49*max_time, write_dir=write_dir, sample_folder=rsamples_folder)

    # t2 = time.perf_counter()

    # # Wait for the barrier to complete
    # request.Wait()
    
    # if me == 0:
    #     print('Random Sampling took %.2f s \n' % (t2 - t1))
    #     sys.stdout.flush()  
    
    # comm.Barrier()

    # ### END RANDOM SAMPLING ###




    # ### START CLUSTERING ALGORITHM ###
    # if me == 0:
    #     print('Start GMM Clustering \n')
    #     sys.stdout.flush()  

    #     t1 = time.perf_counter()

    #     GMM.main(os.path.join(rsamples_folder,'Core_*'), data_folder, 0)

    #     t2 = time.perf_counter()

    #     print('\n Clustering took %.2f s ' % (t2 - t1))
    #     sys.stdout.flush()  

    # comm.Barrier()

    # ## END CLUSTERING ALGORITHM ###
        

    # ## START GAUSSIAN SAMPLING LOOP ###
    # g_iteration = 0

    # N_gaussian = 3

    # for i in range(g_iteration, g_iteration + N_gaussian):

    #     gsamples_folder = ''

    #     if me == 0:
    #         print('Start Gaussian Sampling %dth iteration' % i)
    #         sys.stdout.flush()  

    #         gsamples_folder = os.path.join(data_folder,'Gaussian_Samples_%d' % i)

    #         if not os.path.exists(gsamples_folder):
    #             os.makedirs(gsamples_folder, exist_ok=True)

    #     request = comm.Ibarrier()  # Non-blocking barrier

    #     gsamples_folder = comm.bcast(gsamples_folder, root = 0)

    #     t1 = time.perf_counter()

    #     Gaussian_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=0.49*max_time,
    #                                write_dir=write_dir, sample_folder=gsamples_folder,
    #                                gmm_folder=os.path.join(data_folder,'GMM_%d' % i))


    #     t2 = time.perf_counter()

    #     # Wait for the barrier to complete
    #     request.Wait()

    #     if me == 0:
    #         print('End Gaussian Sampling %dth iteration it took %.2f' % (i, t2- t1))
    #         sys.stdout.flush()  

    #         t1 = time.perf_counter()

    #         GMM.main(os.path.join(gsamples_folder,'Core_*'), data_folder, i + 1)

    #         t2 = time.perf_counter()

    #         print('\n Clustering took %.2f s ' % (t2 - t1))

    #         sys.stdout.flush()  
    #     comm.Barrier()
    
    ### END GAUSSIAN SAMPLING LOOP ###

    # ### OPTIMIZE FOR HE-HE POTENTIAL BY USING THE FINAL CLUSTER OF THE W-HE GMM AS A STARTING POINT ###
            
    # comm.Barrier()
    # g_iteration = 3

    N_gaussian = 4

    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] =   bool(n_knots[2])
    bool_fit['H-He'] = False
    bool_fit['He-He'] = True
    
    # Edit a new Covariance Matrix for the He-He potential
    # if me == 0:
    #     gsamples_folder = os.path.join(data_folder,'Gaussian_Samples_%d' % (g_iteration - 1))
    #     GMM.main(os.path.join(gsamples_folder,'Core_*'), data_folder, g_iteration)

    #     for cov_file in glob.glob('%s/GMM_%d/Cov*' % (data_folder, g_iteration)):
    #         cov_0 = np.loadtxt(cov_file)
    #         cov_1 = np.diag([4, 8, 32, 4, 8, 32])

    #         cov = np.block([[cov_0, np.zeros((cov_0.shape[0], cov_1.shape[0]))], 
    #                        [np.zeros((cov_1.shape[0], cov_0.shape[0])), cov_1]])

    #         cov_name = os.path.basename(cov_file) 
    #         np.savetxt('%s/GMM_%d/%s' % (data_folder, g_iteration, cov_name), cov)

    #     for mean_file in glob.glob('%s/GMM_%d/Mean*' % (data_folder, g_iteration)):
    #         mean_0 = np.loadtxt(mean_file).reshape(-1, 1)
    #         mean_1 = np.zeros((len(cov_1),1))

    #         mean = np.vstack([mean_0, mean_1])

    #         mean_name = os.path.basename(mean_file) 
    #         np.savetxt('%s/GMM_%d/%s' % (data_folder, g_iteration, mean_name), mean)

    # comm.Barrier()

    ### BEGIN GAUSSIAN SAMPLING FOR HE-HE POTENTIAL ###
    # if me == 0:
    #     GMM.main(os.path.join(data_folder,'Gaussian_Samples_3', 'Core_*'), data_folder, 4)
    
    # comm.Barrier()

    # for i in range(6, 7):

    #     gsamples_folder = ''

    #     if me == 0:
    #         print('Start Gaussian Sampling %dth iteration' % i)
    #         sys.stdout.flush()  

    #         gsamples_folder = os.path.join(data_folder, 'Gaussian_Samples_%d' % i)

    #         if not os.path.exists(gsamples_folder):
    #             os.mkdir(gsamples_folder)

    #     request = comm.Ibarrier()  # Non-blocking barrier

    #     gsamples_folder = comm.bcast(gsamples_folder, root = 0)

    #     t1 = time.perf_counter()

    #     Gaussian_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=0.99*max_time,
    #                                write_dir=write_dir, sample_folder=gsamples_folder,
    #                                gmm_folder=os.path.join(data_folder,'GMM_%d' % i))        

    #     t2 = time.perf_counter()

    #     # Wait for the barrier to complete
    #     request.Wait()

    #     if me == 0:
    #         print('End Gaussian Sampling %dth iteration it took %.2f' % (i, t2- t1))
    #         sys.stdout.flush()  


    #         t1 = time.perf_counter()

    #         GMM.main(os.path.join(gsamples_folder, 'Core_*'), data_folder,i + 1)

    #         t2 = time.perf_counter()

    #         print('\n Clustering took %.2f s ' % (t2 - t1))
    #         sys.stdout.flush() 

    #     comm.Barrier()
    # ### END GAUSSIAN SAMPLING FOR HE-HE POTENTIAL ###
    
    g_iteration = 7

    ### OPTIMIZE FOR H-HE POTENTIAL BY USING THE FINAL CLUSTER OF THE W-HE GMM AS A STARTING POINT ###
            
    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] =   bool(n_knots[2])
    bool_fit['H-He'] = True
    bool_fit['He-He'] = True
    
    # Edit a new Covariance Matrix for the H-He potential
    # if me == 0:
    #     for cov_file in glob.glob('%s/GMM_%d/Cov*' % (data_folder,g_iteration)):
    #         cov_0 = np.loadtxt(cov_file)
    #         cov_1 = np.diag([4, 8, 32, 4, 8, 32])

    #         cov = np.block([[cov_0, np.zeros((cov_0.shape[0], cov_1.shape[0]))], 
    #                        [np.zeros((cov_1.shape[0], cov_0.shape[0])), cov_1]])

    #         cov_name = os.path.basename(cov_file) 
    #         np.savetxt('%s/GMM_%d/%s' % (data_folder,g_iteration, cov_name), cov)

    #     for mean_file in glob.glob('%s/GMM_%d/Mean*' % (data_folder,g_iteration)):
    #         mean_0 = np.loadtxt(mean_file).reshape(-1, 1)
    #         mean_1 = np.zeros((len(cov_1),1))

    #         mean = np.vstack([mean_0, mean_1])

    #         mean_name = os.path.basename(mean_file) 
    #         np.savetxt('%s/GMM_%d/%s' % (data_folder, g_iteration, mean_name), mean)

    # comm.Barrier()

    ### BEGIN GAUSSIAN SAMPLING FOR H-HE POTENTIAL ###

    for i in range(8, g_iteration + N_gaussian):

        gsamples_folder = ''

        if me == 0:
            print('Start Gaussian Sampling %dth iteration' % i)
            sys.stdout.flush()  

            gsamples_folder = os.path.join(data_folder, 'Gaussian_Samples_%d' % i)

            if not os.path.exists(gsamples_folder):
                os.mkdir(gsamples_folder)

        request = comm.Ibarrier()  # Non-blocking barrier

        gsamples_folder = comm.bcast(gsamples_folder, root = 0)

        t1 = time.perf_counter()

        Gaussian_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=1.98*max_time,
                                   write_dir=write_dir, sample_folder=gsamples_folder,
                                   gmm_folder=os.path.join(data_folder,'GMM_%d' % i))

        t2 = time.perf_counter()

        # Wait for the barrier to complete
        request.Wait()

        if me == 0:
            print('End Gaussian Sampling %dth iteration it took %.2f' % (i, t2- t1))
            sys.stdout.flush()  

            t1 = time.perf_counter()

            GMM.main( os.path.join(gsamples_folder,'Core_*'), data_folder, i + 1)

            t2 = time.perf_counter()

            print('\n Clustering took %.2f s ' % (t2 - t1))
            sys.stdout.flush() 
        comm.Barrier()

    ### END GAUSSIAN SAMPLING FOR H-HE POTENTIAL ###

    g_iteration += N_gaussian

    if me == 0:
        print('\n Gaussian Sampling took %.2f s \n Start Simplex' % (t2 - t1))
        sys.stdout.flush()  

        folders = glob.glob(os.path.join(data_folder, 'Gaussian_Samples_%d/Core_*' % (g_iteration - 1)))

        lst_samples = []
        lst_loss = []
        for folder in folders:
            lst_loss.append(np.loadtxt(os.path.join(folder, 'Filtered_Loss.txt')))
            lst_samples.append(np.loadtxt(os.path.join(folder, 'Filtered_Samples.txt')))

        loss = np.hstack(lst_loss).reshape(-1, 1)
        samples = np.vstack(lst_samples)

        N_simplex = 10

        for proc in range(nprocs):
            simplex_folder = os.path.join(data_folder, 'Simplex/Core_%d' % proc)
            if not os.path.exists(simplex_folder):
                os.makedirs(simplex_folder, exist_ok=True)

        if nprocs >= len(loss):

            for i in range(len(loss)):
                simplex_folder = os.path.join(data_folder, 'Simplex/Core_%d' % i)
                np.savetxt('%s/Simplex_Init.txt' % simplex_folder, samples[i])

            for i in range(len(loss), nprocs):
                simplex_folder = os.path.join(data_folder, 'Simplex/Core_%d' % i)
                with open('%s/Simplex_Init.txt' % simplex_folder, 'w') as file:
                    file.write('')

        elif len(loss) > nprocs and N_simplex*nprocs > len(loss):
            part = len(loss) // nprocs
            idx = 0

            for proc in range(nprocs - 1):
                simplex_folder = os.path.join(data_folder, 'Simplex/Core_%d' % proc)
                np.savetxt('%s/Simplex_Init.txt' % folders[i], samples[idx: idx + part])
                idx += part

            simplex_folder = '%s/Simplex/Core_%d' % (data_folder, nprocs-1)
            np.savetxt('%s/Simplex_Init.txt' % folders[i], samples[idx:])

        else:
            part = N_simplex
            sort_idx = np.argsort(loss).flatten()
            loss = loss[sort_idx]
            samples = samples[sort_idx]

            idx = 0
            for proc in range(nprocs):
                simplex_folder = os.path.join(data_folder, 'Simplex/Core_%d' % proc)
                np.savetxt('%s/Simplex_Init.txt' % simplex_folder, samples[idx: idx + part])
                idx += part

    comm.Barrier()

    simplex_folder = os.path.join(data_folder, 'Simplex/Core_%d' % me)
    Simplex.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, simplex_folder=simplex_folder, write_dir=write_dir)

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

    if me == 0:
        pot_folder = os.path.join(param_dict['write_dir'], 'Potentials')
        
        if not os.path.exists(pot_folder):
            os.makedirs(pot_folder, exist_ok=True)
    
    comm.Barrier()

    main(param_dict['machine'], float(param_dict['max_time']), param_dict['write_dir'],param_dict['save_dir'])

