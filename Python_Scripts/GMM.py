from sklearn.mixture import GaussianMixture
import numpy as np 
import psutil
import os
import glob
import sys
def main(file_pattern, iter):
    data_lst = []

    # file_pattern = '../W-He_102/Random_Samples/Core_*/Filtered_Samples.txt'
    # file_pattern = '../W-He_102/Core*/Sample/Filtered_Samples.txt'

    for file in glob.glob(file_pattern):
        if os.path.getsize(file) > 0:
            data_lst.append(np.loadtxt(file))
            
    data = np.vstack([x for x in data_lst])

    # Get the process ID
    process_id = os.getpid()

    # Get the process object
    process = psutil.Process(process_id)

    # Get the number of CPU cores used by the process
    cpu_cores = psutil.cpu_count(logical=False)  # logical=False gives you physical cores

    print(f"The process is using {cpu_cores} core(s).")
    sys.stdout.flush()  

    cmp = 1
    gmm = GaussianMixture(n_components=cmp, covariance_type='full', reg_covar=1e-6)
    gmm.fit(data)
    bic_val = gmm.bic(data)
    bic_val_prev = bic_val

    print(cmp, bic_val)
    sys.stdout.flush()  

    while True:

        cmp += 1
        bic_val_prev = bic_val
        gmm = GaussianMixture(n_components=cmp, covariance_type='full', reg_covar=1e-6)
        gmm.fit(data)
        bic_val = gmm.bic(data)
        print(cmp, bic_val, bic_val_prev)

        if 1.01*bic_val > bic_val_prev:
            break

    print(cmp - 1)
    sys.stdout.flush()
      
    gmm = GaussianMixture(n_components=cmp - 1 , covariance_type='full')
    gmm.fit(data)

    param_folder = '../W-He_102' 
    gmm_folder = '%s/GMM_%d' % (param_folder, iter)

    if not os.path.exists(gmm_folder):
        os.mkdir(gmm_folder)

    for i in range(gmm.covariances_.shape[0]):
        np.savetxt(os.path.join(gmm_folder, 'Cov_%d.txt' % i ),gmm.covariances_[i])
        np.savetxt(os.path.join(gmm_folder, 'Mean_%d.txt' % i),gmm.means_[i])

if __name__ == '__main__':
    main(sys.argv[1], 0)