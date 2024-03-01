from sklearn.mixture import GaussianMixture
import numpy as np 
import psutil
import os
import glob
import sys
def main(file_pattern, data_folder, iter):

    # Get the process ID
    process_id = os.getpid()

    # Get the process object
    process = psutil.Process(process_id)

    # Get the number of CPU cores used by the process
    cpu_cores = psutil.cpu_count(logical=False)  # logical=False gives you physical cores

    print(f"The process is using {cpu_cores} core(s).")
    sys.stdout.flush()  


    loss_lst = []  

    for file in glob.glob(os.path.join(file_pattern, 'Filtered_Loss.txt')):
        if os.path.getsize(file) > 0:
            loss_lst.append(np.loadtxt(file))
            
    loss = np.hstack([x for x in loss_lst])

    sample_lst = []  

    for file in glob.glob(os.path.join(file_pattern, 'Filtered_Samples.txt')):
        if os.path.getsize(file) > 0:
            sample_lst.append(np.loadtxt(file))
            
    samples = np.vstack([x for x in sample_lst])

    sort_idx = np.argsort(loss)
    loss = loss[sort_idx]
    samples = samples[sort_idx]

    thresh_idx = np.where(loss < 0.25*loss.mean())[0]

    n = np.clip(10000, a_min = 0, a_max=len(thresh_idx)).astype(int)

    data = samples[thresh_idx[:n]]

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

    gmm_folder = '%s/GMM_%d' % (data_folder, iter)
    np.savetxt(os.path.join(gmm_folder, 'Filtered_Loss.txt'),loss[thresh_idx])

    if not os.path.exists(gmm_folder):
        os.mkdir(gmm_folder)

    for i in range(gmm.covariances_.shape[0]):
        np.savetxt(os.path.join(gmm_folder, 'Cov_%d.txt' % i ),gmm.covariances_[i])
        np.savetxt(os.path.join(gmm_folder, 'Mean_%d.txt' % i),gmm.means_[i])

if __name__ == '__main__':
    main('../ddata_102/Random_Samples/Core_*', '../data_102', 0)