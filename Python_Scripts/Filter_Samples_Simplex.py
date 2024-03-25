import os
import glob
import numpy as np

data_folder = '../data_102'
g_iteration = 11

folders = glob.glob(os.path.join(data_folder, 'Gaussian_Samples_%d/Core_*' % (g_iteration - 1)))
nprocs = len(folders)

lst_samples = []
lst_loss = []
for folder in folders:
    lst_loss.append(np.loadtxt(os.path.join(folder, 'Filtered_Loss.txt')))
    lst_samples.append(np.loadtxt(os.path.join(folder, 'Filtered_Samples.txt')))

loss = np.hstack(lst_loss).reshape(-1, 1)
samples = np.vstack(lst_samples)
print(lst_samples[0].shape)

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
        np.savetxt('%s/Simplex_Init.txt' % folders[proc], samples[idx: idx + part])
        idx += part

    simplex_folder = '%s/Simplex/Core_%d' % (data_folder, nprocs-1)
    np.savetxt('%s/Simplex_Init.txt' % folders[proc], samples[idx:])

else:
    part = N_simplex
    sort_idx = np.argsort(loss).flatten()
    loss = loss[sort_idx]
    samples = samples[sort_idx]

    idx = 0
    for proc in range(nprocs):
        simplex_folder = os.path.join(data_folder, 'Simplex/Core_%d' % proc)
        np.savetxt('%s/Simplex_Init.txt' % simplex_folder, samples[idx: idx + part])
        # print(samples[idx: idx + part])
        idx += part
