
import numpy as np 
import os
import sys

def main(filepath, n_proc):

    path_split = filepath.split('/')
    neb_name = path_split[-2]
    with open('log.lammps', 'r') as file:
        log = file.readlines()

    val = log[-1].split()[-2*n_proc:]

    data = np.array([float(x) for x in val]).reshape(n_proc, 2)

    if not os.path.exists('../Test_Data/Bulk/%s' % neb_name):
        os.makedirs('../Test_Data/Bulk/%s' % neb_name, exist_ok=True)

    np.savetxt(os.path.join('../Test_Data/Bulk' , neb_name, 'neb.txt'), data)

if __name__ == '__main__':

    main(sys.argv[1], int(sys.argv[2]))