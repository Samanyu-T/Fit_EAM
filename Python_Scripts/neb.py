import numpy as np
import sys
import os

def p_fit(x, y):

    N = len(x) + 2

    A1 = np.array([ [x[j]**i for i in range(N)] for j in range(len(x))])

    A2 = np.array([ [i*x[0]**np.clip(i-1, a_min = 0, a_max=None) for i in range(N)],
                    [i*x[-1]**np.clip(i-1, a_min = 0, a_max=None) for i in range(N)] ] )

    A = np.vstack([A1, A2])
    
    y = y.reshape(len(y), 1)

    b = np.vstack([y, np.zeros([2, 1])])
    
    return np.linalg.solve(A, b)


def p_val(a, x):
    return np.sum(np.array([[a[i]*x[j]**i for i in range(len(a))] for j in range(len(x))]), axis = 1)

def main(filename, proc):

    name = filename.split('/')[-1].split('.')[0]

    with open('log.lammps', 'r') as file:
        log = file.readlines()

    n = proc

    val = log[-1].split()[-2*n:]

    data = np.array([float(x) for x in val]).reshape(n, 2)

    if name[0] == 's':
        for i in range(n):
            read = np.loadtxt('Lammps_Dump/Neb/neb.%i.dump' % i, skiprows=9)

            idx = np.where(read[:,1] == 3)[0]

            data[i,0] = read[idx, -1][0]
    
    np.savetxt(os.path.join('Test_Data',name + '.txt'), data)

if __name__ == '__main__':

    main(sys.argv[1], int(sys.argv[2]))