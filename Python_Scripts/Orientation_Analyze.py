from Lammps_PDefect_Classes import Lammps_Point_Defect
import numpy as np
import os
from mpi4py import MPI
import matplotlib.pyplot as plt
import sys

try:

    comm = MPI.COMM_WORLD

    me = comm.Get_rank()

    mode = 'MPI'

except:

    me = 0

    mode = 'Serial'



def test(orientx, orienty, orientz):
    potfile = 'Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy'
    
    # potfile = 'Potentials/WHHe_test.eam.alloy'

    alattice = 3.144221296574379
    size = 10

    lmp = Lammps_Point_Defect(size = 7, n_vac=0, potfile=potfile, surface = False , depth = 0,
                            orientx=orientx, orienty=orienty, orientz=orientz, conv = 10000)

    R_inv = np.vstack([orientx, orienty, orientz]).T
    R = np.linalg.inv(R_inv)

    tet_arr = lmp.get_tetrahedral_sites()

    tet = R @ tet_arr.T

    tet = (tet + 2) % 1
    
    idx = np.argsort(tet[-1,:])

    print(idx.shape)

    tet = tet[:,idx]

    tet = tet.T
    
    tet = alattice * np.linalg.norm(R_inv, axis = 0) * tet

    climb = [tet[0]]

    for _t in tet:

        if _t[2] == climb[-1][2]:

            if len(climb) > 1:

                if np.linalg.norm(climb[-2] - climb[-1]) > np.linalg.norm(climb[-2] - _t):
                    climb[-1] = _t
        
        else:

            climb.append(_t)
    
    climb = np.array(climb)

    # climb[:,-1] = climb[:,-1] + alattice * np.linalg.norm(R_inv, axis = 0)[-1]

    for c in climb:

        # c[2] += 0.5 * alattice * np.linalg.norm(R_inv, axis = 0)[-1]

        pe, pos = lmp.Build_Defect([[],[],[ c ]], dump_name='Test_%d%d%d_tet' % (orientx[0], orientx[1], orientx[2]))

        print(pe, pos, c)

''' Use for 100 surface '''
orientx = [1, 0, 0]
orienty = [0, 1, 0]
orientz = [0 ,0, 1]
test(orientx, orienty, orientz)

''' Use for 110 surface '''
orientx = [1, 1, 0]
orienty = [0, 0,-1]
orientz = [-1,1, 0]
test(orientx, orienty, orientz)

''' Use for 111 surface '''
orientx = [1, 1, 1]
orienty = [-1,2,-1]
orientz = [-1,0, 1]

test(orientx, orienty, orientz)