from Lmp_PDefect import Point_Defect
import json
import numpy as np
import time
import sys
def sim_defect_set(potfile, ref_dict, machine, lammps_folder='Lammps_Dump', proc_id = 0):

    lmp_inst = Point_Defect(size = 7, n_vac=0, potfile=potfile, machine=machine, lammps_folder=lammps_folder, proc_id=proc_id) 

    test_dict = {}
    for key in ref_dict:
        print(key)
        sys.stdout.flush()    
        n_vac = int(key[1])
        n_h = int(key[3])
        n_he = int(key[6])

        if n_h + n_he <= 1: 
            test_dict[key] = {}

            lmp_inst.n_vac = int(key[1])
            ef, rvol, pos = lmp_inst.Build_Defect(ref_dict[key]['pos'])

            test_dict[key]['val'] = ef
            test_dict[key]['rvol'] = rvol
            test_dict[key]['pos'] = pos

        else:
            atom_to_add = 3

            if n_h > 0:
                atom_to_add = 2

            if n_h == 0:
                init_file = 'V%dH%dHe%d' % (n_vac, n_h, n_he-1)
            else:
                init_file = 'V%dH%dHe%d' % (n_vac, n_h - 1, n_he)

            lmp_inst.n_vac = int(key[1])

            ef, rvol = lmp_inst.Find_Min_Config(init_file, atom_to_add)
            test_dict[key] = {}
            test_dict[key]['val'] = ef
            test_dict[key]['rvol'] = rvol      

    return test_dict

def sim_defect_print(potfile):

    with open('refs_formations.json', 'r') as ref_file:
        ref_json = json.load(ref_file)

    with open('my_formations.json', 'r') as my_file:
        my_json = json.load(my_file)

    ref_formation = np.zeros((10,))
    ref_relaxation = np.zeros((10,))
    ref_positions = []
    vac_arr = []

    idx = 0
    for n_vac in range(2):
        for n_he in range(1,6):
            ref_formation[idx] = ref_json['V%dH0He%d' % (n_vac,n_he)]['dft']['val'][0]
            ref_relaxation[idx] = ref_json['V%dH0He%d' % (n_vac,n_he)]['r_vol_dft']['val'][0]
            ref_positions.append(my_json['V%dH0He%d' % (n_vac,n_he)]['xyz_opt'])
            vac_arr.append(n_vac)
            idx += 1

    lmp_inst = Point_Defect(size = 7, n_vac=0, potfile=potfile, machine='') 

    test_formation = np.zeros((10,))
    test_relaxation = np.zeros((10,))

    for i in range(len(ref_formation)):
        lmp_inst.n_vac = vac_arr[i]
        t1 = time.perf_counter()
        test_formation[i], test_relaxation[i], _ = lmp_inst.Build_Defect(ref_positions[i])
        t2= time.perf_counter()
        print(t2-t1)

    return test_formation, ref_formation, test_relaxation, ref_relaxation

# ftest, fref, rtest, rref = sim_defect_print('Potentials/Selected_Potentials/Potential_3/optim102.eam.alloy')

# print(ftest, fref)
# print(rtest, rref)
