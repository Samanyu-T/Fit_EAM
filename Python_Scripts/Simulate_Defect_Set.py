from Lmp_PDefect import Point_Defect
import json
import numpy as np
import time

def sim_defect_set(potfile, ref_dict):

    lmp_inst = Point_Defect(size = 7, n_vac=0, potfile=potfile) 

    test_dict = {}

    for key in ref_dict:
        
        test_dict[key] = {}

        lmp_inst.n_vac = int(key[1])

        ef, rvol, _ = lmp_inst.Build_Defect(ref_dict[key]['pos'])

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

    lmp_inst = Point_Defect(size = 7, n_vac=0, potfile=potfile) 

    test_formation = np.zeros((10,))
    test_relaxation = np.zeros((10,))

    for i in range(len(ref_formation)):
        lmp_inst.n_vac = vac_arr[i]
        test_formation[i], test_relaxation[i], _ = lmp_inst.Build_Defect(ref_positions[i])

    return test_formation, ref_formation, test_relaxation, ref_relaxation

# ftest, fref, rtest, rref = sim_defect_print('Potentials/test.eam.alloy')

# print(ftest, fref)
# print(rtest, rref)
