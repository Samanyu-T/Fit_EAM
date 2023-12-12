import numpy as np
import sys
import json 
from Handle_Files import write_pot, read_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL
from Handle_Dictionaries import data_dict, binding_testing
from Class_Fitting_Potential import Fitting_Potential
import os
def worker_function(proc):

    with open('refs_formations.json', 'r') as ref_file:
        ref_json = json.load(ref_file)

    with open('my_formations.json', 'r') as my_file:
        my_json = json.load(my_file)

    N_Vac = 3
    N_H = 10
    N_He = 10

    # Form a Dictionary containing the formation energies and relaxation volumes for a set of defects
    ref_formations = data_dict(ref_json, my_json, N_Vac, N_H, N_He)

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']

    ref_binding = binding_testing(ref_formations)

    folders = [folder for folder in os.listdir('../Selected_Potentials') if os.path.isdir(os.path.join('../Selected_Potentials', folder))]

    for folder in folders:

        files = os.listdir(folder)

        potfile = [file for file in files if file.endswith('.alloy')]

        potloc = os.path.join(*['../Selected_Potentials', folder, potfile[0]])
                              
        test_formations = sim_defect_set(potloc, ref_formations)
        
        test_binding = binding_testing(test_formations)

        loss = (test_formations['V0H0He1']['val'] - ref_formations['V0H0He1']['val'])**2

        # loss += (test_formations['V0H0He1']['rvol'] - ref_formations['V0H0He1']['rvol'])**2
        binding_loss = (test_binding - ref_binding)**2
        loss += np.sum(binding_loss)

        rvol_loss_lst = []
        # test_rvol = []

        for key in ref_formations:
            # test_rvol.append(test_formations[key]['rvol'])

            if ref_formations[key]['rvol'] is not None:
                
                rvol_loss = (test_formations[key]['rvol'] - ref_formations[key]['rvol'])**2
                rvol_loss_lst.append(rvol_loss)
                loss += rvol_loss

        rvol_loss_lst = np.array(rvol_loss_lst)

        with open(os.path.join(*['../Selected_Potentials', folder, 'Test_Loss.txt']), 'a') as file:

            file.write('%f ' % loss)
            np.savetxt(file, binding_loss, fmt = '%f', newline= ' ')
            np.savetxt(file, rvol_loss_lst, fmt = '%f', newline= ' ')
            file.write('\n')



if __name__ == '__main__':

    if len(sys.argv) > 1:
        worker_function(int(sys.argv[1]))
    else:
        worker_function(-1)