from Handle_Files import read_pot, write_pot
from Class_Fitting_Potential import Fitting_Potential, loss_func
import json
from scipy.optimize import minimize
import os
import numpy as np
from Simulate_Defect_Set import sim_defect_set
from Handle_Dictionaries import data_dict, find_ref_binding
import sys 

def main(output_folder = 'Optimization_Files'):
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open('refs_formations.json', 'r') as ref_file:
        ref_json = json.load(ref_file)

    with open('my_formations.json', 'r') as my_file:
        my_json = json.load(my_file)

    N_Vac = 1
    N_H = 1
    N_He = 3

    ref_dict = data_dict(ref_json, my_json, N_Vac, N_H, N_He)

    pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

    for key in pot:
        pot[key] = pot[key].tolist()

    with open('Init.json','r') as file:
        loaded_data = json.load(file)

    pot = {}

    for key, value in loaded_data.items():
        pot[key] = np.array(value)

    write_pot(pot, starting_lines, 'potential.eam.alloy')

    # F_nknots = 3
    # Rho_nknots = 1
    # V_nknots = 4

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']

    fitting_class = Fitting_Potential(pot_params, starting_lines)

    N = 100

    final_optima = {}
    final_optima['Optima'] = np.zeros((N, fitting_class.len_sample))
    final_optima['Loss'] = np.zeros((N,))
    final_optima['Formation Energy'] = np.zeros((N, N_H*N_Vac*N_He))
    final_optima['Relaxation Volume'] = np.zeros((N, N_H*N_Vac*N_He))

    with open('Final_Optima.txt', 'w') as file:
        file.write('Start Optimization \n')

    for iteration in range(N):
        
        with open('%s/Sample_Files/samples_%d.txt' % (output_folder, iteration), 'w') as file:
            file.write('Start Optimization \n')

        with open('%s/Loss_Files/loss_%d.txt' % (output_folder, iteration), 'w') as file:
            file.write('Start Optimization \n')

        x_init = fitting_class.init_sample(isrand=True)
        x_star = minimize(loss_func, args=(fitting_class, pot, ref_dict, iteration, output_folder), x0=x_init, method = 'COBYLA')

        # Write results to the output file
        final_optima['Optima'][iteration] = x_star.x
        final_optima['Loss'][iteration] = x_star.fun

    for key in final_optima:
        final_optima[key] = final_optima[key].tolist()
        
    with open('Final_Optima.json', 'w') as file:
        json.dump(final_optima, file, indent=2)


if __name__ == '__main__':
    main('Optimization_Files_%s' % sys.argv[0])