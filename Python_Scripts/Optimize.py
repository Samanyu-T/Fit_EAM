# Import necessary packages
from Handle_Files import read_pot
from Class_Fitting_Potential import Fitting_Potential, optim_loss, genetic_algorithm
import json
from scipy.optimize import minimize
import os
import numpy as np
from Handle_Dictionaries import data_dict
import sys 

# Main Function, which takes in each core separetly
def worker_function(proc):

    optim_folder = 'Optimization_Files/Iteration_%d' % proc

    if not os.path.exists(optim_folder):
        os.mkdir(optim_folder)
        os.mkdir(optim_folder + '/Sample_Files')
        os.mkdir(optim_folder + '/Loss_Files')

    with open('refs_formations.json', 'r') as ref_file:
        ref_json = json.load(ref_file)

    with open('my_formations.json', 'r') as my_file:
        my_json = json.load(my_file)

    N_Vac = 2
    N_H = 0
    N_He = 1

    # Form a Dictionary containing the formation energies and relaxation volumes for a set of defects
    ref_formations = data_dict(ref_json, my_json, N_Vac, N_H, N_He)
    
    ref_formations['V0H0He1_oct'] = {}
    ref_formations['V0H0He1_oct']['val'] = 6.38
    ref_formations['V0H0He1_oct']['rvol'] = 0.37
    ref_formations['V0H0He1_oct']['pos'] = [[], [], [[3.5, 3.5, 3]]]

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']

    bool_fit = {}

    bool_fit['He_F(rho)'] = False
    bool_fit['He_rho(r)'] = False
    bool_fit['W-He'] = True
    bool_fit['H-He'] = False
    bool_fit['He-He'] = False

    # Call the main fitting class
    fitting_class = Fitting_Potential(pot_lammps=pot, bool_fit=bool_fit,hyperparams=pot_params,potlines=starting_lines, proc_id=proc)

    genetic_algorithm(ref_formations, fitting_class, N_samples=1000, N_steps=50, mutate_coef=0.1)

    # # Number of optimization instances
    # N = 100

    # # Store the final optimization results
    # final_optima = {}
    # final_optima['Optima'] = np.zeros((N, fitting_class.len_sample))
    # final_optima['Loss'] = np.zeros((N,))

    
    # # x_init_arr = np.loadtxt('Explore_Space/filtered_samples.%d.txt' % proc)

    # # for iteration, x_init in enumerate(x_init_arr):
    
    # for iteration in range(N):
    #     # Store each sample and corresponding loss in files
    #     with open('%s/Sample_Files/samples_%d.txt' % (optim_folder, iteration), 'w') as file:
    #         file.write('Start Optimization \n')

    #     with open('%s/Loss_Files/loss_%d.txt' % (optim_folder, iteration), 'w') as file:
    #         file.write('Start Optimization \n')

    #     # Random initialization for the optimization
    #     x_init = fitting_class.gen_rand()
        
    #     x_star = minimize(optim_loss, args=(fitting_class, ref_formations, iteration, optim_folder), x0=x_init, method = 'COBYLA')

    #     # Write final optima to the output file
    #     final_optima['Optima'][iteration] = x_star.x
    #     final_optima['Loss'][iteration] = x_star.fun

    # for key in final_optima:
    #     final_optima[key] = final_optima[key].tolist()
    
    # # Store all the final optima in a file
    # with open('%s/Final_Optima.json' % optim_folder, 'w') as file:
    #     json.dump(final_optima, file, indent=2)

    
if __name__ == '__main__':

    if len(sys.argv) > 1:
        worker_function(int(sys.argv[1]))
    else:
        worker_function(-1)