# Import necessary packages
from Handle_Files import read_pot
from Spline_Fitting import Fitting_Potential, optim_loss, genetic_algorithm
import json
from scipy.optimize import minimize
import os
import numpy as np
from Handle_Dictionaries import data_dict
import sys 

# Main Function, which takes in each core separetly
def worker_function(proc):
    
    n_knots_lst = [[0,0,1] , [0,0,2], [1,0,1], [1,1,1], [1,0,2], [1,1,2]]

    for n_knots in n_knots_lst:

        bool_fit = {}

        bool_fit['He_F(rho)'] = bool(n_knots[0])
        bool_fit['He_rho(r)'] = bool(n_knots[1])
        bool_fit['W-He'] = bool(n_knots[2])
        bool_fit['H-He'] = False
        bool_fit['He-He'] = False

        optimize(n_knots, bool_fit, proc)

def optimize(n_knots, bool_fit, proc):

    n_params = n_knots[0] + n_knots[1] + 3*n_knots[2]

    N_genetic_samples = int(1e2*n_params**2)
    N_genetic_steps = 20
    genetic_exploration = 2
    genetic_decay = 1.25

    param_folder = '../W-He_%d%d%d' % (n_knots[0], n_knots[1], n_knots[2])

    if not os.path.exists(param_folder):
        os.mkdir(param_folder)

    core_folder = '%s/Core_%d' % (param_folder, proc)

    if not os.path.exists(core_folder):
        os.mkdir(core_folder)

    optim_folder = '%s/Simplex' % core_folder
    genetic_folder = '%s/Genetic' % core_folder

    if not os.path.exists(optim_folder):
        os.mkdir(optim_folder)

    if not os.path.exists(genetic_folder):
        os.mkdir(genetic_folder)

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

    # Call the main fitting class

    fitting_class = Fitting_Potential(pot_lammps=pot, bool_fit=bool_fit, hyperparams=pot_params, potlines=starting_lines, n_knots = n_knots, proc_id=proc)

    genetic_algorithm(ref_formations, fitting_class, N_samples=N_genetic_samples, N_steps=N_genetic_steps, mutate_coef=genetic_exploration, mutate_decay = genetic_decay, output_folder=genetic_folder)

    with os.scandir(genetic_folder) as entries:
        genetic_folders = [entry.name for entry in entries if entry.is_dir()]

    with open(os.path.join(core_folder,'Filtered_Loss.txt'), 'w') as file:
        file.write('')

    with open(os.path.join(core_folder,'Filtered_Samples.txt'), 'w') as file:
        file.write('')

    for genetic_iteration_folder in genetic_folders:

        loss_data = np.genfromtxt(os.path.join(genetic_folder, genetic_iteration_folder, 'loss.txt'), delimiter=' ')
        samples = np.genfromtxt(os.path.join(genetic_folder, genetic_iteration_folder, 'population.txt'), delimiter=' ')

        nan_columns = np.isnan(loss_data[0,:])
        
        loss_data = loss_data[:, ~nan_columns]

        condition = np.logical_and.reduce([loss_data[:,0] < 10, np.abs(loss_data[:, -6] - loss_data[:, -1]) < 0.1, np.abs(loss_data[:, -6] - 0.36) < 0.2])

        filtered_idx = np.where(condition)[0]

        with open(os.path.join(core_folder,'Filtered_Loss.txt'), 'a') as file:
            for idx in filtered_idx:
                np.savetxt(file, loss_data[idx,:], fmt='%f', newline=' ')
                file.write('\n')
        
        with open(os.path.join(core_folder,'Filtered_Samples.txt'), 'a') as file:
            for idx in filtered_idx:
                np.savetxt(file, samples[idx,:], fmt='%f', newline=' ')
                file.write('\n')

    # Store the final optimization results
    final_optima = {}
    final_optima['Optima'] = []
    final_optima['Loss'] = []

    x_init_arr = np.loadtxt(os.path.join(core_folder,'Filtered_Samples.txt'))

    for simplex_iteration, x_init in enumerate(x_init_arr):

        simplex_iteration_folder = os.path.join(optim_folder, 'x_init_%d' % simplex_iteration)

        if not os.path.exists(simplex_iteration_folder):
            os.mkdir(simplex_iteration_folder)

        # Store each sample and corresponding loss in files
        with open('%s/samples.txt' % simplex_iteration_folder, 'w') as file:
            file.write('')

        with open('%s/loss.txt' % simplex_iteration_folder, 'w') as file:
            file.write('')

        # Random initialization for the optimization
        # x_init = fitting_class.gen_rand()
        maxiter = 1000
        x_star = minimize(optim_loss, args=(fitting_class, ref_formations, simplex_iteration_folder), x0=x_init, method = 'COBYLA', options={'maxiter': maxiter})

        # Write final optima to the output file
        final_optima['Optima'].append(x_star.x.tolist())
        final_optima['Loss'].append(float(x_star.fun))

        # Store all the final optima in a file
        with open('%s/Final_Optima.json' % optim_folder, 'w') as file:
            json.dump(final_optima, file, indent=2)

    
if __name__ == '__main__':

    if len(sys.argv) > 1:
        worker_function(int(sys.argv[1]))
    else:
        worker_function(-1)