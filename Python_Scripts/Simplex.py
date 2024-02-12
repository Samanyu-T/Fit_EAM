# Import necessary packages
from random import sample
from Handle_Files import read_pot
from Spline_Fitting import Fitting_Potential, loss_func
import json
import os
from Handle_Dictionaries import data_dict
import sys 
from Lmp_PDefect import Point_Defect
import psutil
import numpy as np
from scipy.optimize import minimize

# Main Function, which takes in each core separetly
def worker_function(proc, machine):
    
    n_knots_lst = [[1,0,2]]

    for n_knots in n_knots_lst:

        bool_fit = {}

        bool_fit['He_F(rho)'] = bool(n_knots[0])
        bool_fit['He_rho(r)'] = bool(n_knots[1])
        bool_fit['W-He'] =   bool(n_knots[2])
        bool_fit['H-He'] = False
        bool_fit['He-He'] = False

        optimize(n_knots, bool_fit, proc, machine)

def optimize(n_knots, bool_fit, proc, machine):

    # Init a Perfect Tungsten Crystal as a starting point
    lmp_inst = Point_Defect(size = 7, n_vac=0, potfile='Potentials/WHHe_test.eam.alloy') 
    t_iter = lmp_inst.Perfect_Crystal()

    # Init Output locations
    param_folder = '../W-He_%d%d%d' % (n_knots[0], n_knots[1], n_knots[2])

    # param_folder = '../He-He_%d' % n_knots[2]
    
    if not os.path.exists(param_folder):
        os.mkdir(param_folder)

    sample_folder = '%s/Simplex' % param_folder

    if not os.path.exists(sample_folder):
        os.mkdir(sample_folder)

    core_folder = '%s/Core_%d' % (sample_folder, proc)

    if not os.path.exists(core_folder):
        os.mkdir(core_folder)


    with open('refs_formations.json', 'r') as ref_file:
        ref_json = json.load(ref_file)

    with open('my_formations.json', 'r') as my_file:
        my_json = json.load(my_file)

    N_Vac = 2
    N_H = 0
    N_He = 1

    if bool_fit['He-He']:
        N_He = 4

    if bool_fit['H-He']:
        N_H = 4


    # Form a Dictionary containing the formation energies and relaxation volumes for a set of defects
    ref_formations = data_dict(ref_json, my_json, N_Vac, N_H, N_He)

    ref_formations['V0H0He1_oct'] = {}
    ref_formations['V0H0He1_oct']['val'] = 6.38
    ref_formations['V0H0He1_oct']['rvol'] = 0.37
    ref_formations['V0H0He1_oct']['pos'] = [[], [], [[3.5, 3.5, 3]]]

    
    
    ref_formations['V0H0He1_inter'] = {}
    ref_formations['V0H0He1_inter']['val'] = None
    ref_formations['V0H0He1_inter']['rvol'] = None
    ref_formations['V0H0He1_inter']['pos'] = [[], [], [[3.375, 3.5, 3]]]


    ref_formations['V0H0He1_inter2'] = {}
    ref_formations['V0H0He1_inter2']['val'] = None
    ref_formations['V0H0He1_inter2']['rvol'] = None
    ref_formations['V0H0He1_inter2']['pos'] = [[], [], [[3.375, 3.375, 3]]]


    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']

    # Call the main fitting class
    fitting_class = Fitting_Potential(pot_lammps=pot, bool_fit=bool_fit, hyperparams=pot_params, potlines=starting_lines, n_knots = n_knots, machine = machine, proc_id=proc)

    # Store the final optimization results
    final_optima = {}
    final_optima['Optima'] = []
    final_optima['Loss'] = []

    datafile = os.path.join(param_folder, 'Gaussian_Samples' , 'Core_%d' % proc ,  'Simplex_Init.txt')

    if os.path.getsize(datafile) > 0:

        x_init_arr = np.loadtxt(datafile)
        
        if x_init_arr.ndim == 1:
            x_init_arr = x_init_arr.reshape(1, -1)

        for simplex_iteration, x_init in enumerate(x_init_arr):

            simplex_iteration_folder = os.path.join(core_folder, 'x_init_%d' % simplex_iteration)

            if not os.path.exists(simplex_iteration_folder):
                os.mkdir(simplex_iteration_folder)

            # Store each sample and corresponding loss in files
            with open('%s/samples.txt' % simplex_iteration_folder, 'w') as file:
                file.write('')

            with open('%s/loss.txt' % simplex_iteration_folder, 'w') as file:
                file.write('')

            maxiter = 1000
            x_star = minimize(loss_func, args=(fitting_class, ref_formations, simplex_iteration_folder), x0=x_init, method = 'COBYLA', options={'maxiter': maxiter, 'tol':1e-6})

            # Write final optima to the output file
            final_optima['Optima'].append(x_star.x.tolist())
            final_optima['Loss'].append(float(x_star.fun))

            # Store all the final optima in a file
            with open('%s/Final_Optima.json' % core_folder, 'w') as file:
                json.dump(final_optima, file, indent=2)




if __name__ == '__main__':

    if len(sys.argv) > 1:
        worker_function(int(sys.argv[1]), sys.argv[2])
        
    else:
        worker_function(-1, '')