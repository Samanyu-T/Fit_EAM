import numpy as np
import os
import json
from Spline_Fitting import Fitting_Potential, optim_loss, genetic_algorithm
from Handle_Files import read_pot, write_pot
import shutil

# Directory path
directory = '../'

# List all directories in the given path
all_folders = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

# Exclude the folder named 'exclude_folder'
exclude_folders = [os.path.join(directory, 'git_folder'), os.path.join(directory, 'Selected_Potentials')]
filtered_folders = [folder for folder in all_folders if folder not in exclude_folders]

if not os.path.exists('../Selected_Potentials'):
    os.mkdir('../Selected_Potentials')

n_chosen = 0

for param_folder in filtered_folders:

    pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']

    n_knots = [int(param_folder[-3]), int(param_folder[-2]), int(param_folder[-1])]

    bool_fit = {}

    bool_fit['He_F(rho)'] = bool(n_knots[0])
    bool_fit['He_rho(r)'] = bool(n_knots[1])
    bool_fit['W-He'] = bool(n_knots[2])
    bool_fit['H-He'] = False
    bool_fit['He-He'] = False

    core_folders = [folder for folder in os.listdir(param_folder) if os.path.isdir(os.path.join(param_folder, folder))]

    for core in core_folders:

        fitting_class = Fitting_Potential(pot, bool_fit, pot_params, starting_lines, n_knots, 0)

        if os.path.exists(os.path.join(*[param_folder, core, 'Simplex', 'Final_Optima.json'])):

            filepath = os.path.join(*[param_folder, core, 'Simplex', 'Final_Optima.json'])
            
            with open(filepath, 'r') as data_file:
                data = json.load(data_file)

            optima = np.array(data['Optima'])
            loss = np.array(data['Loss'])

            chosen_idx = np.where(loss < 0.1)[0].astype(int)
            
            for idx in chosen_idx:
                
                savepath = os.path.join('../Selected_Potentials', 'Potential_%d' % n_chosen)

                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                
                shutil.copy(os.path.join(*[param_folder, core, 'Simplex', 'x_init_%d' % idx, 'loss.txt']), os.path.join(savepath, 'loss.txt'))
                shutil.copy(os.path.join(*[param_folder, core, 'Simplex', 'x_init_%d' % idx, 'sample.txt']), os.path.join(savepath, 'sample.txt'))

                fitting_class.sample_to_file(optima[idx])

                write_pot(fitting_class.pot_lammps, fitting_class.potlines, 'optim%d%d%d.eam.alloy' % (n_knots[0], n_knots[1], n_knots[2]) )


