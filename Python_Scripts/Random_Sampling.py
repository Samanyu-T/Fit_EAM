# Import necessary packages
from random import sample
from Handle_Files import read_pot
from Spline_Fitting import Fitting_Potential, random_sampling
import json
import os
from Handle_Dictionaries import data_dict
import sys 
from Lmp_PDefect import Point_Defect
import psutil

# Main Function, which takes in each core separetly
def worker_function(proc, machine, max_time):
    
    n_knots_lst = [[1,0,2]]

    for n_knots in n_knots_lst:

        bool_fit = {}

        bool_fit['He_F(rho)'] = bool(n_knots[0])
        bool_fit['He_rho(r)'] = bool(n_knots[1])
        bool_fit['W-He'] =   bool(n_knots[2])
        bool_fit['H-He'] = False
        bool_fit['He-He'] = False

        optimize(n_knots, bool_fit, proc, machine, max_time)

def optimize(n_knots, bool_fit, proc, machine, max_time=11):

    # Init a Perfect Tungsten Crystal as a starting point
    lmp_inst = Point_Defect(size = 7, n_vac=0, potfile='Potentials/WHHe_test.eam.alloy') 
    t_iter = lmp_inst.Perfect_Crystal()

    # Init Output locations
    param_folder = '../W-He_%d%d%d' % (n_knots[0], n_knots[1], n_knots[2])

    # param_folder = '../He-He_%d' % n_knots[2]
    
    if not os.path.exists(param_folder):
        os.mkdir(param_folder)

    sample_folder = '%s/Random_Samples' % param_folder

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

    
    # Init Optimization Parameter
    t_iter *= len(ref_formations)
    n_params = n_knots[0] + n_knots[1] + 3*n_knots[2]

    T_max = 3600*max_time

    N_samples = int(T_max//t_iter)

    print('The number of divisions made on each dim: %.2f' % N_samples**(1/n_params))

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']

    # Call the main fitting class
    fitting_class = Fitting_Potential(pot_lammps=pot, bool_fit=bool_fit, hyperparams=pot_params, potlines=starting_lines, n_knots = n_knots, machine = machine, proc_id=proc)

    random_sampling(ref_formations, fitting_class, N_samples=N_samples, output_folder=core_folder)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        worker_function(proc = int(sys.argv[1]), machine = sys.argv[2], max_time = int(sys.argv[3]))

    else:
        worker_function(-1, '', 0.1)