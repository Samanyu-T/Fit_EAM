# Import necessary packages
from random import sample
from Handle_Files import read_pot
from Spline_Fitting import Fitting_Potential, gaussian_sampling
import json
import os
from Handle_Dictionaries import data_dict
import sys 
from Lmp_PDefect import Point_Defect
import time
import numpy as np
from Simulate_Defect_Set import sim_defect_set

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

def optimize(n_knots, bool_fit, proc, machine, max_time=11, write_dir = '',
             sample_folder='../W-He_102/Gaussian_Samples', gmm_folder = '../W-He_102/GMM'):

    # Init a Perfect Tungsten Crystal as a starting point
    lmp_inst = Point_Defect(size = 7, n_vac=0, potfile='Potentials/WHHe_test.eam.alloy') 
    t_iter = lmp_inst.Perfect_Crystal()

    data_folder = os.path.dirname(sample_folder)

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


    lammps_folder = os.path.join(write_dir,'Lammps_Dump_%d' % proc)

    if not os.path.exists(lammps_folder):
        os.makedirs(lammps_folder, exist_ok=True)

    # Init Optimization Parameter
    t1 = time.perf_counter()
    _ = sim_defect_set('Potentials/WHHe_test.eam.alloy', ref_formations, machine, lammps_folder)
    t2 = time.perf_counter()

    t_iter = (t2 - t1)

    n_params = n_knots[0] + n_knots[1] + 3*n_knots[2]

    T_max = 3600*max_time

    N_samples = int(T_max//t_iter)

    if proc == 0:
        print('The Approximate number of Samples: %d \n Number of Dimensions: %d' % (N_samples, n_params))
        sys.stdout.flush()    

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fitting_Potential(pot_lammps=pot, bool_fit=bool_fit,
                                      hyperparams=pot_params, potlines=starting_lines,
                                      n_knots = n_knots, machine = machine, proc_id=proc, write_dir=write_dir)

    files = os.listdir(gmm_folder)

    N = int(len(files)/2)

    select = proc % N
    
    mean = np.loadtxt('%s/Mean_%d.txt' % (gmm_folder, iter, select))
    cov = np.loadtxt('%s/Cov_%d.txt' % (gmm_folder, iter, select))

    gaussian_sampling(ref_formations, fitting_class, max_time=T_max, output_folder=core_folder, mean=mean, cov=cov)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        worker_function(proc = int(sys.argv[1]), machine = sys.argv[2], max_time = int(sys.argv[3]))

    else:
        worker_function(-1, '', 0.1)