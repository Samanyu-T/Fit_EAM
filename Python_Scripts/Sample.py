# Import necessary packages
from Handle_Files import read_pot
from Class_Fitting_Potential import Fitting_Potential, sample_loss
import json
import os
from Handle_Dictionaries import data_dict
import sys 

# Main Function, which takes in each core separetly
def worker_function(proc):

    sample_folder = 'Explore_Space'
    sample_filepath = sample_folder + '/samples_%d.txt' % proc

    if not os.path.exists(sample_folder):
        os.mkdir(sample_folder)

    with open('refs_formations.json', 'r') as ref_file:
        ref_json = json.load(ref_file)

    with open('my_formations.json', 'r') as my_file:
        my_json = json.load(my_file)

    N_Vac = 1
    N_H = 1
    N_He = 3

    # Form a Dictionary containing the formation energies and relaxation volumes for a set of defects
    ref_dict = data_dict(ref_json, my_json, N_Vac, N_H, N_He)

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']

    # Call the main fitting class
    fitting_class = Fitting_Potential(pot, pot_params, starting_lines, proc)

    # Number of optimization instances
    N = int(1e6)

    for iteration in range(N):

        sample = fitting_class.init_sample(isrand=True)
        loss = sample_loss(sample, fitting_class, ref_dict, sample_filepath)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        worker_function(int(sys.argv[1]))
    else:
        worker_function(-1)