import numpy as np
import os 
import json 
from Handle_Files import write_pot, read_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL
from Handle_Dictionaries import data_dict, find_ref_binding
from Class_Fitting_Potential import Fitting_Potential


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
fitting_class = Fitting_Potential(pot, pot_params, starting_lines, proc=0)

samples = np.loadtxt('test_samples.txt')

for sample in samples:

    potloc = 'Potentials/test_set.eam.alloy'

    write_pot(fitting_class.pot_lammps, fitting_class.potlines, potloc)
