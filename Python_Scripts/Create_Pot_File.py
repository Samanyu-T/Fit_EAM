
import numpy as np
import sys
import json 
from Handle_Files import write_pot, read_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL
from Handle_Dictionaries import data_dict, binding_testing
from Class_Fitting_Potential import Fitting_Potential


sample = np.array([4.674558, -0.674741, -0.132412, 0.014577, -0.357314, 0.985033, 0.423835, -0.025393, -0.265163, 1.895311])


# Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']


# Call the main fitting class
fitting_class = Fitting_Potential(pot, pot_params, starting_lines)

potloc = 'Potentials/trial_new.eam.alloy'

fitting_class.sample_to_file(sample)

write_pot(fitting_class.pot_lammps, fitting_class.potlines, potloc)

for key in pot:
    pot[key] = pot[key].tolist()

with open('trial.json', 'w') as file:
    json.dump(pot, file)