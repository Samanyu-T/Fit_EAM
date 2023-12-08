
import numpy as np
import sys
import json 
from Handle_Files import write_pot, read_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL
from Handle_Dictionaries import data_dict, binding_testing
from Class_Fitting_Potential import Fitting_Potential


sample = np.array([-0.884224, -0.054990, -0.069627])

bool_fit = {}

bool_fit['He_F(rho)'] = False
bool_fit['He_rho(r)'] = False
bool_fit['W-He'] = True
bool_fit['H-He'] = False
bool_fit['He-He'] = False

# Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']

# Call the main fitting class
fitting_class = Fitting_Potential(pot_lammps=pot, bool_fit=bool_fit,hyperparams=pot_params,potlines=starting_lines, proc_id=0)

potloc = 'Potentials/trial_new.eam.alloy'

fitting_class.sample_to_file(sample)

write_pot(fitting_class.pot_lammps, fitting_class.potlines, potloc)

for key in pot:
    pot[key] = pot[key].tolist()

with open('trial.json', 'w') as file:
    json.dump(pot, file)