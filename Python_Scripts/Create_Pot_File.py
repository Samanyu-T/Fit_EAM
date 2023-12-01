
import numpy as np
import sys
import json 
from Handle_Files import write_pot, read_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL
from Handle_Dictionaries import data_dict, binding_testing
from Class_Fitting_Potential import Fitting_Potential


sample = np.array([6.569761, -0.285214, -0.080556, -0.147748, -0.140890, 0.178565, 0.990110, 0.783649, -0.204698, 1.011782])


# Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']


# Call the main fitting class
fitting_class = Fitting_Potential(pot, pot_params, starting_lines)

potloc = 'Potentials/trial.eam.alloy'

fitting_class.sample_to_file(sample)

write_pot(fitting_class.pot_lammps, fitting_class.potlines, potloc)