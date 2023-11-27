from Handle_Files import read_pot, write_pot
import json
import numpy as np

pot, starting_lines, pot_params = read_pot('Potentials/WHHe_test.eam.alloy')

for key in pot:
    pot[key] = pot[key].tolist()

with open('Test_negative_rho.json','r') as file:
    loaded_data = json.load(file)

pot = {}

for key, value in loaded_data.items():
    pot[key] = np.array(value)

write_pot(pot, starting_lines, 'potential-rho.eam.alloy')