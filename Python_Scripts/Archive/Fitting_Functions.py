import numpy as np
import os 
import json 
from Handle_Files import write_pot, read_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL

def fit_sample(sample, F_nknots = 3, Rho_nknots = 3, V_nknots = 4):

    ''' Sample ordered as follows: F(He), Rho(He), V(W-He), V(H-He), V(He-He) '''
    ''' Inside of each of the above params: sqrt_coef(if it exists), x_knot, y_knot, dy_knot, d2y_knot'''

    map = {}

    keys  = ['He_F(rho)','He_rho(r)', 'W-He', 'H-He', 'He-He']
    items = ['knots', 'y', 'dy', 'd2y']

    splice_F = [F_nknots - 1, F_nknots - 2, F_nknots - 1, F_nknots - 1]
    splice_Rho = [Rho_nknots - 1, Rho_nknots - 1, Rho_nknots - 1, Rho_nknots - 1]
    splice_V = [V_nknots - 1, V_nknots - 2, V_nknots - 2, V_nknots - 2]

    splices = splice_F + splice_Rho + 3*splice_V

    idx = 1
    iter = 0
    for key in keys:
        map[key] = {}
        for item in items:
            map[key][item] = [i for i in range(idx, idx + splices[iter])]
            idx += splices[iter]
            iter += 1

    coef_dict   = {}
    knot_dict = {}

    coef_dict['sqrt_coef He_F'] = sample[0]

    key = 'He_F(rho)'
    coef_dict[key], knot_dict[key] = fit_F(sample[map[key]['knots']], sample[map[key]['y']], sample[map[key]['dy']], sample[map[key]['d2y']])

    key = 'He_rho(r)'
    coef_dict[key], knot_dict[key] = fit_Rho(sample[map[key]['knots']], sample[map[key]['y']], sample[map[key]['dy']], sample[map[key]['d2y']])

    key = 'W-He'
    coef_dict[key], knot_dict[key] = fit_V(sample[map[key]['knots']], sample[map[key]['y']], sample[map[key]['dy']], sample[map[key]['d2y']])

    key = 'H-He'
    coef_dict[key], knot_dict[key] = fit_V(sample[map[key]['knots']], sample[map[key]['y']], sample[map[key]['dy']], sample[map[key]['d2y']])

    key = 'He-He'
    coef_dict[key], knot_dict[key] = fit_V(sample[map[key]['knots']], sample[map[key]['y']], sample[map[key]['dy']], sample[map[key]['d2y']])

    return coef_dict, knot_dict
    
def update_potfile(pot_dict, pot_params, coef_dict, knot_dict):

    rho = np.linspace(0, pot_params['drho']*(pot_params['Nrho']-1), pot_params['Nrho'])

    pot_dict['He_F(rho)'] = eval_F(rho, coef_dict['sqrt_coef He_F'], coef_dict['He_F(rho)'], knot_dict['He_F(rho)'])

    r = np.linspace(0, pot_params['dr']*(pot_params['Nr']-1), pot_params['Nr'])

    pot_dict['He_rho(r)'] = eval_Rho(r, coef_dict['He_rho(r)'], knot_dict['He_rho(r)'])

    charge = [[74, 2], [1, 2], [2,2]]

    for i, key in enumerate(['W-He', 'H-He', 'He-He']):
        
        pot_dict[key][1:] = r[1:]*eval_V(r[1:], coef_dict[key], knot_dict[key], charge[i][0], charge[i][1])

    return pot_dict

def extend_sample(sample, x, y, dy, d2y):
    for element in [x, y, dy, d2y]:
        for val in element:
            sample.append(val)
    return sample

def init_sample(pot_params, F_nknots = 3, Rho_nknots = 3, V_nknots = 4):

    sample = []

    with open("Init.json", "r") as json_file:
        loaded_data = json.load(json_file)

    pot = {}
    for key, value in loaded_data.items():
        pot[key] = np.array(value)

    Nrho = pot_params['Nrho']
    drho = pot_params['drho']
    Nr   = pot_params['Nr']
    dr   = pot_params['dr']
    rc   = pot_params['rc']

    coef_dict   = {}
    knot_dict = {}

    coef_dict['sqrt_coef He_F'] = 0

    sample.append(coef_dict['sqrt_coef He_F'])

    knots = np.linspace(0, 2*drho*Nrho, F_nknots)
    y = 0.37*knots[1:-1]
    dy = 0.37*np.ones((F_nknots - 1,))
    d2y = np.zeros((F_nknots - 1,))
    
    sample = extend_sample(sample, knots[1:], y, dy, d2y)
    coef_dict['He_F(rho)'], knot_dict['He_F(rho)'] = fit_F(knots[1:], y, dy, d2y)

    knots = np.linspace(0, dr*Nr, Rho_nknots)
    y = np.zeros((Rho_nknots - 1,))
    dy = np.zeros((Rho_nknots - 1,))
    d2y = np.zeros((Rho_nknots - 1,))

    sample = extend_sample(sample, knots[1:], y, dy, d2y)
    coef_dict['He_rho(r)'], knot_dict['He_rho(r)'] = fit_Rho(knots[1:], y, dy, d2y)

    knots = np.linspace(0, dr*Nr, V_nknots)

    knots_idx = (knots/dr).astype(int)

    charge = [[74, 2], [1, 2], [2,2]]

    for i, key in enumerate(['W-He', 'H-He', 'He-He']):
        
        zbl_class = ZBL(charge[i][0], charge[i][1])

        f = pot[key][knots_idx[1:-1]]/knots[1:-1]
        f_backward = pot[key][knots_idx[1:-1] - 1]/(knots[1:-1]-dr)
        f_forward = pot[key][knots_idx[1:-1] + 1]/(knots[1:-1]+dr)
        
        y = f - zbl_class.eval_zbl(knots[1:-1])
        dy = (f-f_backward)/dr - zbl_class.eval_grad(knots[1:-1])
        d2y = (f_forward - 2*f + f_backward)/dr**2 - zbl_class.eval_hess(knots[1:-1])

        coef_dict[key], knot_dict[key] = fit_V(knots[1:], y, dy,d2y, Zi = charge[i][0], Zj = charge[i][1])

        sample = extend_sample(sample, knots[1:], y, dy, d2y)
        
    return np.array(sample)
    
def eval_piecewise_poly(x, coef, knots):

    dof = coef.shape[1]

    n = len(knots)

    lower_bound = knots[np.arange(n-1)]
    upper_bound = knots[np.arange(1,n)]

    masks = np.array( [( (lower_bound[i] <= x) & (x < upper_bound[i]) ).astype(int) for i in range(n-1)])

    Psi = np.array([x**i for i in range(dof)])
    
    return np.sum( np.array([ masks[i]*(coef[i]@Psi) for i in range(n-1)]), axis = 0)

def eval_V(r, coef, knots, Zi = 74, Zj = 2):

    poly = eval_piecewise_poly(r, coef, knots)
    zbl_class = ZBL(Zi, Zj)
    mask = (r < knots[-2]).astype(int)
    zbl = mask * zbl_class.eval_zbl(r)

    return poly + zbl

def eval_F(r, sqrt_coef, poly_coef, knots):

    poly = eval_piecewise_poly(r, poly_coef, knots)

    sqrt = sqrt_coef*np.sqrt(r)

    return poly + sqrt

def eval_Rho(r, coef, knots):

    Rho =  eval_piecewise_poly(r, coef, knots)

    return Rho

def fit_piecewise_poly(x_knot, y_knot, dy_knot, d2y_knot, dof = 6 ):
    
    n = len(x_knot)
    
    pairs = np.column_stack( [np.arange(n-1), np.arange(1,n)] )

    coef  = np.zeros((n-1, dof))

    for j in range(n-1):

        psi_y = np.array([x_knot[pairs[j]]**i for i in range(dof)]).T

        psi_dy = np.array([i*x_knot[pairs[j]]**np.clip(i-1, a_min=0, a_max=None) for i in range(dof)]).T

        psi_d2y = np.array([i*(i-1)*x_knot[pairs[j]]**np.clip(i-2, a_min=0, a_max=None) for i in range(dof)]).T

        Psi = np.vstack([psi_y, psi_dy, psi_d2y])

        Y = np.hstack([y_knot[pairs[j]], dy_knot[pairs[j]], d2y_knot[pairs[j]]])

        coef[j] = np.linalg.inv(Psi.T@Psi) @ Psi.T @ Y
        
    return coef

def fit_poly(x, y, dy, d2y, dof = 6):

    psi_y = np.array([x**i for i in range(dof)]).T

    psi_dy = np.array([i*x**np.clip(i-1, a_min=0, a_max=None) for i in range(dof)]).T

    psi_d2y = np.array([i*(i-1)*x**np.clip(i-2, a_min=0, a_max=None) for i in range(dof)]).T

    Psi = np.vstack([psi_y, psi_dy, psi_d2y])

    Y = np.hstack([y, dy, d2y])

    coef = np.linalg.inv(Psi.T@Psi) @ Psi.T @ Y

    return coef

def fit_V(x, y, dy, d2y, dof = 6, Zi = 74, Zj = 2):

    n = len(y) + 2

    x_knot = np.zeros((n,))
    y_knot = np.zeros((n,))
    dy_knot = np.zeros((n,))
    d2y_knot = np.zeros((n,))
    
    x_knot[1:] = x
    y_knot[1:-1] = y
    dy_knot[1:-1] = dy
    d2y_knot[1:-1] = d2y


    x_with_index = np.column_stack([x_knot, np.arange(n)])

    x_with_index = x_with_index[np.argsort(x_with_index[:,0])]

    x_knot = x_with_index[:,0]

    x_index = x_with_index[:,1].astype(int)

    y_knot = y_knot[x_index]
    dy_knot = dy_knot[x_index]
    d2y_knot = d2y_knot[x_index]

    zbl_class = ZBL(Zi, Zj)
    
    # y_knot[-1] = -zbl_class.eval_zbl(x_knot[-1])
    # dy_knot[-1] = -zbl_class.eval_grad(x_knot[-1])
    # d2y_knot[-1] = -zbl_class.eval_hess(x_knot[-1])

    # coef = fit_piecewise_poly(x_knot, y_knot, dy_knot, d2y_knot)

    coef = np.zeros((n-1, dof))

    coef[:-1] = fit_piecewise_poly(x_knot[:-1], y_knot[:-1], dy_knot[:-1], d2y_knot[:-1])

    y_knot[-2] += zbl_class.eval_zbl(x_knot[-2])
    dy_knot[-2] += zbl_class.eval_grad(x_knot[-2])
    d2y_knot[-2] += zbl_class.eval_hess(x_knot[-2])
        
    coef[-1] = fit_poly(x_knot[-2:], y_knot[-2:], dy_knot[-2:], d2y_knot[-2:])
        
    return coef, x_knot

def fit_F(x, y, dy, d2y, dof = 6):

    n = len(x) + 1

    x_knot = np.zeros((n,))
    y_knot = np.zeros((n,))
    dy_knot = np.zeros((n,))
    d2y_knot = np.zeros((n,))
    
    x_knot[1:] = x
    y_knot[1:-1] = y
    dy_knot[:-1] = dy
    d2y_knot[:-1] = d2y

    x_with_index = np.column_stack([x_knot, np.arange(n)])

    x_with_index = x_with_index[np.argsort(x_with_index[:,0])]

    x_knot = x_with_index[:,0]

    x_index = x_with_index[:,1].astype(int)

    y_knot = y_knot[x_index]
    dy_knot = dy_knot[x_index]
    d2y_knot = d2y_knot[x_index]

    coef = fit_piecewise_poly(x_knot, y_knot, dy_knot, d2y_knot, dof)

    return coef, x_knot

def fit_Rho(x, y, dy, d2y, dof = 6):

    n = len(x) + 1

    x_knot = np.zeros((n,))
    y_knot = np.zeros((n,))
    dy_knot = np.zeros((n,))
    d2y_knot = np.zeros((n,))
    
    x_knot[1:] = x
    y_knot[:-1] = y
    dy_knot[:-1] = dy
    d2y_knot[:-1] = d2y

    x_with_index = np.column_stack([x_knot, np.arange(n)])

    x_with_index = x_with_index[np.argsort(x_with_index[:,0])]

    x_knot = x_with_index[:,0]

    x_index = x_with_index[:,1].astype(int)

    y_knot = y_knot[x_index]
    dy_knot = dy_knot[x_index]
    d2y_knot = d2y_knot[x_index]
    
    coef = fit_piecewise_poly(x_knot, y_knot, dy_knot, d2y_knot, dof)

    return coef, x_knot

def loss_func(sample,pot_dict, pot_params, starting_lines, F_nknots = 3, Rho_nknots = 3, V_nknots = 4):

    potfile_dest = 'Potentials/test.eam.alloy'

    coef_dict, knot_dict = fit_sample(sample, F_nknots, Rho_nknots, V_nknots)

    pot_dict = update_potfile(pot_dict, pot_params, coef_dict, knot_dict)
     
    write_pot(pot_dict, starting_lines, potfile_dest)

    loss = sim_defect_set(potfile_dest)

    with open('loss.txt','a') as file:
        file.write('%f \n' % loss)
    return loss