import numpy as np
import os 
import json 
from Handle_Files import write_pot, read_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL

def extend_sample(sample, x, y):
    for element in [x, y]:
        for val in element:
            sample.append(val)
    return sample

def fit_sample(sample, F_nknots = 3, Rho_nknots = 3, V_nknots = 4):

    ''' Sample ordered as follows: F(He), Rho(He), V(W-He), V(H-He), V(He-He) '''
    ''' Inside of each of the above params: sqrt_coef(if it exists), x_knot, y_knot, dy_knot, d2y_knot'''

    map = {}

    keys  = ['He_F(rho)','He_rho(r)', 'W-He', 'H-He', 'He-He']
    items = ['knots', 'y']
    
    # knot constraints: xF[0] = 0, xrho[0] = 0, xV[0] = 0
    # value constraints: F(0) = 0, Rho(xrho[-1]) = 0, V(0) = 0, V(xV[-1]) = 0

    splice_F = [F_nknots - 1, F_nknots - 2]
    splice_Rho = [Rho_nknots - 1, Rho_nknots - 1]
    splice_V = [V_nknots - 1, V_nknots - 2]

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
    coef_dict[key], knot_dict[key] = fit_F(sample[map[key]['knots']], sample[map[key]['y']])

    key = 'He_rho(r)'
    coef_dict[key], knot_dict[key] = fit_Rho(sample[map[key]['knots']], sample[map[key]['y']])

    try:
        key = 'W-He'
        coef_dict[key], knot_dict[key] = fit_V(sample[map[key]['knots']], sample[map[key]['y']], 74, 2)
    
    except:
        print(sample, sample.shape)
        

    key = 'H-He'
    coef_dict[key], knot_dict[key] = fit_V(sample[map[key]['knots']], sample[map[key]['y']], 2, 1)

    key = 'He-He'
    coef_dict[key], knot_dict[key] = fit_V(sample[map[key]['knots']], sample[map[key]['y']], 2, 2)

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
    y = 0.37*knots
    
    sample = extend_sample(sample, knots[1:], y[1:-1])

    coef_dict['He_F(rho)'], knot_dict['He_F(rho)'] = fit_F(knots[1:], y[1:-1])

    knots = np.linspace(0, dr*Nr, Rho_nknots)
    y = np.zeros((Rho_nknots,))

    sample = extend_sample(sample, knots[1:], y[:-1])
    coef_dict['He_rho(r)'], knot_dict['He_rho(r)'] = fit_Rho(knots[1:], y[:-1])

    knots = np.linspace(0, dr*Nr, V_nknots)

    knots_idx = (knots/dr).astype(int)

    charge = [[74, 2], [1, 2], [2,2]]

    for i, key in enumerate(['W-He', 'H-He', 'He-He']):
        
        zbl_class = ZBL(charge[i][0], charge[i][1])

        f = pot[key][knots_idx[1:-1]]/knots[1:-1]
        
        y = f - zbl_class.eval_zbl(knots[1:-1])

        coef_dict[key], knot_dict[key] = fit_V(knots[1:], y, Zi = charge[i][0], Zj = charge[i][1])

        sample = extend_sample(sample, knots[1:], y)
        
    return np.array(sample)
    

def polyval(x, coef = None, dof = None):

    if coef is None:
        return np.array([x**i for i in range(dof)]).T
    
    else:
        dof = len(coef)
        phi = np.array([x**i for i in range(dof)]).T

        if phi.ndim == 1:
            return np.dot(phi, coef)
        else:
            return np.dot(phi, coef.reshape(-1,1)).flatten()


def d_polyval(x, coef = None, dof = None):

    if coef is None:
        return np.array([i*x**np.clip(i-1, a_min=0, a_max=None) for i in range(dof)]).T
    
    else:
        dof = len(coef)
        phi = np.array([i*x**np.clip(i-1, a_min=0, a_max=None) for i in range(dof)]).T

        if phi.ndim == 1:
            return np.dot(phi, coef)
        else:
            return np.dot(phi, coef.reshape(-1,1)).flatten()
         
def d2_polyval(x, coef = None, dof = None):

    if coef is None:
        return np.array([i*(i-1)*x**np.clip(i-2, a_min=0, a_max=None) for i in range(dof)]).T
    
    else:
        dof = len(coef)
        phi = np.array([i*(i-1)*x**np.clip(i-2, a_min=0, a_max=None) for i in range(dof)]).T

        if phi.ndim == 1:
            return np.dot(phi, coef)
        else:
            return np.dot(phi, coef.reshape(-1,1)).flatten()
        
def spline_eval(x, coef_lst, knots):

    # n = len(knots)

    # lower_bound = knots[np.arange(n-1)]
    # upper_bound = knots[np.arange(1,n)]

    # masks = np.array( [( (lower_bound[i] <= x) & (x < upper_bound[i]) ).astype(int) for i in range(n-1)])
    
    # return np.sum( np.array([ masks[i]*(polyval(x, coef = coef_lst[i])) for i in range(n-1)]), axis = 0)

    dr = knots[1] - knots[0]

    idx = np.floor(x/dr).astype(int)

    y = np.zeros((len(x),))

    for i in range(len(x)):
        
        if idx[i] < len(knots) - 1:

            x_norm = (x[i] - knots[idx[i]])/dr

            y[i] = polyval( x_norm, coef_lst[idx[i]])
        
        else:

            y[i] = 0
    
    return y

def spline_fit(x_arr, y_arr, n_knots, init_grad = None, final_grad = None):

    #At x = 0, y = 0, dy = 0, d2y = 0

    dof = 4

    coef_lst = []

    if init_grad is None:

        dof = 2

        # x = x_arr[[0,1]]

        x = np.array([0,1])

        Y = y_arr[[0,1]]

        Phi = polyval(x, coef = None, dof=dof)

        coef = np.linalg.solve(Phi, Y)

        coef_lst.append(coef)
    
    else:

        dof = 4

        # x = x_arr[0]

        x = 0

        phi_y = polyval(x, coef = None, dof=dof)

        phi_dy = d_polyval(x, coef = None, dof=dof)

        phi_d2y = d2_polyval(x, coef = None, dof=dof)

        # x = x_arr[1]

        x = 1

        phi_y1 = polyval(x, coef = None, dof=dof)

        Phi = np.vstack([phi_y, phi_dy, phi_d2y, phi_y1])

        Y = np.hstack([y_arr[0], init_grad[0], init_grad[1], y_arr[1]])

        coef = np.linalg.solve(Phi, Y)

        coef_lst.append(coef)
    
    dof = 4

    for j in range(1, n_knots-2):

        # x = x_arr[j]

        x = 0

        phi_y = polyval(x, coef = None, dof=dof)

        phi_dy = d_polyval(x, coef = None, dof=dof)

        phi_d2y = d2_polyval(x, coef = None, dof=dof)

        # x = x_arr[j + 1]

        x = 1

        dy = d_polyval(x, coef = coef_lst[-1])

        d2y = d2_polyval(x, coef = coef_lst[-1])

        phi_y1 = polyval(x, coef = None, dof=dof)

        Phi = np.vstack([phi_y, phi_dy, phi_d2y, phi_y1])

        Y = np.hstack([y_arr[j], dy, d2y, y_arr[j + 1]])

        coef = np.linalg.solve(Phi, Y)
        
        coef_lst.append(coef)
    
    if final_grad is None:

        dof = 4

        x = 0

        phi_y = polyval(x, coef = None, dof=dof)

        phi_dy = d_polyval(x, coef = None, dof=dof)

        phi_d2y = d2_polyval(x, coef = None, dof=dof)

        # x = x_arr[j + 1]

        x = 1

        dy = d_polyval(x, coef = coef_lst[-1])

        d2y = d2_polyval(x, coef = coef_lst[-1])

        phi_y1 = polyval(x, coef = None, dof=dof)

        Phi = np.vstack([phi_y, phi_dy, phi_d2y, phi_y1])

        Y = np.hstack([y_arr[j], dy, d2y, y_arr[j + 1]])

        coef = np.linalg.solve(Phi, Y)
        
        coef_lst.append(coef)
        
    else:

        dof = 6

        # x = x_arr[n_knots - 2]

        x = 0

        phi_y = polyval(x, coef = None, dof=dof)

        phi_dy = d_polyval(x, coef = None, dof=dof)

        phi_d2y = d2_polyval(x, coef = None, dof=dof)

        # x = x_arr[j + 1]

        x = 1

        dy = d_polyval(x, coef = coef_lst[-1])

        d2y = d2_polyval(x, coef = coef_lst[-1])

        phi_y1 = polyval(x, coef = None, dof=dof)

        phi_dy1 = d_polyval(x, coef = None, dof=dof)

        phi_d2y1 = d2_polyval(x, coef = None, dof=dof)

        Phi = np.vstack([phi_y, phi_dy, phi_d2y, phi_y1, phi_dy1, phi_d2y1])

        Y = np.hstack([y_arr[n_knots - 2], dy, d2y, y_arr[n_knots - 1], final_grad[0], final_grad[1]])

        coef = np.linalg.solve(Phi, Y)

        coef_lst.append(coef)
    
    return coef_lst

def polyfit(x, y, init_grad = None, final_grad=None):

    if init_grad is None and final_grad is None:

        dof = len(x)

        phi = polyval(x, coef = None, dof=dof)  

        return np.linalg.solve(phi, y)

    elif init_grad is not None and final_grad is None:

        dof = len(x) + 2

        phi_y = polyval(x, coef = None, dof=dof)

        phi_dy = d_polyval(x[0], coef = None, dof=dof)

        phi_d2y = d2_polyval(x[0], coef = None, dof=dof)

        phi = np.vstack([phi_y, phi_dy, phi_d2y])

        Y = np.hstack([y, init_grad])

        return np.linalg.solve(phi, Y)
    
    elif init_grad is None and final_grad is not None:

        dof = len(x) + 2

        phi_y = polyval(x, coef = None, dof=dof)

        phi_dy = d_polyval(x[-1], coef = None, dof=dof)

        phi_d2y = d2_polyval(x[-1], coef = None, dof=dof)

        phi = np.vstack([phi_y, phi_dy, phi_d2y])

        Y = np.hstack([y, final_grad])
        
        return np.linalg.solve(phi, Y)
    
    elif init_grad is not None and final_grad is not None:

        dof = len(x) + 4

        phi_y = polyval(x, coef = None, dof=dof)

        phi_dy = d_polyval(x[0], coef = None, dof=dof)

        phi_d2y = d2_polyval(x[0], coef = None, dof=dof)

        phi_dy1 = d_polyval(x[-1], coef = None, dof=dof)

        phi_d2y1 = d2_polyval(x[-1], coef = None, dof=dof)

        phi = np.vstack([phi_y, phi_dy, phi_d2y, phi_dy1, phi_d2y1])

        Y = np.hstack([y.flatten(), init_grad.flatten(), final_grad.flatten()])
        
        return np.linalg.solve(phi, Y)
    

def fit_V(knots, y, Zi, Zj):

    zbl_class = ZBL(Zi, Zj)

    x = np.hstack([0,knots])
    y = np.hstack([0,y,zbl_class.eval_zbl(knots[-1])])
    dy = zbl_class.eval_grad(knots[-1])
    d2y = zbl_class.eval_hess(knots[-1])

    return polyfit(x, y, init_grad=np.zeros((2,)), final_grad=np.array([dy ,d2y])), x

def fit_Rho(knots, y):

    x = np.hstack([0,knots])
    y = np.hstack([y,0])

    return polyfit(x, y, init_grad=None, final_grad=np.zeros((2,))), x

def fit_F(knots, y):

    x = np.hstack([0,knots])
    y = np.hstack([0,y,0])

    return polyfit(x, y, init_grad=None, final_grad=np.zeros((2,))), x

def eval_V(x, coef, knots, Zi, Zj):

    zbl_class = ZBL(Zi, Zj)

    zbl = zbl_class.eval_zbl(x)

    poly = polyval(x, coef)

    return zbl + poly

def eval_Rho(x, coef, knots):

    return polyval(x, coef)

def eval_F(x, sqrt_coef, coef, knots ):

    sqrt = sqrt_coef*np.sqrt(x)

    poly = polyval(x, coef)

    return sqrt + poly

def loss_func(sample,pot_dict, pot_params, starting_lines, F_nknots = 3, Rho_nknots = 3, V_nknots = 4):

    potfile_dest = 'Potentials/test.eam.alloy'

    coef_dict, knot_dict = fit_sample(sample, F_nknots, Rho_nknots, V_nknots)

    pot_dict = update_potfile(pot_dict, pot_params, coef_dict, knot_dict)
     
    write_pot(pot_dict, starting_lines, potfile_dest)

    loss = sim_defect_set(potfile_dest)

    with open('loss.txt','a') as file:
        file.write('%f \n' % loss)
    return loss