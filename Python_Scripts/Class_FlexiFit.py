import numpy as np
import os 
import json 
from Handle_Files import write_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL
from Handle_Dictionaries import binding_fitting

class Fitting_Potential():

    def __init__(self, pot_lammps, hyperparams, potlines, proc_id = 0):

        # Decompose Sample as follows: [ F, Rho, W-He, H-He, He-He ]

        # For F the knots are: [0, rho_1 ... rho_f, rho_c], [0, F(rho_1) ... F(rho_f), F(rho_c)] f + 2 is the number knot points requires: 2f + 1 params 

        # For Rho the knots are: [0, r_1 ... r_rho, r_c], [0, Rho(r_1) ... Rho(r_rho), Rho(r_c)] rho + 2 is the number knot points requires: 2rho + 1 params

        # For V (pair pot) are: [0, r_1 ... r_v, r_c], [0, V(r_1) ... V(r_v), -Z(r_c)] v + 2 is the number knot points requires: 2v params

        self.pot_lammps = pot_lammps

        self.proc_id = proc_id

        self.keys  = ['He_F(rho)','He_rho(r)', 'W-He', 'H-He', 'He-He']

        self.hyper = hyperparams

        self.potlines = potlines

        self.nf = 0
        self.nrho = -1
        self.nv = 3

        self.knot_pts = {}

        self.knot_pts['He_F(rho)'] = np.linspace(0, self.hyper['rho_c'], self.nf + 2)
        self.knot_pts['He_rho(r)'] = np.linspace(0, self.hyper['rc'], self.nrho + 2)
        self.knot_pts['W-He'] = np.array([0, 2.0376, 2.8161, 3.6429, self.hyper['rc']])
        self.knot_pts['H-He'] = np.array([0, 1.4841, 2.3047, 3.6429, self.hyper['rc']])
        self.knot_pts['He-He'] = np.array([0, 1.4841, 2.3047, 3.6429, self.hyper['rc']])

        self.map = {}

        map_idx = [self.nf + 1] + [self.nrho + 1] + 3*[self.nv]

        idx = 0
        iter = 0

        idx = 0
        iter = 0

        for key in self.keys:
            self.map[key] = slice(idx, idx + map_idx[iter])
            idx += map_idx[iter]
            iter += 1
        
        self.len_sample = idx

    def sample_to_array(self, sample_dict):
        
        sample_lst = []

        for key in self.keys:

            for val in sample_dict[key]:

                sample_lst.append(val)

        return np.array(sample_lst)
    
    def array_to_sample(self, sample_arr):
        
        sample_dict = {}

        for key in self.keys:
            sample_dict[key] = sample_arr[self.map[key]]

        return sample_dict


    def init_sample(self, isrand = False):

        sample = np.zeros((self.len_sample,))

        if isrand:
            
            sample = self.gen_rand()

        else:
            
            with open("Init.json", "r") as json_file:
                loaded_data = json.load(json_file)

            ref = {}
            for key, value in loaded_data.items():
                ref[key] = np.array(value)

            # Initialize based on prior beliefs
            # Knot points are equispaced in the given range

            # Assume linear relationship between F and rho
            sample[self.map['He_F(rho)']] = 0.37*np.linspace(0, self.hyper['rho_c'], self.nf + 2)[1:]

                        
            # Assume that the electron density is zero
            sample[self.map['He_rho(r)']] = np.zeros((self.nrho + 1,))

            charge = np.array([[74, 2],
                              [1, 2],
                              [2, 2]])
            
            
            for i, key in enumerate(['W-He', 'H-He', 'He-He']):
                
                zbl = ZBL(charge[i,0], charge[i,1])
                                
                idx = np.floor(sample[self.knot_pts[key]]/self.hyper['dr']).astype(int)

                sample[self.map[key]] = ref[key][idx]/sample[self.knot_pts[key]] - zbl.eval_zbl(sample[self.knot_pts[key]])

        return sample
    
    def gen_rand(self):
            
        sample = np.zeros((self.len_sample,))
        
        # Randomly Generate Knot Values for F(rho)
        sample[self.map['He_F(rho)']] = self.hyper['rho_c']*np.random.rand(self.nf + 1)

        # Randomly Generate Knot Values for Rho(r)
        sample[self.map['He_rho(r)']] = np.random.randn(self.nrho + 1)*1e-1

        for i, key in enumerate(['W-He', 'H-He', 'He-He']):

            # Randomly Generate Knot Values for Rho(r)
            scale = 2
            shift = 0.5

            sample[self.map[key]] = scale*(np.random.rand(self.nv) - shift)
    
        return sample
    
    def polyfit(self, x_arr, y_arr, dy_arr, d2y_arr):
        
        n_none = 0

        for lst in [y_arr, dy_arr, d2y_arr]:
            
            lst = lst.tolist()
            n_none += lst.count(None)
        
        dof = 3*len(x_arr) - n_none

        Phi = []
        Y   = []

        for i, x in enumerate(x_arr):

            y = y_arr[i]
            dy = dy_arr[i]
            d2y = d2y_arr[i]

            if y is not None:
                Phi.append(np.array([x**i for i in range(dof)]).T)
                Y.append(y)

            if dy is not None:
                Phi.append(np.array([i*x**np.clip(i-1, a_min=0, a_max=None) for i in range(dof)]).T)
                Y.append(dy)

            if d2y is not None:
                Phi.append(np.array([i*(i-1)*x**np.clip(i-2, a_min=0, a_max=None) for i in range(dof)]).T)
                Y.append(d2y)
            
        Phi = np.array(Phi)

        Y  = np.array(Y)

        return np.linalg.solve(Phi, Y)

    def polyval(self, x, coef, func = True, grad = False, hess = False):

        dof = len(coef)

        if func:
            Phi = np.array([x**i for i in range(dof)]).T
        
        elif grad:
            Phi = np.array([i*x**np.clip(i-1, a_min=0, a_max=None) for i in range(dof)]).T

        elif hess:
            Phi = np.array([i*(i-1)*x**np.clip(i-2, a_min=0, a_max=None) for i in range(dof)]).T

        if x.ndim == 1:
            return np.dot(Phi, coef)

        else:
            return np.dot(Phi, coef.reshape(-1,1)).flatten()
    
    def fit_sample(self, sample):

        coef_dict = {}

        y = np.zeros((self.nf + 2,))

        y[1:] = sample[self.map['He_F(rho)']]

        dy = np.full(y.shape, None, dtype=object)

        d2y = np.full(y.shape, None, dtype=object)

        coef_dict['He_F(rho)'] = self.polyfit(self.knot_pts['He_F(rho)'], y, dy, d2y)


        y = np.zeros((self.nrho + 2,))

        y[:-1] = sample[self.map['He_rho(r)']]

        dy = np.full(y.shape, None)

        d2y = np.full(y.shape, None)

        dy[-1] = 0

        d2y[-1] = 0

        coef_dict['He_rho(r)'] = self.polyfit(self.knot_pts['He_rho(r)'], y, dy, d2y)

        charge = [[74, 2],[1, 2],[2, 2]]

        for i, key in enumerate(['W-He', 'H-He', 'He-He']):

            zbl_class = ZBL(charge[i][0], charge[i][1])
            
            x = self.knot_pts[key]

            y = np.zeros((self.nv + 2,))

            y[1:-1] = sample[self.map[key]]

            dy = np.full(y.shape, None)

            d2y = np.full(y.shape, None)

            dy[0] = 0

            d2y[0] = 0

            y[-1] = -zbl_class.eval_zbl(x[-1])[0]

            dy[-1] = -zbl_class.eval_grad(x[-1])[0]

            d2y[-1] = -zbl_class.eval_hess(x[-1])[0]

            coef_dict[key] = self.polyfit(x, y, dy, d2y)

        return coef_dict
    
    def sample_to_file(self, sample):

        coef_dict = self.fit_sample(sample)
        
        rho = np.linspace(0, self.hyper['rho_c'], self.hyper['Nrho'])

        r = np.linspace(0, self.hyper['rc'], self.hyper['Nr'])

        self.pot_lammps['He_F(rho)'] = self.polyval(rho, coef_dict['He_F(rho)'], func = True, grad = False, hess = False)
        self.pot_lammps['He_rho(r)'] = self.polyval(r, coef_dict['He_rho(r)'], func = True, grad = False, hess = False)

        charge = [[74, 2],[1, 2],[2, 2]]
        for i, key in enumerate(['W-He', 'H-He', 'He-He']):

            zbl_class = ZBL(charge[i][0], charge[i][1])
            zbl = zbl_class.eval_zbl(r[1:])

            poly = self.polyval(r[1:], coef_dict[key], func = True, grad = False, hess = False)

            self.pot_lammps[key][1:] = r[1:]*(zbl + poly)



def optim_loss(sample, fitting_class, ref_formations, iteration = 0, output_folder = 'Optimization_Files'):

    potloc = 'Potentials/test.%d.eam.alloy' % fitting_class.proc_id
    
    fitting_class.sample_to_file(sample)
     
    write_pot(fitting_class.pot_lammps, fitting_class.potlines, potloc)

    test_formations = sim_defect_set(potloc, ref_formations)
    
    ref_binding = binding_fitting(ref_formations)
    test_binding = binding_fitting(test_formations)

    loss = 0 

    loss = (test_formations['V0H0He1']['val'] - ref_formations['V0H0He1']['val'])**2

    # loss += (test_formations['V0H0He1']['rvol'] - ref_formations['V0H0He1']['rvol'])**2

    loss += np.sum((test_binding - ref_binding)**2)

    test_rvol = []

    for key in ref_formations:
        test_rvol.append(test_formations[key]['rvol'])

        if ref_formations[key]['rvol'] is not None:
            loss += (test_formations[key]['rvol'] - ref_formations[key]['rvol'])**2

    test_rvol = np.array(test_rvol)

    # Open the file in 'append' mode
    with open('%s/Loss_Files/loss_%d.txt' % (output_folder, iteration), 'a') as file:
        
        # Write the loss value
        file.write('Loss: %f ' % loss)
        
        file.write(' SIA Ef: %f' % (test_formations['V0H0He1']['val'] - ref_formations['V0H0He1']['val']))
        # Use numpy.savetxt to write the NumPy array to the file
        file.write(' Binding: ')

        np.savetxt(file, test_binding - ref_binding, fmt='%.2f', newline=' ')

        file.write(' Rvol: ')
        np.savetxt(file, test_rvol, fmt='%.2f', newline=' ')

        # Add a newline character at the end
        file.write('\n')
    
    with open('%s/Sample_Files/samples_%d.txt' % (output_folder, iteration), 'a') as file:

        np.savetxt(file, sample, fmt='%f', newline=' ')
        file.write('\n')

    return loss


def sample_loss(sample, fitting_class, ref_formations, sample_filepath = 'samples.txt'):

    potloc = 'Potentials/test.%d.eam.alloy' % fitting_class.proc_id
    
    fitting_class.sample_to_file(sample)
     
    write_pot(fitting_class.pot_lammps, fitting_class.potlines, potloc)

    test_formations = sim_defect_set(potloc, ref_formations)
    
    ref_binding = binding_fitting(ref_formations)
    test_binding = binding_fitting(test_formations)

    loss = 0 

    loss = (test_formations['V0H0He1']['val'] - ref_formations['V0H0He1']['val'])**2

    # loss += (test_formations['V0H0He1']['rvol'] - ref_formations['V0H0He1']['rvol'])**2
    binding_loss = (test_binding - ref_binding)**2
    loss += np.sum(binding_loss)

    rvol_loss_lst = []
    # test_rvol = []

    for key in ref_formations:
        # test_rvol.append(test_formations[key]['rvol'])

        if ref_formations[key]['rvol'] is not None:
            
            rvol_loss = (test_formations[key]['rvol'] - ref_formations[key]['rvol'])**2
            rvol_loss_lst.append(rvol_loss)
            loss += rvol_loss

    rvol_loss_lst = np.array(rvol_loss_lst)

    with open(sample_filepath, 'a') as file:
        file.write('%.2f ' % loss)
        np.savetxt(file, sample, fmt='%f', newline=' ')
        file.write(' %.2f ' %  (test_formations['V0H0He1']['val'] - ref_formations['V0H0He1']['val']))
        np.savetxt(file, binding_loss, fmt='%.2f', newline=' ')
        file.write(' ')
        np.savetxt(file, rvol_loss_lst, fmt='%.2f', newline=' ')
        file.write('\n')

    return loss