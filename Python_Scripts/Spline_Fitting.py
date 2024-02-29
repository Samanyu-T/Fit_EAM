import numpy as np
import os 
import sys 
from Handle_Files import write_pot
from Simulate_Defect_Set import sim_defect_set
from ZBL_Class import ZBL
from Handle_Dictionaries import binding_fitting
import time 

class Fitting_Potential():

    def __init__(self, pot_lammps, bool_fit, hyperparams, potlines, n_knots, machine='', proc_id = 0, write_dir = ''):

        # Decompose Sample as follows: [ F, Rho, W-He, H-He, He-He ]

        # For F the knots are: [0, rho_1 ... rho_f, rho_c], [0, F(rho_1) ... F(rho_f), F(rho_c)] f + 2 is the number knot points requires: 2f + 1 params 

        # For Rho the knots are: [0, r_1 ... r_rho, r_c], [0, Rho(r_1) ... Rho(r_rho), Rho(r_c)] rho + 2 is the number knot points requires: 2rho + 1 params

        # For V (pair pot) are: [0, r_1 ... r_v, r_c], [0, V(r_1) ... V(r_v), -Z(r_c)] v + 2 is the number knot points requires: 2v params

        self.pot_lammps = pot_lammps

        self.write_dir = write_dir

        self.lammps_folder = os.path.join(write_dir, 'Lammps_Dump_%d' % proc_id)

        self.pot_folder = os.path.join(write_dir, 'Potentials')

        self.proc_id = proc_id

        self.keys  = ['He_F(rho)','He_rho(r)', 'W-He', 'H-He', 'He-He']

        self.hyper = hyperparams

        self.potlines = potlines
        self.bool_fit = bool_fit

        self.nf = n_knots[0] - 1
        self.nrho = n_knots[1] - 1
        self.nv = n_knots[2]

        high_symm_pts = np.array([2.7236, 1.7581, 3.6429])

        phi_knots = np.hstack([0, high_symm_pts[:self.nv], self.hyper['rc']])

        self.knot_pts = {}
        self.machine = machine

        self.knot_pts['He_F(rho)'] = np.linspace(0, self.hyper['rho_c'], self.nf + 2)
        self.knot_pts['He_rho(r)'] = np.linspace(0, self.hyper['rc'], self.nrho + 2)
        self.knot_pts['W-He'] = np.sort(phi_knots)
        self.knot_pts['H-He'] = np.sort(phi_knots)
        self.knot_pts['He-He'] = np.sort(phi_knots)

        self.map = {}

        full_map_idx = [self.nf + 1] + [self.nrho + 1] + 3*[self.nv*3]

        map_idx = []

        for idx, key in enumerate(bool_fit):
            if bool_fit[key]:
                map_idx.append(full_map_idx[idx])

        idx = 0
        iter = 0

        idx = 0
        iter = 0

        for key in self.keys:
            if self.bool_fit[key]:
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
    
    def gen_rand(self):
            
        sample = np.zeros((self.len_sample,))

        if self.bool_fit['He_F(rho)']:
            # Randomly Generate Knot Values for F(rho)
            sample[self.map['He_F(rho)']] = self.hyper['rho_c']*np.random.rand(self.nf + 1)

        if self.bool_fit['He_rho(r)']:
            # Randomly Generate Knot Values for Rho(r)
            sample[self.map['He_rho(r)']] = np.random.rand(self.nrho + 1)

        ymax = 4
        dymax = 20
        d2ymax = 100

        for key in ['W-He', 'H-He', 'He-He']:
            if self.bool_fit[key]:
                # Randomly Generate Knot Values for Phi(r)
                for i in range(self.nv):

                    sample[self.map[key]][3*i]     = ymax*(np.random.rand() - 0.5)
                    sample[self.map[key]][3*i + 1] = dymax*(np.random.rand() - 0.5)
                    sample[self.map[key]][3*i + 2] = d2ymax*(np.random.rand() - 0.5)

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

        Phi = np.array([])

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

    def splinefit(self, x_arr, y_arr, dy_arr, d2y_arr):

        coef_lst = []

        for i in range(len(x_arr) - 1):
            coef_lst.append(self.polyfit(x_arr[i:i+2], y_arr[i:i+2], dy_arr[i:i+2], d2y_arr[i:i+2]))
        
        return coef_lst
    
    def splineval(self, x_arr, coef_lst, knots_pts, func = True, grad = False, hess = False):
        
        y = np.zeros(x_arr.shape)

        for i ,x in enumerate(x_arr): 
            idx = np.searchsorted(knots_pts, x) - 1
            y[i] = self.polyval(x, coef_lst[idx], func, grad, hess).flatten()

        return y
    
    def fit_sample(self, sample):

        coef_dict = {}

        if self.bool_fit['He_F(rho)']:
            y = np.zeros((self.nf + 2,))

            y[1:] = sample[self.map['He_F(rho)']]

            dy = np.full(y.shape, None, dtype=object)

            d2y = np.full(y.shape, None, dtype=object)

            coef_dict['He_F(rho)'] = self.polyfit(self.knot_pts['He_F(rho)'], y, dy, d2y)

        if self.bool_fit['He_rho(r)']:

            y = np.zeros((self.nrho + 2,))

            y[:-1] = sample[self.map['He_rho(r)']]

            dy = np.full(y.shape, None)

            d2y = np.full(y.shape, None)

            dy[-1] = 0

            d2y[-1] = 0

            coef_dict['He_rho(r)'] = self.polyfit(self.knot_pts['He_rho(r)'], y, dy, d2y)

        charge = [[74, 2],[1, 2],[2, 2]]

        for i, key in enumerate(['W-He', 'H-He', 'He-He']):

            if self.bool_fit[key]:
                zbl_class = ZBL(charge[i][0], charge[i][1])
                
                x = self.knot_pts[key]

                y = np.zeros((self.nv + 2,))

                dy = np.zeros((self.nv + 2,))

                d2y = np.zeros((self.nv + 2,))

                for i in range(self.nv):

                    y[i + 1] = sample[self.map[key]][3*i]

                    dy[i + 1] = sample[self.map[key]][3*i + 1]

                    d2y[i + 1] = sample[self.map[key]][3*i + 2]

                y[-1] = -zbl_class.eval_zbl(x[-1])[0]

                dy[-1] = -zbl_class.eval_grad(x[-1])[0]

                d2y[-1] = -zbl_class.eval_hess(x[-1])[0]

                coef_dict[key] = self.splinefit(x, y, dy, d2y)

        return coef_dict
    
    def sample_to_file(self, sample):

        coef_dict = self.fit_sample(sample)
        
        rho = np.linspace(0, self.hyper['rho_c'], self.hyper['Nrho'])

        r = np.linspace(0, self.hyper['rc'], self.hyper['Nr'])

        if self.bool_fit['He_F(rho)']:
            self.pot_lammps['He_F(rho)'] = self.polyval(rho, coef_dict['He_F(rho)'], func = True, grad = False, hess = False)
        
        if self.bool_fit['He_rho(r)']:
            self.pot_lammps['He_rho(r)'] = self.polyval(r, coef_dict['He_rho(r)'], func = True, grad = False, hess = False)

        charge = [[74, 2],[1, 2],[2, 2]]

        for i, key in enumerate(['W-He', 'H-He', 'He-He']):
            if self.bool_fit[key]:
                zbl_class = ZBL(charge[i][0], charge[i][1])
                zbl = zbl_class.eval_zbl(r[1:])

                poly = self.splineval(r[1:], coef_dict[key], self.knot_pts[key] ,func = True, grad = False, hess = False)

                self.pot_lammps[key][1:] = r[1:]*(zbl + poly)


def loss_func(sample, fitting_class, ref_formations, output_folder, genetic = False, write = False):

    potloc = '%s/test.%d.eam.alloy' % (fitting_class.pot_folder, fitting_class.proc_id)

    fitting_class.sample_to_file(sample)
     
    write_pot(fitting_class.pot_lammps, fitting_class.potlines, potloc)

    test_formations = sim_defect_set(potloc, ref_formations, fitting_class.machine, fitting_class.lammps_folder, fitting_class.proc_id)
    
    ref_binding = binding_fitting(ref_formations)

    test_binding = binding_fitting(test_formations)

    loss = 0

    # Quadratic Loss of Interstitial Formation Energies
    tet_diff = (test_formations['V0H0He1']['val'] - ref_formations['V0H0He1']['val'])

    oct_diff = (test_formations['V0H0He1_oct']['val'] - ref_formations['V0H0He1_oct']['val'])

    migration = (test_formations['V0H0He1_inter2']['val'] - test_formations['V0H0He1']['val'])

    loss = tet_diff**2

    loss += (oct_diff - tet_diff)**2

    loss += np.abs(migration - 0.06)

    tet_dist = np.linalg.norm(test_formations['V0H0He1']['pos'][-1][0] - np.array(ref_formations['V0H0He1']['pos'][-1][0])) 

    oct_dist = np.linalg.norm(test_formations['V0H0He1_oct']['pos'][-1][0] - np.array(ref_formations['V0H0He1_oct']['pos'][-1][0]))

    inter_dist = np.abs(test_formations['V0H0He1_inter2']['pos'][-1][0][0] - test_formations['V0H0He1_inter2']['pos'][-1][0][1]) + \
                 np.abs(test_formations['V0H0He1_inter2']['pos'][-1][0][2] - ref_formations['V0H0He1_inter2']['pos'][-1][0][2])
    
    # Soft Constraint to ensure correct ordering of formation energies and relaxation volumes

    loss +=  100*(test_formations['V0H0He1']['val'] > test_formations['V0H0He1_inter2']['val']) 

    loss +=  100*(test_formations['V0H0He1_inter2']['val'] > test_formations['V0H0He1_oct']['val']) 

    loss +=  100*(test_formations['V0H0He1']['rvol'] > test_formations['V0H0He1_oct']['rvol'])

    loss +=  100*( not (-0.005 < test_formations['V0H0He1_inter']['val'] - test_formations['V0H0He1']['val']  < 0.05 ) )

    loss += 100*(tet_dist>1e-3)

    loss += 100*(oct_dist>1e-3)

    loss += 100*(inter_dist>1e-3)

    # Quadratic Loss of Binding Energies
    for i in range(len(ref_binding)):
        loss += np.sum((test_binding[i] - ref_binding[i])**2)
    
    # Quadratic Loss of Relaxation Volumes
    for key in ref_formations:
        if ref_formations[key]['rvol'] is not None:
            loss += (test_formations[key]['rvol'] - ref_formations[key]['rvol'])**2

    if write:
        # Write the Loss and the Sample Data to files for archiving
        with open(os.path.join(output_folder,'loss.txt'), 'a') as file:

            file.write('Loss: %f TIS: %f IIS: %f OIS: %f Interstitial_Binding: ' % (loss,
                                                    test_formations['V0H0He1']['val'], 
                                                    test_formations['V0H0He1_inter']['val'], 
                                                    test_formations['V0H0He1_oct']['val']
                                                
                                                    )
                    )
            np.savetxt(file, test_binding[0] - ref_binding[0], fmt = '%f', newline= ' ')

            file.write('Vacancy_Binding: ')
            
            np.savetxt(file, test_binding[1] - ref_binding[1], fmt = '%f', newline= ' ')

            file.write('Di-Vacancy_Binding: ')
            
            np.savetxt(file, test_binding[2] - ref_binding[2], fmt = '%f', newline= ' ')

            file.write('Rvol: %f %f %f' % 
                    (test_formations['V0H0He1']['rvol'] - ref_formations['V0H0He1']['rvol'],
                        test_formations['V0H0He1_oct']['rvol'] - ref_formations['V0H0He1_oct']['rvol'],
                        test_formations['V1H0He1']['rvol'] - ref_formations['V1H0He1']['rvol']
                        )
                    )

            # Add a newline character at the end
            file.write('\n')
        
        sample_filename = 'samples.txt'

        if genetic:
            sample_filename = 'population.txt'

        with open(os.path.join(output_folder, sample_filename), 'a') as file:
            np.savetxt(file, sample, fmt = '%f', newline=' ')
            file.write('\n')

    return loss

def random_sampling(ref_formations, fitting_class, max_time, output_folder):
    
    lst_loss = []
    lst_samples = []
    t_init = time.perf_counter()
    idx = 0

    while True:
        sample = fitting_class.gen_rand()
        loss = loss_func(sample, fitting_class, ref_formations, output_folder, False)
        idx += 1

        lst_loss.append(loss)
        lst_samples.append(sample)
        
        t_end = time.perf_counter()
        
        if t_end - t_init > max_time:
            break

        if idx % 1000 == 0 and fitting_class.proc_id == 0:
            print(t_end - t_init)
            sys.stdout.flush()  

    lst_loss = np.array(lst_loss)
    lst_samples = np.array(lst_samples)

    idx = np.argsort(lst_loss)

    lst_loss = lst_loss[idx]
    lst_samples = lst_samples[idx]

    n = int( len(lst_loss) * 0.1 )

    np.savetxt(os.path.join(output_folder, 'Filtered_Samples.txt'), lst_samples[:n])
    np.savetxt(os.path.join(output_folder, 'Filtered_Loss.txt'), lst_loss[:n])

def gaussian_sampling(ref_formations, fitting_class, max_time, output_folder, cov, mean):
    
    lst_loss = []
    lst_samples = []
    
    t_init = time.perf_counter()
    idx = 0
    while True:
        sample = np.random.multivariate_normal(mean=mean, cov=cov)
        loss = loss_func(sample, fitting_class, ref_formations, output_folder, False)
        idx += 1
        
        lst_loss.append(loss)
        lst_samples.append(sample)
        
        t_end = time.perf_counter()
        
        if t_end - t_init > max_time:
            break

        if idx % 1000 == 0 and fitting_class.proc_id == 0:
            print(t_end - t_init)
            sys.stdout.flush()  

    lst_loss = np.array(lst_loss)
    lst_samples = np.array(lst_samples)

    idx = np.argsort(lst_loss)

    lst_loss = lst_loss[idx]
    lst_samples = lst_samples[idx]

    n = int( len(lst_loss) * 0.1 )

    np.savetxt(os.path.join(output_folder, 'Filtered_Samples.txt'), lst_samples[:n])
    np.savetxt(os.path.join(output_folder, 'Filtered_Loss.txt'), lst_loss[:n])



def genetic_algorithm(ref_formations, fitting_class, N_samples, N_steps, mutate_coef = 1, mutate_decay = 1.175, output_folder = '../Genetic'):

    population = np.zeros((N_samples, fitting_class.len_sample))

    for i in range(N_samples):
        population[i] = fitting_class.gen_rand()
    
    for iteration in range(N_steps):                                                                                        
        
        mutate_coef /= mutate_decay
        N_samples = len(population)
        fitness = np.zeros((N_samples,))

        folder = '%s/Iteration_%d' % (output_folder, iteration)

        if not os.path.exists(folder):
            os.mkdir(folder)

        with open('%s/loss.txt' % folder, 'w') as file:
            file.write('')
        
        with open('%s/population.txt' % folder, 'w') as file:
            file.write('')

        for i in range(N_samples):

            loss = loss_func(population[i], fitting_class, ref_formations, folder, True)
            fitness[i] = 1/(1+loss)
        
        k = 3

        N_reproduce = N_samples // k

        new_population =  np.zeros((N_reproduce*k , fitting_class.len_sample))

        for i in range(N_reproduce):
            
            cprob = np.cumsum(fitness/np.sum(fitness))

            rng = np.random.rand(2)

            parent_idx = np.searchsorted(cprob, rng) - 1

            children = np.zeros((k,fitting_class.len_sample))

            for j in range(k):
                children[j] = population[parent_idx[0]] + ((j + 1)/(k + 1))*(population[parent_idx[1]] - population[parent_idx[0]])

            children += children * mutate_coef * (np.random.rand(k ,fitting_class.len_sample) - 0.5)

            new_population[k*i: k*i + k] = children

            population = np.delete(population, parent_idx, axis = 0)

            fitness = np.delete(fitness, parent_idx)
            
        population = new_population

            




