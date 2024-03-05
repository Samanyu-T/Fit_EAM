import numpy as np
from lammps import lammps
from mpi4py import MPI
import itertools
import copy 
import os
def test_config(datafile, potfile,output_folder):

    lmp = lammps()

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data %s' % datafile)

    lmp.command('pair_style eam/alloy' )

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('run 0')

    lmp.command('thermo 10')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz')

    filename = datafile.split('/')[-1]

    lmp.command('compute pot all pe/atom')

    lmp.command('run 0')

    lmp.command('write_dump all custom %s/%s id type x y z c_pot' % (output_folder, filename))

    pe = lmp.get_thermo('pe')

    lmp.close()

    return pe
# template to replace MPI functionality for single threaded use
class MPI_to_serial():

    def bcast(self, *args, **kwargs):

        return args[0]

    def barrier(self):

        return 0

class Lammps_Point_Defect():

    def __init__(self, size, n_vac, potfile = 'WHHe_test.eam.alloy', surface = False, depth = 0, 
                 orientx = [1, 0, 0], orienty=[0,1,0], orientz=[0, 0, 1], conv = 1000, machine = ''):

        # try running in parallel, otherwise single thread
        try:

            self.comm = MPI.COMM_WORLD

            self.me = self.comm.Get_rank()

            self.nprocs = self.comm.Get_size()

            self.mode = 'MPI'

        except:

            self.me = 0

            self.nprocs = 1

            self.comm = MPI_to_serial()

            self.mode = 'serial'

        self.comm.barrier()

        self.orientx = orientx
        self.orienty = orienty
        self.orientz = orientz

        self.R = np.linalg.inv(np.array([orientx, orienty, orientz]).T)

        self.size  = int(size)
        self.n_vac = int(n_vac)

        self.depth = depth

        self.defect_pos = np.array([self.size//2, self.size//2, self.depth])

        if surface:
            self.surface = 10
        else:
            self.surface = 0

        self.potfile = potfile
        
        self.conv = conv

        self.machine = machine

        self.Perfect_Crystal()  

    def Perfect_Crystal(self,alattice = 3.144221296574379):

        ''' xyz_inter gives a list of the intersitial atoms for each species i,e W H He in that order
            they are in lattice units and are consistent with the Lammps co-ords of the cell'''

        lmp = lammps(name = self.machine, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary p p p')

        lmp.command('lattice bcc %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % 
                    (alattice,
                    self.orientx[0], self.orientx[1], self.orientx[2],
                    self.orienty[0], self.orienty[1], self.orienty[2], 
                    self.orientz[0], self.orientz[1], self.orientz[2]
                    ) 
                    )

        # size_x = self.size // np.sqrt(np.dot(self.orientx, self.orientx))
        # size_y = self.size // np.sqrt(np.dot(self.orienty, self.orienty))
        # size_z = self.size // np.sqrt(np.dot(self.orientz, self.orientz))

        lmp.command('region r_simbox block %f %f %f %f %f %f units lattice' % (

            -1e-9, self.size + 1e-9, -1e-9, self.size + 1e-9, -1e-9 - 0.5*self.surface, self.size + 1e-9 + 0.5*self.surface
        ))

        lmp.command('region r_atombox block %f %f %f %f %f %f units lattice' % (

            -1e-4, self.size + 1e-4, -1e-4, self.size + 1e-4, -1e-4, self.size + 1e-4
        ))

        lmp.command('create_box 3 r_simbox')
        
        lmp.command('create_atoms 1 region r_atombox')

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % self.potfile)

        lmp.command('fix 3 all box/relax aniso 0.0')

        lmp.command('run 0')

        lmp.command('thermo 5')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
        
        lmp.command('minimize 1e-9 1e-12 100 1000')
        lmp.command('minimize 1e-12 1e-15 100 1000')
        lmp.command('minimize 1e-13 1e-16 %d %d' % (100000, 100000))

        lmp.command('write_data Lammps_Dump/perfect.data')
        
        pxx = lmp.get_thermo('pxx')
        pyy = lmp.get_thermo('pyy')
        pzz = lmp.get_thermo('pzz')
        pxy = lmp.get_thermo('pxy')
        pxz = lmp.get_thermo('pxz')
        pyz = lmp.get_thermo('pyz')

        self.stress0 = np.array([pxx, pyy, pzz, pxy, pxz, pyz]) 
        
        self.alattice = lmp.get_thermo('xlat') / np.sqrt(np.dot(self.orientx, self.orientx))

        # self.alattice = lmp.get_thermo('lx') / ( size_x * np.sqrt(np.dot(self.orientx, self.orientx)) )

        self.N_atoms = lmp.get_natoms()

        self.pe0 = lmp.get_thermo('pe')

        self.vol0 = lmp.get_thermo('vol')

        lmp.close()

    def Build_Defect(self, xyz_inter = [[], [], []], alattice = 3.144221296574379, dump_name = ''):

        ''' xyz_inter gives a list of the intersitial atoms for each species i,e W H He in that order
            they are in lattice units and are consistent with the Lammps co-ords of the cell'''

        lmp = lammps(name = self.machine, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data Lammps_Dump/perfect.data')

        #Create a Vacancy of n-atoms along the <1,1,1> direction the vacancy will be at the centre of the cell

        for i in range(self.n_vac):

            vac_pos = (self.defect_pos + 0.5*(i+1)*np.ones((3,))) * self.alattice

            lmp.command('region r_vac_%d sphere %f %f %f 0.2 units box'     
                        % (i, vac_pos[0], vac_pos[1], vac_pos[2]) )
            
            lmp.command('delete_atoms region r_vac_%d ' % i)

        #Create the set of intersitials

        for element, xyz_element in enumerate(xyz_inter):
            for xyz in xyz_element:
                lmp.command('create_atoms %d single %f %f %f units box' % (element + 1, xyz[0], xyz[1], xyz[2]))

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % self.potfile)

        lmp.command('run 0')

        lmp.command('thermo 50')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        lmp.command('minimize 1e-9 1e-12 100 1000')

        lmp.command('minimize 1e-12 1e-15 100 1000')

        lmp.command('minimize 1e-13 1e-16 %d %d' % (self.conv, self.conv))
        
        pe = lmp.get_thermo('pe')

        pxx = lmp.get_thermo('pxx')
        pyy = lmp.get_thermo('pyy')
        pzz = lmp.get_thermo('pzz')
        pxy = lmp.get_thermo('pxy')
        pxz = lmp.get_thermo('pxz')
        pyz = lmp.get_thermo('pyz')

        self.vol = lmp.get_thermo('vol')

        self.stress_voigt = np.array([pxx, pyy, pzz, pxy, pxz, pyz]) - self.stress0

        self.strain_tensor = self.find_strain()

        self.relaxation_volume = 2*np.trace(self.strain_tensor)*self.vol/self.alattice**3

        # self.relaxation_volume = 2*(self.vol - self.vol0)/self.alattice**3

        pe = lmp.get_thermo('pe')

        xyz_system = np.array(lmp.gather_atoms('x',1,3))

        xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

        xyz_inter_relaxed = [[],[],[]]

        N_atoms = self.N_atoms - self.n_vac

        idx = 0

        for element, xyz_element in enumerate(xyz_inter):
            for i in range(len(xyz_element)):
                vec = (xyz_system[N_atoms + idx])
                xyz_inter_relaxed[element].append(vec.tolist())
                idx += 1


        if dump_name == '':
            
            dump_name = '../Lammps_Dump/Surface/Orient_%d%d%d_Depth(%.2f)' %(self.orientx[0], self.orientx[1], self.orientx[2], self.depth)
        
        lmp.command('write_data %s.data' % dump_name)

        lmp.command('write_dump all custom %s.atom id x y z' % dump_name)

        lmp.close()

        e0 = self.pe0/self.N_atoms

        return pe - self.pe0 + self.n_vac*e0 + len(xyz_inter[1])*2.121, self.relaxation_volume, xyz_inter_relaxed
    
    def Find_Min_Config(self, init_config, atom_to_add = 3):

        sites = self.get_all_sites()

        lmp = lammps(name = self.machine, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])
        
        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data %s.data' % init_config)

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % self.potfile)

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        N = lmp.get_natoms()

        filename = os.path.basename(init_config)

        n_inter = [0, int(filename[3]), int(filename[-1])]

        n_inter[atom_to_add - 1] += 1

        pe_lst = []
        pos_lst = []

        for site_type in sites:
            for site in sites[site_type]:
                
                site *= self.alattice

                lmp.command('create_atoms %d single %f %f %f units box' % (atom_to_add, site[0], site[1], site[2]))

                lmp.command('run 0')
                
                pe_lst.append(lmp.get_thermo('pe'))

                pos_lst.append(site)

                lmp.command('group delete id %d' % (N + 1) )

                lmp.command('delete_atoms group delete')

        pe_arr = np.array(pe_lst)

        min_idx = pe_arr.argmin()

        lmp.command('create_atoms %d single %f %f %f units box' % 
                    (atom_to_add, pos_lst[min_idx][0], pos_lst[min_idx][1], pos_lst[min_idx][2]))

        lmp.command('minimize 1e-9 1e-12 10 10')
        lmp.command('minimize 1e-9 1e-12 100 100')
        lmp.command('minimize 1e-9 1e-12 10000 10000')

        pe = lmp.get_thermo('pe')

        pxx = lmp.get_thermo('pxx')
        pyy = lmp.get_thermo('pyy')
        pzz = lmp.get_thermo('pzz')
        pxy = lmp.get_thermo('pxy')
        pxz = lmp.get_thermo('pxz')
        pyz = lmp.get_thermo('pyz')

        self.vol = lmp.get_thermo('vol')

        self.stress_voigt = np.array([pxx, pyy, pzz, pxy, pxz, pyz]) - self.stress0

        self.strain_tensor = self.find_strain()

        self.relaxation_volume = 2*np.trace(self.strain_tensor)*self.vol/self.alattice**3

        pe = lmp.get_thermo('pe')

        lammps_folder = os.path.dirname(init_config)

        lmp.command('write_data %s/V%dH%dHe%d.data' % (lammps_folder, self.n_vac, n_inter[1], n_inter[2]))
        
        lmp.command('write_dump all custom  %s/V%dH%dHe%d.atom id x y z'% (lammps_folder, self.n_vac, n_inter[1], n_inter[2]))

        xyz_system = np.array(lmp.gather_atoms('x',1,3))

        xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

        xyz_inter_relaxed = [[],[],[]]

        N_atoms = self.N_atoms - self.n_vac

        idx = 0

        for element, xyz_element in enumerate(n_inter):
            for i in range(xyz_element):
                vec = (xyz_system[N_atoms + idx])
                xyz_inter_relaxed[element].append(vec.tolist())
                idx += 1

        lmp.close()

        e0 = self.pe0/(2*self.size**3)
        
        return pe - self.pe0 + self.n_vac*e0 + n_inter[1]*2.121, self.relaxation_volume, xyz_inter_relaxed
    

    def get_octahedral_sites(self):

        oct_sites_0 = np.zeros((3,3))

        k = 0

        for [i,j] in itertools.combinations([0, 1, 2],2):
            oct_sites_0[k,[i,j]] = 0.5
            k += 1
            
        oct_sites_1 = np.ones((3,3))
        k = 0

        for [i,j] in itertools.combinations([0, 1, 2],2):
            oct_sites_1[k,[i,j]] = 0.5
            k += 1

        oct_sites_unit = np.vstack([oct_sites_0, oct_sites_1])

        n_iter = np.clip(self.n_vac, a_min = 1, a_max = None)

        oct_sites = np.vstack([oct_sites_unit + i*0.5 for i in range(n_iter)])

        return np.unique(oct_sites, axis = 0)

    def get_tetrahedral_sites(self):

        tet_sites_0 = np.zeros((12,3))
        k = 0

        for [i,j] in itertools.combinations([0, 1, 2],2):
            tet_sites_0[4*k:4*k+4,[i,j]] = np.array( [[0.5 , 0.25],
                                                [0.25, 0.5],
                                                [0.5 , 0.75],
                                                [0.75, 0.5] ])

            k += 1

        tet_sites_1 = np.ones((12,3))
        k = 0

        for [i,j] in itertools.combinations([0, 1, 2],2):
            tet_sites_1[4*k:4*k+4,[i,j]] = np.array( [[0.5 , 0.25],
                                                [0.25, 0.5],
                                                [0.5 , 0.75],
                                                [0.75, 0.5] ])

            k += 1

        tet_sites_unit = np.vstack([tet_sites_0, tet_sites_1])

        n_iter = np.clip(self.n_vac, a_min = 1, a_max = None)

        tet_sites = np.vstack([tet_sites_unit + i*0.5 for i in range(n_iter)])

        return np.unique(tet_sites, axis = 0)
    
    def get_diagonal_sites(self):

        diag_sites_0 = np.array([ 
                                [0.25, 0.25, 0.25],
                                [0.75, 0.75, 0.75],
                                [0.25, 0.75, 0.75],
                                [0.75, 0.25, 0.25],
                                [0.75, 0.25, 0.75],
                                [0.25, 0.75, 0.25],
                                [0.75, 0.75, 0.25],
                                [0.25, 0.25, 0.75]
                            ])    


        n_iter = np.clip(self.n_vac, a_min = 1, a_max = None)

        diag_sites_unit = np.vstack([diag_sites_0 + i*0.5 for i in range(n_iter)])

        return np.unique(diag_sites_unit, axis = 0)
    
    def get_central_sites(self):

        central_sites = [ (i+1)*np.array([0.5, 0.5, 0.5]) for i in range(self.n_vac)]

        return np.array(central_sites)
    
    def get_all_sites(self):

        sites = {}

        sites['oct'] =  self.defect_pos + self.get_octahedral_sites()

        sites['tet'] = self.defect_pos + self.get_tetrahedral_sites()

        sites['diag'] = self.defect_pos + self.get_diagonal_sites()

        if len(self.get_central_sites()) > 0:
            sites['central'] = self.defect_pos + self.get_central_sites()
        else:
            sites['central'] = []

        return sites
    
    def find_strain(self):

        C11 = 3.201
        C12 = 1.257
        C44 = 1.020

        C = np.array( [
            [C11, C12, C12, 0, 0, 0],
            [C12, C11, C12, 0, 0, 0],
            [C12, C12, C11, 0, 0, 0],
            [0, 0, 0, C44, 0, 0],
            [0, 0, 0, 0, C44, 0],
            [0, 0, 0, 0, 0, C44]
        ])

        conversion = 1.602177e2

        C = conversion*C

        stress = self.stress_voigt*1e-4

        self.strain_voigt = np.linalg.solve(C, stress)

        strain_tensor = np.array( [ 
            [self.strain_voigt[0], self.strain_voigt[3]/2, self.strain_voigt[4]/2],
            [self.strain_voigt[3]/2, self.strain_voigt[1], self.strain_voigt[5]/2],
            [self.strain_voigt[4]/2, self.strain_voigt[5]/2, self.strain_voigt[2]]
        ])

        return strain_tensor

    def minimize_single_intersitial(self, element_idx):

        energy_lst = []

        type_lst = []

        relaxed_lst = []

        available_sites = self.get_all_sites()

        for site_type in available_sites:

            test_site = [[],[],[]]
            
            if len(available_sites[site_type]) > 0:
                test_site[element_idx].append(available_sites[site_type][0])

                energy, relaxed = self.Build_Defect(test_site)

                energy_lst.append(energy)

                type_lst.append(site_type)

                relaxed_lst.append(relaxed)
            
            else:
                energy_lst.append(np.inf)

                type_lst.append(site_type)

        energy_arr = np.array(energy_lst)

        min_energy_idx = np.argmin(energy_arr)

        xyz_init = [[],[],[]]

        xyz_init[element_idx].append(available_sites[type_lst[min_energy_idx]][0])    

        return energy_lst[min_energy_idx], xyz_init, relaxed_lst[min_energy_idx]

    def check_proximity(self,xyz_init, test_site):

        for xyz_element in xyz_init:
            for vec in xyz_element:
                distance = np.linalg.norm(test_site - vec)
                if distance < 0.1:
                    return False
        
        return True
        
    def minimize_add_intersitial(self, element_idx, xyz_init):

        energy_lst = []

        type_lst = []

        idx_lst = []

        relaxed_lst = []

        available_sites = self.get_all_sites()

        for site_type in available_sites:

            for idx, site in enumerate(available_sites[site_type]):

                valid = self.check_proximity(xyz_init, site)

                test_site = copy.deepcopy(xyz_init)

                test_site[element_idx].append(site.tolist())

                if valid:
                    energy, relaxed = self.Build_Defect(test_site)

                else:

                    energy = np.inf

                    relaxed = copy.deepcopy(test_site)

                energy_lst.append(energy)

                type_lst.append(site_type)

                idx_lst.append(idx)

                relaxed_lst.append(relaxed)

        energy_arr = np.array(energy_lst)

        energy_arr = np.nan_to_num(energy_arr)

        min_energy_idx = np.argmin(energy_arr)

        xyz_init_new = copy.deepcopy(xyz_init)

        xyz_init_new[element_idx].append(available_sites[type_lst[min_energy_idx]][idx_lst[min_energy_idx]].tolist())    

        _ = self.Build_Defect(xyz_init_new)

        return energy_lst[min_energy_idx], xyz_init_new, relaxed_lst[min_energy_idx]
