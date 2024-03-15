import numpy as np
import os 
import glob
import sys

def create_neb_script(init, final ,potfile, save_folder, neb_script_folder, idx):
    txt = '''
units metal 

atom_style atomic

atom_modify map array sort 0 0.0

read_data %s

mass 1 183.84

mass 2 1.00784

mass 3 4.002602

pair_style eam/alloy

pair_coeff * * %s W H He

thermo 10

run 0

fix 1 all neb 1e-4

timestep 1e-3

min_style quickmin

thermo 100 

variable i equal part

neb 10e-8 10e-10 5000 5000 100 final %s

write_dump all custom %s/neb.$i.atom id type x y z ''' % (init, potfile, final, save_folder)

    with open('%s/fine_%d.neb' % (neb_script_folder, idx), 'w') as file:
        file.write(txt)


def find_unique_images(orient, potfile):
    
    he_lst = []

    file_pattern = glob.glob('../Neb_Dump/Surface/%s/Min_Neb_Images/neb.*.atom' % orient)

    for i in range(len(file_pattern)):

        filename = '../Neb_Dump/Surface/%s/Min_Neb_Images/neb.%d.atom' % (orient, i)
        with open(filename, 'r') as file:
            file.readline()
            file.readline()
            file.readline()
            N = int(file.readline())

        data = np.loadtxt(filename, skiprows=9)
        he_idx = np.where(data[:,0] == N)[0]
        # print(he_idx)x
        xyz = data[he_idx, -3:].flatten()

        print(filename,xyz)

        add = True

        for xyz_lst in he_lst:
            if np.linalg.norm(xyz - xyz_lst)<1 or xyz[-1] <= xyz_lst[-1]:
                add = False
                break
        
        if add:

            dir = os.path.dirname(os.path.dirname(filename))
            folder = os.path.join(dir, 'Neb_Init')

            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            with open(filename, 'r') as file:
                lines = file.readlines()

            with open(os.path.join(folder, 'init.%d.atom' % len(he_lst)), 'w') as file:
                file.write(lines[3])
                file.writelines(lines[9:])
            
            sep = '.'

            with open('%s.data' % sep.join(filename.split('.')[:-1]), 'r') as file:
                lines = file.readlines()

            print('%s.data' % sep.join(filename.split('.')[:-1]))

            with open(os.path.join(folder, 'init.%d.data' % len(he_lst)), 'w') as file:
                file.writelines(lines)

            he_lst.append(xyz)

        if len(he_lst) > 1:

            save_folder = '../Neb_Dump/Surface/%s/Neb_Images_%d' % (orient, len(he_lst) - 1)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)

            create_neb_script(init=os.path.join(folder, 'init.%d.data' % (len(he_lst) - 2)),
                              final=os.path.join(folder, 'init.%d.atom' % (len(he_lst) - 1)),
                              potfile=potfile,
                              save_folder=save_folder,
                              neb_script_folder='../Neb_Scripts/Surface/%s' % orient,
                              idx=len(he_lst) - 1)
    print(he_lst)
    
if __name__ == '__main__':

    potfile='Potentials/WHHe_test.eam.alloy'

    if len(sys.argv) > 1:
        potfile=sys.argv[1]

    print(potfile)
    
    find_unique_images('100', potfile)
    find_unique_images('110', potfile)
    find_unique_images('111', potfile)