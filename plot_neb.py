import matplotlib.pyplot as plt
import numpy as np
import os
import glob 
from scipy.interpolate import interp1d, CubicSpline

# Graph for Surface Neb

surface_folders = [os.path.join("../Test_Data/Surface", folder)
                   for folder in os.listdir("../Test_Data/Surface")
                   if os.path.isdir(os.path.join("../Test_Data/Surface", folder))]

fig, axs = plt.subplots(1, len(surface_folders), figsize = (20,5))

for i, surface_orient in enumerate(surface_folders):
    data_lst = []
    for file in glob.glob('%s/neb_split*.txt' % surface_orient):
        data = np.loadtxt(file)
        data_lst.append(data)

    plot_data = np.vstack(data_lst)
    idx = np.argsort(plot_data[:,0])

    plot_data[:,1] -= plot_data[:,1].min()

    x_sct = plot_data[idx, 0]
    y_sct = plot_data[idx, 1]

    cs = interp1d(x_sct, y_sct)
    x_plt = np.linspace(x_sct.min(), x_sct.max(), 1000)
    y_plt = cs(x_plt)
    
    axs[i].scatter(x_sct, y_sct)
    axs[i].plot(x_plt,y_plt)
    axs[i].set_xlabel('Depth of He atom /A')
    axs[i].set_ylabel('Formation Energy /eV')
    axs[i].set_title('Surface Orientation: %s' % os.path.basename(surface_orient))

fig.suptitle('Formation Energy of an Interstitial Helium atom on the Surface of a Tungsten', fontsize=18)
plt.show()
# Graph for Bulk Neb

bulk_folders = [os.path.join("../Test_Data/Bulk", folder)
                   for folder in os.listdir("../Test_Data/Bulk")
                   if os.path.isdir(os.path.join("../Test_Data/Bulk", folder))]

fig, axs = plt.subplots(1, len(bulk_folders), figsize = (20,5))

print(bulk_folders)
for i, folder in enumerate(bulk_folders):
    data_lst = []
    for file in glob.glob('%s/*.txt' % folder):
        print(file)
        data = np.loadtxt(file)
        data_lst.append(data)


    plot_data = np.vstack(data_lst)
    plot_data[:,1] -= plot_data[:,1].min()

    cs = CubicSpline(plot_data[:,0], plot_data[:,1])
    x_plt = np.linspace(plot_data[:,0].min(), plot_data[:,0].max(), 1000)
    y_plt = cs(x_plt)
    axs[i].plot(x_plt,y_plt)

    axs[i].scatter(plot_data[:,0], plot_data[:,1])
    axs[i].set_xlabel('Depth of He atom /A')
    axs[i].set_ylabel('Formation Energy /eV')
    axs[i].set_title('Neb of: %s' % os.path.basename(folder))
plt.show()
