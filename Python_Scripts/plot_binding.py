import matplotlib.pyplot as plt
from Handle_Dictionaries import find_binding, data_dict
import numpy as np
import os 
import json

with open('formations.json', 'r') as file:
    lmp = json.load(file)

with open('refs_formations.json', 'r') as file:
    dft = json.load(file)

data =  data_dict(dft, lmp, 2, 6, 6)

he_vac_binding = []
x_data = []

he_vac_binding.append(find_binding(data, [0, 0, 1], [0, 0, 1], [0,0,1]))
x_data.append(np.arange(1, len(he_vac_binding[-1]) + 1))

for i in range(1,3):
    he_vac_binding.append(find_binding(data, [i, 0, 0], [0, 0, 1], [0,0,1]))
    x_data.append(np.arange(0, len(he_vac_binding[-1])))

for i in range(len(he_vac_binding)):
    plt.scatter(x_data[i],he_vac_binding[i])
    plt.plot(x_data[i],he_vac_binding[i] ,label='n_vac: %d' % i)

plt.xlabel('Number of Helium atoms already present in the cluster')
plt.ylabel('Binding Energy/ eV')
plt.title('Binding of Interstitial Helium atoms to Helium Clusters in Bulk Tungsten')
plt.legend()
plt.show()


h_heint_binding = []
x_data = []

for i in range(4):
    h_heint_binding.append(find_binding(data, [0, i, 1], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(1, len(h_heint_binding[-1]) + 1))


for i in range(len(h_heint_binding)):
    plt.scatter(x_data[i],h_heint_binding[i])
    plt.plot(x_data[i],h_heint_binding[i] ,label='n_h: %d' % i)

plt.xlabel('Number of Helium atoms already present in the cluster')
plt.ylabel('Binding Energy/ eV')
plt.title('Binding of Interstitial Helium atoms to Helium Clusters in Bulk Tungsten')
plt.legend()
plt.show()


h_hevac_binding = []
x_data = []

for i in range(4):
    h_hevac_binding.append(find_binding(data, [1, i, 0], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(0, len(h_hevac_binding[-1])))

for i in range(len(h_hevac_binding)):
    plt.scatter(x_data[i],h_hevac_binding[i])
    plt.plot(x_data[i],h_hevac_binding[i] ,label='n_h: %d' % i)


plt.xlabel('Number of Helium atoms already present in the cluster')
plt.ylabel('Binding Energy/ eV')
plt.title('Binding of Interstitial Helium atoms to Helium Clusters in Bulk Tungsten')
plt.legend()
plt.show()


h_hedivac_binding = []
x_data = []

for i in range(4):
    h_hedivac_binding.append(find_binding(data, [2, i, 0], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(0, len(h_hedivac_binding[-1])))


for i in range(len(h_hedivac_binding)):
    plt.scatter(x_data[i],h_hedivac_binding[i])
    plt.plot(x_data[i],h_hedivac_binding[i] ,label='n_h: %d' % i)

plt.xlabel('Number of Helium atoms already present in the cluster')
plt.ylabel('Binding Energy/ eV')
plt.title('Binding of Interstitial Helium atoms to Helium Clusters in Bulk Tungsten')
plt.legend()
plt.show()
