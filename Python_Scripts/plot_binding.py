import matplotlib.pyplot as plt
from Handle_Dictionaries import find_binding, data_dict
import numpy as np
import os 
import json

dft_binding = {}

with open('formations.json', 'r') as file:
    eam = json.load(file)

with open('refs_formations.json', 'r') as file:
    dft_raw = json.load(file)

color = ['red', 'blue', 'green', 'orange', 'cyan', 'pink', 'yellow']

dft =  data_dict(dft_raw, eam, 2, 6, 6)

y_data = []
x_data = []

y_data.append(find_binding(dft, [0, 0, 1], [0, 0, 1], [0,0,1]))
x_data.append(np.arange(1, len(y_data[-1]) + 1))

for i in range(1, 3):
    y_data.append(find_binding(dft, [i, 0, 0], [0, 0, 1], [0,0,1]))
    x_data.append(np.arange(0, len(y_data[-1])))

for i in range(len(y_data)):
    plt.scatter(x_data[i], y_data[i], color=color[i])
    plt.plot(x_data[i], y_data[i], label='dft V%d' % i, linestyle='-', color=color[i])

    
y_data = []
x_data = []

y_data.append(find_binding(eam, [0, 0, 1], [0, 0, 1], [0,0,1]))
x_data.append(np.arange(1, len(y_data[-1]) + 1))

for i in range(1, 3):
    y_data.append(find_binding(eam, [i, 0, 0], [0, 0, 1], [0,0,1]))
    x_data.append(np.arange(0, len(y_data[-1])))

for i in range(len(y_data)):
    plt.scatter(x_data[i], y_data[i], color=color[i])
    plt.plot(x_data[i], y_data[i], label='eam V%d' % i, linestyle='--', color=color[i])



plt.xlabel('Number of Helium atoms already present in the cluster')
plt.ylabel('Binding Energy/ eV')
plt.title('Binding of Interstitial Helium atoms to Helium Clusters')
plt.legend()
plt.show()


y_data = []
x_data = []

for i in range(4):
    y_data.append(find_binding(dft, [0, i, 1], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(1, len(y_data[-1]) + 1))


for i in range(len(y_data)):
    plt.scatter(x_data[i], y_data[i], color=color[i])
    plt.plot(x_data[i], y_data[i], label='dft H%d' % i, linestyle='-', color=color[i])


y_data = []
x_data = []

for i in range(4):
    y_data.append(find_binding(eam, [0, i, 1], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(1, len(y_data[-1]) + 1))


for i in range(len(y_data)):
    plt.scatter(x_data[i], y_data[i], color=color[i])
    plt.plot(x_data[i], y_data[i], label='eam H%d' % i, linestyle='--', color=color[i])

plt.xlabel('Number of Helium atoms already present in the cluster')
plt.ylabel('Binding Energy/ eV')
plt.title('Binding of Interstitial Hydrogen atoms to Interstital Helium Hydrogen Clusters')
plt.legend()
plt.show()


y_data = []
x_data = []

for i in range(4):
    y_data.append(find_binding(dft, [1, i, 0], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(0, len(y_data[-1])))

for i in range(len(y_data)):
    plt.scatter(x_data[i], y_data[i], color=color[i])
    plt.plot(x_data[i], y_data[i], label='dft H%d' % i, linestyle='-', color=color[i])


y_data = []
x_data = []

for i in range(4):
    y_data.append(find_binding(eam, [1, i, 0], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(0, len(y_data[-1])))

for i in range(len(y_data)):
    plt.scatter(x_data[i], y_data[i], color=color[i])
    plt.plot(x_data[i], y_data[i], label='eam H%d' % i, linestyle='--', color=color[i])

plt.xlabel('Number of Helium atoms already present in the cluster')
plt.ylabel('Binding Energy/ eV')
plt.title('Binding of Interstitial Hydrogen atoms to Vacancy Helium Hydrogen Clusters')
plt.legend()
plt.show()



y_data = []
x_data = []

for i in range(4):
    y_data.append(find_binding(dft, [1, i, 0], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(0, len(y_data[-1])))

for i in range(len(y_data)):
    plt.scatter(x_data[i], y_data[i], color=color[i])
    plt.plot(x_data[i], y_data[i], label='dft H%d' % i, linestyle='-', color=color[i])


y_data = []
x_data = []

for i in range(4):
    y_data.append(find_binding(eam, [1, i, 0], [0, 1, 0], [0,0,1]))
    x_data.append(np.arange(0, len(y_data[-1])))

for i in range(len(y_data)):
    plt.scatter(x_data[i], y_data[i], color=color[i])
    plt.plot(x_data[i], y_data[i], label='eam H%d' % i, linestyle='--', color=color[i])

plt.xlabel('Number of Helium atoms already present in the cluster')
plt.ylabel('Binding Energy/ eV')
plt.title('Binding of Interstitial Hydrogen atoms to a Di-vacancy Helium Hydrogen Clusters')
plt.legend()
plt.show()
