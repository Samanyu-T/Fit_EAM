import numpy as np
from Lmp_PDefect import Point_Defect
import time

lmp = Point_Defect(size=7, n_vac=0, potfile='Potentials/WHHe_test.eam.alloy', surface=False, depth=0, machine='')

t1 = time.perf_counter()
pe, rvol , pos= lmp.Build_Defect([[], [], [[0.25,0.5, 0]]])
t2 = time.perf_counter()
print(pe, t2 - t1)

t1 = time.perf_counter()
pe, rvol , pos= lmp.Find_Min_Config('V0H0He1', 3)
t2 = time.perf_counter()
print(pe, t2 - t1)

t1 = time.perf_counter()
pe, rvol , pos= lmp.Find_Min_Config('V0H0He2', 3)
t2 = time.perf_counter()
print(pe, t2 - t1)

t1 = time.perf_counter()
pe, rvol , pos= lmp.Find_Min_Config('V0H0He3', 3)
t2 = time.perf_counter()
print(pe, t2 - t1)

t1 = time.perf_counter()
pe, rvol , pos= lmp.Find_Min_Config('V0H0He4', 3)
t2 = time.perf_counter()
print(pe, t2 - t1)