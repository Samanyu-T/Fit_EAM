import numpy as np
from Lmp_PDefect import Point_Defect

lmp = Point_Defect(size=7, n_vac=0, potfile='Potentials/WHHe_test.eam.alloy', surface=False, depth=0, machine='')

pe, _ = lmp.Build_Defect([[], [], [[0.25,0.5, 0]]])
