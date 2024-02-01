from Lmp_PDefect import Point_Defect


inst = Point_Defect(7, 0, potfile='Potentials/WHHe_test.eam.alloy')

tet = inst.get_tetrahedral_sites()

_ = inst.Build_Defect([[],[tet[0]],[]])

print(inst.vol, inst.strain_tensor)