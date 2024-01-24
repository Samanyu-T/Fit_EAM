with open('Lammps_Dump/Surface/Orient_111_Depth(1.00).atom', 'r') as file:
    lines = file.readlines()

    print(lines[1:4])