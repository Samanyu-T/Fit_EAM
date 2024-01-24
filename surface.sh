# Create Test set
proc=8
N_images=6
rm -rf Test_Data
rm -rf Lammps_Dump/Surface/100
rm -rf Lammps_Dump/Surface/110
rm -rf Lammps_Dump/Surface/111
rm -rf Lammps_Scripts

mkdir Lammps_Scripts
mkdir Lammps_Dump/Surface/100
mkdir Lammps_Dump/Surface/110
mkdir Lammps_Dump/Surface/111

mkdir Test_Data
mkdir Test_Data/Test_Graphs

for potfile in Potentials/Selected_Potentials/Potential_3/*.eam.alloy; do


    echo $potfile

    mpiexec -n $proc python Python_Scripts/Edge_Dislocation_fixed.py $potfile
    echo Finished
    mpiexec -n $proc python Python_Scripts/Cylinder_Screw_fixed.py $potfile
    echo Finished
    mpiexec -n $proc python Python_Scripts/Surface_Binding.py $potfile $N_images
    echo Finished

    mpiexec -np $proc lmp -p ${proc}x1 -in Lammps_Scripts/tet_tet_2.neb
    python Python_Scripts/neb.py 'tet_tet' $proc $file

    mpiexec -np $proc lmp -p ${proc}x1 -in Lammps_Scripts/tet_oct.neb
    python Python_Scripts/neb.py 'tet_oct' $proc $file

    for file in ls Lammps_Scripts/surface*; do

        mpiexec -np $proc lmp -p ${proc}x1 -in $file
        python Python_Scripts/neb.py $file $proc

    done

    python Python_Scripts/Test_Graphs.py $potfile 24
done
