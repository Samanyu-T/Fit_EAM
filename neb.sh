# Create Test set
proc=9
lmp_exec=lmp
machine=""
rm -rf ../Neb_Dump
rm -rf ../Neb_Scripts
rm -rf ../Test_Data

for potfile in Potentials/Selected_Potentials/Potential_3/*.eam.alloy; do

    mpiexec -n 8 python Python_Scripts/Neb_Surface.py $potfile $machine
    
    for simple_neb_script in ../Neb_Scripts/Surface/*/simple.neb; do
        mpiexec -n $proc $lmp_exec -p "$proc"x1 -in $simple_neb_script 
        python Python_Scripts/read_neb_log.py $simple_neb_script $proc
    done

    mpiexec -n 8 python Python_Scripts/Min_Neb_Images.py $potfile $machine

    python Python_Scripts/Neb_unique.py $potfile

    for fine_neb_script in ../Neb_Scripts/Surface/*/fine*.neb; do
        mpiexec -n $proc $lmp_exec -p "$proc"x1 -in $fine_neb_script
        python Python_Scripts/read_neb_log.py $fine_neb_script $proc
    done


    mpiexec -n 8 python Python_Scripts/Neb_Bulk.py $potfile $machine

    for simple_neb_script in ../Neb_Scripts/Bulk/*/simple.neb; do
        mpiexec -n $proc $lmp_exec -p "$proc"x1 -in $simple_neb_script
        python Python_Scripts/read_neb_log_bulk.py $simple_neb_script $proc
    done

    
done


