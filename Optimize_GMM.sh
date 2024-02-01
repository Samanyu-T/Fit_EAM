nproc=28
time=11
machine=""

rm -rf ../Output
mkdir ../Output

rm -rf ../Error
mkdir ../Error

for ((i = 0; i < nproc; i++)); do
    
    python "Python_Scripts/Random_Sampling.py" $i $machine $time > "../Output/output.$i.txt" 2> "../Error/error.$i.txt" &

done

wait

echo "Finished Random Sampling!!"

python "Python_Scripts/GMM.py"

for ((i = 0; i < nproc; i++)); do
    
    python "Python_Scripts/Gaussian_Sampling.py" $i $machine $time > "../Output/output.$i.txt" 2> "../Error/error.$i.txt" &

done

wait

echo "Finished Gaussian Sampling!!"

for ((i = 0; i < nproc; i++)); do
    
    python "Python_Scripts/Simplex.py" $i $machine $time > "../Output/output.$i.txt" 2> "../Error/error.$i.txt" &

done

wait

echo "Finished Simplex!!"
