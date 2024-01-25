nproc=28
rm -rf ../Output
mkdir ../Output

rm -rf ../Error
mkdir ../Error

for ((i = 0; i < nproc; i++)); do
    
    python "Python_Scripts/Optimize.py" $i > "../Output/output.$i.txt" 2> "../Error/error.$i.txt" &

done

wait

echo "Finished!!"