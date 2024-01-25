nproc=28

for ((i = 0; i < nproc; i++)); do
    
    python "Python_Scripts/Optimize.py" $i > "../output.$i.txt" 2> "../error.$i.txt" &

done

wait

echo "Finished!!"