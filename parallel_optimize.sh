nproc=2

for ((i = 0; i < nproc; i++)); do
    
    python "Python_Scripts/Optimize.py" $i > /dev/null 2> "error.$i.txt" &

done
