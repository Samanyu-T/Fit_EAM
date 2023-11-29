nproc=16

for ((i = 0; i < nproc; i++)); do
    
    python "Python_Scripts/Sample.py" $i > /dev/null 2> "error.$i.txt" &

done

wait

echo "Finished!!"