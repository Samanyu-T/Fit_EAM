num_proc=$(nproc)

for ((i = 0; i < num_proc; i++)); do

    chmod -x "Python_Scripts/Fitting_Script.py"

    python "Python_Scripts/Fitting_Script.py" "$i" &

done

wait

echo "FINISHED !!"

cd ../