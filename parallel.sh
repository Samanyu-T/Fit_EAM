num_proc = 12


file = "Python_Scripts/Fitting_Script.py"

for i in 0 1 2 3 4 5 6 7; do

    python "$file" "$i" &

done

wait

echo "FINISHED !!"

