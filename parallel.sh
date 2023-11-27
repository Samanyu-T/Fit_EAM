
for i in {0..3}; do

    chmod -x "Python_Scripts/Fitting_Script.py"

    python "Python_Scripts/Fitting_Script.py" "$i" &

done

wait

echo "FINISHED !!"

cd ../