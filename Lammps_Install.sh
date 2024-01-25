git clone -b release https://github.com/lammps/lammps.git Lammps
cd Lammps
mkdir build; cd build
cmake ../cmake
cmake -C ../cmake/presets/basic.cmake ../cmake
cmake -D PKG_REPLICA=yes ../cmake
cmake -D PKG_EXTRA-FIX=yes ../cmake
cmake -D BUILD_SHARED_LIBS=yes ../cmake
cmake -D PKG_PYTHON=yes ../cmake
#Change to your Python exe file - use which python or which python 3
cmake -D PYTHON_EXECUTABLE=/Users/cd8607/anaconda3/bin/python ../cmake
cmake —-build .
make install
make install-python
