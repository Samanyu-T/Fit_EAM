module purge
module load rhel8/default-icl
module load intel/mkl
module load fftw

git clone -b release https://github.com/lammps/lammps.git lammps
cd Lammps
mkdir build; cd build
cmake ../cmake
cmake -C ../cmake/presets/basic.cmake ../cmake
cmake -D PKG_REPLICA=yes ../cmake
cmake -D PKG_EXTRA-FIX=yes ../cmake
cmake -D BUILD_SHARED_LIBS=yes ../cmake
cmake -D PKG_PYTHON=yes ../cmake
#Change to your Python exe file - use which python or which python 3
cmake -D PYTHON_EXECUTABLE=($which python) ../cmake
cmake —-build .
make install
make install-python



echo '' >> ~/.bashrc

export LD_LIBRARY_PATH=~/Lammps/src/liblammps_intel_cpu_intelmpi.so:$LD_LIBRARY_PATH


module purge; module load rhel8/default-icl; module load intel/mkl; module load fftw


make yes-basic; make yes-extra-fix; make yes-replica;  make yes-python; 
make -j 8 mode=shared mpi
make install-python
export LD_LIBRARY_PATH=$HOME/lammps/src:$LD_LIBRimpor ARY_PATH 
export LD_LIBRARY_PATH=$HOME/.conda/envs/pylammps/lib:$LD_LIBRARY_PATH 
export PATH=$HOME/lammps/src/:$PATH

