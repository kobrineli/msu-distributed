rm -rf run && mkdir run && cd run
mpirun -v -np 11 --enable-recovery --with-ft=ulfm --oversubscribe ../fdtd
cd ../
