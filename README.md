# A plugin for the MC-VQE algorithm

## Dependencies
```
Compiler (C++17): GNU 8+
CMake 3.12+
```

## Build instructions
First install [XACC](https://github.com/eclipse/xacc). You can now proceed with the following commands, paying attention to the XACC installation directory if you change it.
```bash
$ git clone https://code.ornl.gov/6d3/mc-vqe.git
$ mkdir build; cd build
$ cmake .. (-DXACC_DIR=/path/to/xacc/install)
$ make install
```

For the time being (until vqe.cpp is merged into xacc/master), copy `vqe/vqe.cpp` to `xacc/quantum/plugins/algorithms/vqe/` and rebuild XACC.

Running on laptop (8 CPU cores):
```bash
OMP_PLACES=cores OMP_NUM_THREADS=2 OMP_DYNAMIC=false mpiexec -n 4 ./mc_vqe_example --n-virtual-qpus 2 --n-chromophores 18 --n-states 1 --opt-maxiter 1 --n-cycles 2 --exatn-log-level 2 --double-depth true
```

Running on Summit:
```bash
OMP_PLACES=cores OMP_NUM_THREADS=7 OMP_DYNAMIC=false jsrun -n 12 -r 6 -a 1 -c 7 -g 1 -brs ./mc_vqe_example --n-virtual-qpus 3 --n-chromophores 18 --n-states 1 --opt-maxiter 1 --n-cycles 2 --exatn-log-level 2 --double-depth true
```
