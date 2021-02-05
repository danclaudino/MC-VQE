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

A simple test (4 chromophores, 2 virtual QPUs) can be run with `ctest`. Be aware that this requires TNQVM, ExaTN, MPI to be installed in order to be executed, but should be able to compile even in their absence. 

Running on laptop (8 CPU cores):
```bash
OMP_PLACES=cores OMP_NUM_THREADS=2 OMP_DYNAMIC=false mpiexec -n 4 ./mc_vqe_example --n-virtual-qpus 2 --n-chromophores 18 --n-states 1 --opt-maxiter 1 --n-cycles 2 --exatn-log-level 2 --double-depth true
```

Running on Summit:
```bash
OMP_PLACES=cores OMP_NUM_THREADS=7 OMP_DYNAMIC=false jsrun -n 12 -r 6 -a 1 -c 7 -g 1 -brs ./mc_vqe_example --n-virtual-qpus 3 --n-chromophores 18 --n-states 1 --opt-maxiter 1 --n-cycles 2 --exatn-log-level 2 --double-depth true
```

The following can be added to the command line to control the execution:
* `--n-chromophores`: number of chromophores/qubits
* `--n-virtual-qpus`: number of virtual QPUs (defaults to 1)
* `--exatn-log-level`: controls the level of printing/logging in XACC all the way through TNQVM and ExaTN (defaults to 0)
* `--mcvqe-log-level`: controls the level of printing/logging from the MC-VQE algorithm, such as state energies, state preparation circuits, etc. (defaults to 0 (prints nothing))
* `--print-tnqvm-log`: controls whether to print TNQVM logging (defaults to `false`)
* `--exatn-buffer-size`: ExaTN memomory buffer size in GiB (defaults to 4)
* `--opt-maxiter`: maximum number of optimization iterations (defaults to 1)
* `--n-states`: number of states to consider in the MC-VQE optimization. Needs to be less or equal than the number of chromophores (defaults to 1)
* `--n-cycles`: number of times the application is executed to probe performance (defaults to 1)
* `--accelerator`: name of the accelerator (defaults to `tnqvm` using the `exatn` visitor)
* `--data-path`: string with the directory with the data file (defaults to `{CMAKE_SOURCE_DIR}/examples` where the file selected depends on the number of chromophores)
* `--double-depth`: compute expectation values via conjugate tensor network,  double-depth circuit (defaults to `false`)
* `--max-qubit`: number of qubits above which TNQVM turns `exp-val-by-conjugate` on (defaults to 8)
