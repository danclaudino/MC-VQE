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