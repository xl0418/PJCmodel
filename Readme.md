jc 2.2.0
jcpp 2.2.0

# Clone, compile and link

## Peregrine
```bash
module load tbb/2020.3-GCCcore-10.2.0
module load CMake
git clone git@github.com:xl0418/PJCmodel.git jc2
cd jc2
mkdir build 
cd build
cmake ..
cmake --build . --target install
```

## Linux
```bash
apt install libtbb-dev   # or some other packet manager if not on Ubuntu/Debian
git clone git@github.com:xl0418/PJCmodel.git jc2
cd jc2
mkdir build 
cd build
cmake ..
cmake --build . --target install
```

You should find `jc` and `jcpp` in `/home/pxxxxx/jc2/bin/`:


```txt
./jc --help

Usage: jc [OPTIONS]... PARAMETER...
Options:
  --help           prints this text and exits
  --version        prints version information and exits
  -v, --verbose    verbose output to console
  --check          check parameter set, no execution
  --torus          use periodic boundary
  --profile        create profile file(s)
  threads          number of threads

Model parameter:
  L                grid size
  L0               optional initial area size, clamped to L
  Linc             optional L increment, defaults to 1
  Lg               area increment L <- L + Linc every Lg years
  ticks            time steps [y]
  mu               1/average life time [1/y], G = mu * L * L
  nu               speciation rate, Nu = nu * G
  s_nu             optional sigma speciation radius, defaults to L/4
  Psi              strength of PJC effect
  Phi	             size of phylogenetic effect
  s_jc             width of PJC effect wrt spatial distance, use '-1' for 'infinite'
  jc_cutoff        max. distance spatial PJC kernel.
  s_disp           width of dispersal kernel, use '-1' for 'infinite'
  disp_cutoff      max. distance dispersal kernel.

Output Parameter:
  log              optional log interval [events] for D and R, defaults to last state
  continue         result file of former simulation to be continued
  file             data file path

Alternative:
  batch            parameter file path, a file that contains multiple parameter sets
```
