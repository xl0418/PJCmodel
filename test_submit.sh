#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=gelifes

module load tbb
datafile=/data/$USER/jc2/c1000_16.m


"${HOME}/jc2/jc" --torus -v --profile L=1000 ticks=100000 mu=0.01 nu=1e-5 Psi=0.5 Phi=1e-5 s_jc=2 jc_cutoff=4 s_disp=10 disp_cutoff=20 file$
