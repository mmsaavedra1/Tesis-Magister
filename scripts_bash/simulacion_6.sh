#!/bin/bash
#SBATCH --time=7-00:00:00              # specify time limit (D-HH:MM)
#SBATCH --nodes=1                       # do not change
#SBATCH --ntasks=1                  
#SBATCH --mem-per-cpu=80G               # specify total memory
#SBATCH --job-name=simulacion_6      # do not change
#SBATCH --output=~/Desktop/Produccion-Tesis/Output/simulacion_6.out         # do not change

module load python/3.9.6 gurobi/9.5.2 mpi4py
source ~/Desktop/env_gurobi/bin/activate

cd ~/Desktop/Produccion-Tesis
python ~/Desktop/Produccion-Tesis/simulacion6.py
