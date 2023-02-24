 #!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --nodes=1 
#SBATCH --ntasks=4 
#SBATCH --cpus-per-task=1 
#SBATCH --gres=gpu:1 
#SBATCH --time=4:00:00

module purge > /dev/null 2>&1 
module load compilers/nvhpc-21.9-mpi 
export MYEXE=hello_world

mpicc -o hello_world hello_world.c
mpirun -np 2 --report-bindings ${MYEXE} 2>&1 | tee out.${SLURM_JOBID}
