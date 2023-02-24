#!/bin/bash
#SBATCH --partition=gpu          
#SBATCH --nodes=1                # Max is 1
#SBATCH --ntasks=16              # Max is 16 (1/8 of 2x 64 AMD EPYC CPUs)
#SBATCH --cpus-per-task=2        # Max is 2 (Clustered Multithreading is on)
#SBATCH --gres=gpu:1             # Max is 1 (1 single A100)
#SBATCH --time=4:00:00           # Max is 4 hours


module purge > /dev/null 2>&1 

module load conda/2022.9
module load cuda/11.7.0

export MYEXE='test.py'
export OMP_PROC_BIND=true 
export OMP_NUM_THREADS=5

#source ~/.bashrc
conda activate VINT 
#./${MYEXE} 2>&1 | tee out.${SLURM_JOBID}
python ${MYEXE} 2>&1 | tee out.${SLURM_JOBID}
