#!/bin/bash
#SBATCH --job-name=sim_factorized_drop
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_rnqcp_220208_073559_factorized_drop_%A_%a.out
#SBATCH --ntasks=2
#SBATCH --time=0-5
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=2G
#SBATCH --array=0-0

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/factorized_all_sim_rnqcp_220208_073559_pe_cpl_20220406.dat" --drop 1 --num $SLURM_ARRAY_TASK_ID
