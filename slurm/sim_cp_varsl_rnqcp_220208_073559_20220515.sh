#!/bin/bash
#SBATCH --job-name=sim_cpl_varg
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_cpl_varg_rnqcp_220208_073559_20220515_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-1
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=3G
#SBATCH --array=0-25

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/sim_rnqcp2_rndpl_cpfa_to_recycle_20220515.dat" --num $SLURM_ARRAY_TASK_ID
