#!/bin/bash
#SBATCH --job-name=ppta_to_rcl_cpfa
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ppta_to_rcl_cpfa_20220513_%A_%a.out
#SBATCH --ntasks=64
#SBATCH --time=0-2
#SBATCH --mem-per-cpu=1G
#SBATCH --tmp=4G
#SBATCH --array=8,13

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpfa_to_recycle_20220513.dat" --num $SLURM_ARRAY_TASK_ID
