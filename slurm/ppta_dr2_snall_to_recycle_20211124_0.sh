#!/bin/bash
#SBATCH --job-name=ppta_to_rcl_h
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ppta_to_rcl_%A_%a.out
#SBATCH --ntasks=128
#SBATCH --time=0-5
#SBATCH --mem-per-cpu=1G
#SBATCH --tmp=4G
#SBATCH --array=1,11,12,13,24

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_to_recycle_20211124.dat" --num $SLURM_ARRAY_TASK_ID
