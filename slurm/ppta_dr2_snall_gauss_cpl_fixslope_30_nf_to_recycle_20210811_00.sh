#!/bin/bash
#SBATCH --job-name=ppta_rng_cpl_to_rcl
#SBATCH --output=/fred/oz002/bgoncharov/logs_pta_gwb_priors/ppta_rng_cpl_to_rcl_%A_%a.out
#SBATCH --ntasks=128
#SBATCH --time=0-5
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=4G
#SBATCH --array=0

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_rnprior_20210806_cpl_fixgam_30_nf_to_recycle_20210811.dat" --num $SLURM_ARRAY_TASK_ID
