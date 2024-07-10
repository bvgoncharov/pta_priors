#!/bin/bash
#SBATCH --job-name=p_ms_cpl_fixsl_rng_xg1
#SBATCH --output=/fred/oz031/correlated_noise_logs/ppta_snall_ms_cpl_fixslope_rntgauss_30_nf_set_xg1_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=2-21
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --array=0

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_ms_common_pl_fixgam_rntgauss_30_nf_set_xg1_20220406.dat" --num $SLURM_ARRAY_TASK_ID
