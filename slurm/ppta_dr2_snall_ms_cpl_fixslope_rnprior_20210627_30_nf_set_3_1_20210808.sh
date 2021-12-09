#!/bin/bash
#SBATCH --job-name=p_ms_cpl_fixsl_30nf_rnp20210627_31_bmc
#SBATCH --output=/fred/oz031/correlated_noise_logs/ppta_snall_ms_cpl_fixslope_rnprior_20210627_30_nf_set_3_1_bmc_%A_%a.out
#SBATCH --ntasks=64
#SBATCH --time=0-8
#SBATCH --mem-per-cpu=4G
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
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_ms_common_pl_fixgam_rnprior_20210627_30_nf_set_3_1_20210808.dat" --num $SLURM_ARRAY_TASK_ID
