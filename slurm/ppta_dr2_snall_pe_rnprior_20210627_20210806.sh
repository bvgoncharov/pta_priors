#!/bin/bash
#SBATCH --job-name=p_pe_rnp20210627_bmc
#SBATCH --output=/fred/oz031/correlated_noise_logs/ppta_snall_pe_rnprior_20210627_bmc_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=1-21
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=4G
#SBATCH --array=19

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_pe_rnprior_20210627_20210806.dat" --num $SLURM_ARRAY_TASK_ID
