#!/bin/bash
#SBATCH --job-name=ppta_norm_biv_trunc_lg_A_gamma_all_2
#SBATCH --output=/fred/oz002/bgoncharov/logs_pta_gwb_priors/ppta_norm_biv_trunc_lg_A_gamma_all_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=0-5
#SBATCH --mem-per-cpu=2G
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
srun python /home/bgonchar/pta_gwb_priors/run_hyper.py --result "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_to_recycle_20210626.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_hpe_norm_biv_trunc_lg_A_gamma_20211011_2.dat"
