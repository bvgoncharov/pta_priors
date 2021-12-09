#!/bin/bash
#SBATCH --job-name=ppta_cpl_deltaf_prod_lg_A
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ppta_cpl_deltaf_prod_lg_A_%A_%a.out
#SBATCH --ntasks=8
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
srun python /home/bgonchar/pta_gwb_priors/run_hyper.py --result "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpl_fixgam_30_nf_to_recycle_20210811.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snunif_hpe_cpl_lg_A_delf_20210820.dat"
