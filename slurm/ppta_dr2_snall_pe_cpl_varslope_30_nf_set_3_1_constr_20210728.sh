#!/bin/bash
#SBATCH --job-name=ppta_snall_pe_cpl_varsl_30nf_set31_const_1
#SBATCH --output=/fred/oz002/bgoncharov/logs_pta_gwb_priors/ppta_snall_pe_cpl_varslope_30_nf_set_3_1_const_1_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=0-23
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=6G
#SBATCH --array=0

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis_constr.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_pe_common_pl_vargam_30_nf_set_3_1_ephem_0_20210728.dat" --num $SLURM_ARRAY_TASK_ID
