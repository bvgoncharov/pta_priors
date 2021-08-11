#!/bin/bash
#SBATCH --job-name=ppta_snall_pe_cpl_varsl_30nf_sp_const_2
#SBATCH --output=/fred/oz002/bgoncharov/logs_pta_gwb_priors/ppta_snall_pe_cpl_varslope_30_nf_sp_const_2_bmc_%A_%a.out
#SBATCH --ntasks=64
#SBATCH --time=0-5
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=8G
#SBATCH --array=0,13,20

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis_constr.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_pe_common_pl_vargam_30_nf_single_ephem_0_20210809.dat" --num $SLURM_ARRAY_TASK_ID
