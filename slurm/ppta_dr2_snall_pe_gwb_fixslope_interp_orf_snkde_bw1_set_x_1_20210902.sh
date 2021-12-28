#!/bin/bash
#SBATCH --job-name=ppta_ms_gwb_iorf_fsl_snkde_bw1_x1
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ppta_ms_gwb_iorf_fsl_snkde_bw1_set_x1_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=2-23
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
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_pe_gwb_fixslope_interp_orf_snkde_bw1_set_x_1_20210902.dat" --num $SLURM_ARRAY_TASK_ID
