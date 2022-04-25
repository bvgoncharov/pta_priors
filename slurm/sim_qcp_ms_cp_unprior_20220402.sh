#!/bin/bash
#SBATCH --job-name=sim_qcp_ms_cp_u
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_qcp_ms_cp_unpr_20220402_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=0-8
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=4G
#SBATCH --array=0-0

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/sim_qcp_ms_cp_unpr_20220402.dat" --num $SLURM_ARRAY_TASK_ID
