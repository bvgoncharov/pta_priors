#!/bin/bash
#SBATCH --job-name=ppta_ms_snkde_bw0.5_spsr
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ppta_ms_snkde_bw0.5_spsr_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=0-23
#SBATCH --mem-per-cpu=4G
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
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_ms_snkde_bw0.5_singlepsr_20210904.dat" --num $SLURM_ARRAY_TASK_ID
