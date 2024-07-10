#!/bin/bash
#SBATCH --job-name=sim_qcp2_ppl_wnpe
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_qcp2_ppl_wnpe_20220104_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=15:00
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=3G
#SBATCH --array=10,11,24

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/sim_qcp2_ppl_wnpe_20220104.dat" --num $SLURM_ARRAY_TASK_ID
