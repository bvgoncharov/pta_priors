#!/bin/bash
#SBATCH --job-name=sim_rngwb_wnpe
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_rngwb_wnpe_20220128_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-2
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=3G
#SBATCH --array=0-25

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/sim_rngwb_wnpe_20220128.dat" --num $SLURM_ARRAY_TASK_ID
