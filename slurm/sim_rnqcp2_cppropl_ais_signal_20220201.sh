#!/bin/bash
#SBATCH --job-name=sim_rnqcp2_ais_signal_20220201
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_rnqcp2_ais_signal_20220201_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=3G
#SBATCH --array=1-298

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/sim_rnqcp2_rndpl_cp_to_recycle_20220115.dat" --target "/home/bgonchar/pta_gwb_priors/params/sim_rnqcp2_rndpl_cp_to_recycle_20220115.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/sim_rnqcp2_is_signal_20220201.dat" --save_iterations $SLURM_ARRAY_TASK_ID
