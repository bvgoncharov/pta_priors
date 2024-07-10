#!/bin/bash
#SBATCH --job-name=sim_rngwb_ais_noise_20220201
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_rngwb_ais_noise_20220201_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-2
#SBATCH --mem-per-cpu=5G
#SBATCH --tmp=4G
#SBATCH --array=5-249

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/sim_rngwb_to_recycle_20220131.dat" --target "/home/bgonchar/pta_gwb_priors/params/sim_rngwb_to_recycle_20220131.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/sim_rngwb_is_noise_20220201.dat" --n_grid_iter 1000 --save_iterations $SLURM_ARRAY_TASK_ID
