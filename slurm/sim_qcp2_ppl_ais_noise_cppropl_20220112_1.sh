#!/bin/bash
#SBATCH --job-name=sim_qcp2_ppl_ais_noise_20220112
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_qcp2_ais_noise_20220112_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-8
#SBATCH --mem-per-cpu=10G
#SBATCH --tmp=6G
#SBATCH --array=5-249

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/sim_qcp2_ppl_to_recycle_cpl_20220107.dat" --target "/home/bgonchar/pta_gwb_priors/params/sim_qcp2_ppl_to_recycle_cpl_20220107.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/sim_qcp2_ppl_is_noise_20220112_1.dat" --n_grid_iter 1000 --save_iterations $SLURM_ARRAY_TASK_ID
