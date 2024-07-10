#!/bin/bash
#SBATCH --job-name=sim_qrn2_ais_noise_20220105
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_qrn2_ais_noise_20220105_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-9
#SBATCH --mem-per-cpu=10G
#SBATCH --tmp=6G
#SBATCH --array=1-249

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/sim_qcp2_wnpe_20220104.dat" --target "/home/bgonchar/pta_gwb_priors/params/sim_qcp2_wnpe_20220104.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/sim_qrn2_is_noise_20220105.dat" --n_grid_iter 1000 --save_iterations $SLURM_ARRAY_TASK_ID
