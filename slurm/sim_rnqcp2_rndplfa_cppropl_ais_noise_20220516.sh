#!/bin/bash
#SBATCH --job-name=sim_rnqcp2_rndplfa_ais_noise_20220516
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_rnqcp2_ais_noise_rndplfa_cpprop_20220516_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=04:00
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=3G
#SBATCH --array=5-249

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/sim_rnqcp2_rndpl_cpfa_to_recycle_20220515.dat" --target "/home/bgonchar/pta_gwb_priors/params/sim_rnqcp2_rndpl_cpfa_to_recycle_20220515.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/sim_rnqcp2_ppl_cpfa_is_noise_20220516.dat" --n_grid_iter 1000 --save_iterations $SLURM_ARRAY_TASK_ID
