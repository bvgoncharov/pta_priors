#!/bin/bash
#SBATCH --job-name=ng_ais_rn_all_cppropl_20220620
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ng12_ais_rn_all_cppropl_20220620_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-3
#SBATCH --mem-per-cpu=7G
#SBATCH --tmp=6G
#SBATCH --array=5-249

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

export OMP_NUM_THREADS=1

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/ng12.5_snall_cpfg_to_recycle_20220619.dat" --target "/home/bgonchar/pta_gwb_priors/params/ng12.5_snall_cpfg_to_recycle_20220619.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ng12.5/ng12.5_is_rn_set_all_cppropl_20220620.dat" --n_grid_iter 1000 --save_iterations $SLURM_ARRAY_TASK_ID
