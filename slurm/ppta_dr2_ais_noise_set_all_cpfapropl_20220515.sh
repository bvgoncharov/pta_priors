#!/bin/bash
#SBATCH --job-name=ppta_ais_cpfa_all_cppropl_20220515
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ppta_ais_cpfa_all_cppropl_20220515_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-6
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=6G
#SBATCH --array=2-499

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

export OMP_NUM_THREADS=1

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpfa_to_recycle_20220513.dat" --target "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpfa_to_recycle_20220513.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_is_rn_set_all_cpfapropl_20220515.dat" --n_grid_iter 1000 --save_iterations $SLURM_ARRAY_TASK_ID
