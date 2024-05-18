#!/bin/bash
#SBATCH --job-name=ept_ais_rn_all_cppropl_20230316
#SBATCH --output=/fred/oz031/epta_code_image/logs_epta_dr2/epta_trim_ais_rn_all_cppropl_20230404_post_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-3
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=6G
#SBATCH --array=6-249

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

export OMP_NUM_THREADS=1

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun singularity exec --bind "/fred/oz031/epta_code_image/image_content/:$HOME" /fred/oz031/epta_code_image/EPTA_ENTERPRISE.sif python3 /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/single_pulsar_analyses/epta_dr2_trim_snall_cpfg_to_recycle_20230403.dat" --target "/home/bgonchar/pta_gwb_priors/params/single_pulsar_analyses/epta_dr2_trim_snall_cpfg_to_recycle_20230403.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/epta_dr2_is_rn_set_all_cppropl_20230316.dat" --n_grid_iter 1000 --save_iterations $SLURM_ARRAY_TASK_ID
