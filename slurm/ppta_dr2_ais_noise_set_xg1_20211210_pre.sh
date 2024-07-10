#!/bin/bash
#SBATCH --job-name=ppta_ais_noise_gx3_20211124_p
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ppta_ais_noise_gx3_20211124_p_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-2
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=6G
#SBATCH --array=0-499

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

export OMP_NUM_THREADS=1

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_to_recycle_20211124.dat" --target "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpl_fixgam_30_nf_to_recycle_20210811.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_is_noise_set_gx3_20211210.dat" --save_iterations $SLURM_ARRAY_TASK_ID
