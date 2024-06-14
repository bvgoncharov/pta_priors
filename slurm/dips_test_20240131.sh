#!/bin/bash
#SBATCH --job-name=ms_dips
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ng15_ms_dips_20240131_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=0-23
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=4G
#SBATCH --array=7

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun singularity exec --bind "/fred/oz031/skymap_container/image_content/:$HOME" /fred/oz031/skymap_container/anpta.sif mpiexec -n 4 python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ms_dips_20240131.dat" --num $SLURM_ARRAY_TASK_ID
