#!/bin/bash
#SBATCH --job-name=wnpe_p
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ng15_wnpe_pint_20240318_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=0-23
#SBATCH --mem-per-cpu=1G
#SBATCH --tmp=4G
#SBATCH --array=24

export PMIX_MCA_psec=^munge
export OMP_NUM_THREADS=1

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

export TEMPO2_CLOCK_DIR=/home/bgonchar/clock_nanograv_15/
export PINT_CLOCK_OVERRIDE=/home/bgonchar/clock_nanograv_15/

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
singularity exec --bind "/fred/oz031/skymap_container/image_content/:$HOME" /fred/oz031/skymap_container/pulsarenv_20240319.sif mpirun -np 4 python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/wnpe_pint_20240318.dat" --num $SLURM_ARRAY_TASK_ID
