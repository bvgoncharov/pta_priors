#!/bin/bash
#SBATCH --job-name=cp_vanilla_poly
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ng15_cp_vanilla_poly_20240716_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=0-3
#SBATCH --mem-per-cpu=5G
#SBATCH --tmp=4G
#SBATCH --array=0-67

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

export TEMPO2_CLOCK_DIR=/home/bgonchar/clock_nanograv_15/
export PINT_CLOCK_OVERRIDE=/home/bgonchar/clock_nanograv_15/

export PMIX_MCA_psec=^munge

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun singularity exec --bind "/fred/oz031/skymap_container/image_content/:$HOME" /fred/oz031/skymap_container/pulsarenv_20240403.sif mpirun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/cp_vanilla_polych_20240716.dat" --num $SLURM_ARRAY_TASK_ID
