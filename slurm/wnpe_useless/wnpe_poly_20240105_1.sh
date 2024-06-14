#!/bin/bash
#SBATCH --job-name=wnpe_poly
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ng15_wnpe_2040105_poly_%A_%a.out
#SBATCH --ntasks=12
#SBATCH --time=1-23
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=8G
#SBATCH --array=30

export PMIX_MCA_psec=^munge
export OMP_NUM_THREADS=1

export SINGULARITYENV_TEMPO2_CLOCK_DIR=/home/bgonchar/clock_nanograv_15/
export SINGULARITYENV_PINT_CLOCK_OVERRIDE=/home/bgonchar/clock_nanograv_15/

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun singularity exec --bind "/fred/oz031/skymap_container/image_content/:$HOME" /fred/oz031/skymap_container/pulsarenv_20240319.sif mpirun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/wnpe_poly_20240105.dat" --num $SLURM_ARRAY_TASK_ID
