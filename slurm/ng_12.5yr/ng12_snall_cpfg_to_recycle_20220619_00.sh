#!/bin/bash
#SBATCH --job-name=ng_to_rcl_cpfg
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ng12_to_rcl_cpfg_20220619_%A_%a.out
#SBATCH --ntasks=128
#SBATCH --time=1-11
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=8G
#SBATCH --array=7,20

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/run_analysis.py --prfile "/home/bgonchar/pta_gwb_priors/params/ng12.5_snall_cpfg_to_recycle_20220619.dat" --num $SLURM_ARRAY_TASK_ID
