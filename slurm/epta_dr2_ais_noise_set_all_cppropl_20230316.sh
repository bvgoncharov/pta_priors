#!/bin/bash
#SBATCH --job-name=ppta_ais_rn_all_cppropl_20220204
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/ppta_ais_rn_all_cppropl_20220204_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-6
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=6G
#SBATCH --array=48,49,67,68,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,150,151,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

export OMP_NUM_THREADS=1

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/fred/oz031/epta_code_image/image_content/epta_dr3/params/_epta_dr3_snall_cpfg_to_recycle_20230314.dat" --target "/fred/oz031/epta_code_image/image_content/epta_dr3/params/_epta_dr3_snall_cpfg_to_recycle_20230314.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/epta_dr2_is_rn_set_all_cppropl_20230316.dat" --n_grid_iter 1000 --save_iterations $SLURM_ARRAY_TASK_ID
