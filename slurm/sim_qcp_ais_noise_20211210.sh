#!/bin/bash
#SBATCH --job-name=sim_qcp_ais_noise_20211210
#SBATCH --output=/fred/oz031/logs_pta_gwb_priors/sim_qcp_ais_noise_20211210_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-3
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=3G
#SBATCH --array=0

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/pta_gwb_priors/params/sim_qcp_to_recycle_20211206.dat" --target "/home/bgonchar/pta_gwb_priors/params/sim_qcp_to_recycle_cpl_vargam_20211206.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/sim_qcp_is_noise_20211210.dat"