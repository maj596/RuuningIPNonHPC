#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 70:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=maj596@nyu.edu
# gpu:v100:1 300G memory i  can ask


source ~/.bashrc
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
# conda init bash
# conda shell.bash hook
# conda activate /scratch/maj596/conda-envs/ipnv1

#Your application commands go here
python train.py