#!/bin/sh
#SBATCH -t 1-0:00
#SBATCH --output=saveCheckpoint.out
#SBATCH --nodelist chela-g01
#SBATCH --partition mahaguru
#SBATCH --job-name=saveCheckpoint
#SBATCH -o /home/ajouffray/slurm/saveCheckpoint.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000

source /opt/software/anaconda3/2020.11/anaconda/etc/profile.d/conda.sh

python saveCheckpoint.py /home/ajouffray/TF-Object-Detection/training/apples_frcnn_resnet50/

