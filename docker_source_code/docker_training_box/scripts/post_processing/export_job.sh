#!/bin/sh
#SBATCH -t 1-0:00
#SBATCH --output=export_job.out
#SBATCH --nodelist chela-g01
#SBATCH --partition mahaguru
#SBATCH --job-name=modelExport
#SBATCH -o /home/ajouffray/slurm/export_job.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=61000

source /opt/software/anaconda3/2020.11/anaconda/etc/profile.d/conda.sh

nvidia-smi
 
conda activate tensorflow

python export.py /home/ajouffray/TF-Object-Detection/training/fruits2_frcnn_resnet50-0.0/ 0 
