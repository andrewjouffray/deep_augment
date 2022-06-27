#!/bin/sh
#SBATCH -t 1-0:00
#SBATCH --output=data_prep_job.out
#SBATCH --nodelist chela-g01
#SBATCH --partition mahaguru
#SBATCH --job-name=deepWeeds
#SBATCH -o /home/ajouffray/slurm/data_prep_job.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=62000

source /opt/software/anaconda3/2020.11/anaconda/etc/profile.d/conda.sh

nvidia-smi
 
conda activate tensorflow


python data_prep_boxes.py /home/ajouffray/Data/dataset_fruits/augmented_fruits2/ 1 true 
