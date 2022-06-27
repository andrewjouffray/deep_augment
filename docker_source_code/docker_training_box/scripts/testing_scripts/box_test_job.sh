#!/bin/sh
#SBATCH -t 1-0:00
#SBATCH --output=test.out
#SBATCH --nodelist chela-g01
#SBATCH --partition mahaguru
#SBATCH --job-name=deepWeeds
#SBATCH -o /home/ajouffray/slurm/test.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=31000

source /opt/software/anaconda3/2020.11/anaconda/etc/profile.d/conda.sh

nvidia-smi
 
activate tensorflow


python box_test_model.py /home/ajouffray/Data/home_fruit_test_2/imgs /home/ajouffray/Data/home_fruit_test_2/xml /home/ajouffray/TF-Object-Detection/exported/fruits2_frcnn_resnet50-0.0/fold0/ /home/ajouffray/Data/dataset_fruits/augmented_fruits2/label.pbtxt 0
