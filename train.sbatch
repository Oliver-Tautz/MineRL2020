#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#sBATCH --mail-user=otautz@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --export=ALL


#export PYTHONPATH=/media/compute/homes/otautz/gits/MineRL2020/env/bin
source /media/compute/homes/otautz/gits/MineRL2020/env/bin/activate
srun python train.py test_gpu 
