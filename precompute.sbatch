#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#sBATCH --mail-user=otautz@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --export=ALL

#export PYTHONPATH=/media/compute/homes/otautz/gits/MineRL2020/env/bin
source /media/compute/homes/otautz/gits/MineRL2020/env3.8/bin/activate
srun python use_resnet.py
