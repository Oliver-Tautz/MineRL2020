#!/bin/bash
#SBATCH --gres=gpu:gtx
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --time=100:00:00
#sBATCH --mail-user=otautz@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --export=ALL

#export PYTHONPATH=/media/compute/homes/otautz/gits/MineRL2020/env/bin
source /media/compute/homes/otautz/gits/MineRL2020/env3.8/bin/activate
srun python train.py "$@"
