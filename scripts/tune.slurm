#!/bin/bash
#SBATCH --partition=desktop-batch
#SBATCH --time=47:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30000

module load cuda11.0 gcc9
#SBATCH --gres=gpu:1050:1

export PYTHONPATH=/dcs/18/u1824558/Desktop/CS344
source /dcs/18/u1824558/Desktop/CS344/venv/bin/activate

python bbwrl/tuner.py $1
exit
