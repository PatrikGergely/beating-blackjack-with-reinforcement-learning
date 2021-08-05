#!/bin/bash
#
#SBATCH --partition=desktop-debug
#SBATCH --mem-per-cpu=2499
#SBATCH --time=47:00:00
#SBATCH --ntasks-per-node=1
#
##SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this

module load gcc9
export PYTHONPATH=/dcs/18/u1824558/Desktop/CS344
source /dcs/18/u1824558/Desktop/CS344/venv/bin/activate

for ((  i=0; i<$SLURM_TASKS_PER_NODE; i++ ))
do
  proc_id=$((SLURM_ARRAY_TASK_ID*SLURM_TASKS_PER_NODE + i))
  python3 bbwrl/simulator.py $1 ${proc_id} &
done

wait
exit
