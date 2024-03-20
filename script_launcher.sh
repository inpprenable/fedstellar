#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH -p gpu

cd fedstellar || exit
module load lang/Python
. .venv/bin/activate
python app/main.py > output.log &
echo FedStellar has been launched
sleep 5
echo "Start the script controller"
python -m controller script_json/
echo "Script have been executed"
python app/main.py -st