#!/bin/bash
###############################################################################
##                                                                           ##
##                           Campus Cluster                                  ##
##                   Sample Python Job Batch Script                          ##
##                                                                           ##
## SLURM Options (To view, run the following command below)                  ##
##                                                                           ##
##     man sbatch                                                            ##
##                                                                           ##
###############################################################################
#                                                                            
#SBATCH --time=03:00:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=16             # Number of task (cores/ppn) per node
#SBATCH --job-name=python_job            # Name of batch job
#SBATCH --gres=gpu:A100:1                  # Request 1 GPU of type A100 (if needed)
#SBATCH --partition=IllinoisComputes-GPU       # Partition (queue)
#SBATCH --account=jiaxins8-ic               # Bactch account to use   
#SBATCH --output=python_job.o%j             # Name of batch job output file
#SBATCH --mail-user=jiaxins8@illinois.edu  # Send email notifications
#SBATCH --mail-type=BEGIN,END              # Type of email notifications to send
#                                                                            
###############################################################################
# Change to the directory from which the batch job was submitted
# Note: SLURM defaults to running jobs in the directory where
# they are submitted, no need for cd'ing to $SLURM_SUBMIT_DIR

cd /u/jiaxins8/Fair-Mechanism-Shapley-Value/MNIST/


# Load Python/Anaconda module (Enable Python in batch job environment)
# by loading a Python or Anaconda modulefile (Only one modulfile should be
# loaded. To use the python modulefile below, uncomment that "module load ..."
# line and comment out the anaconda "module load ..." line.) 

#module load python
#  or
module load python
pip install tqdm


# Run the python script(code)
python main.py --method br --params fedavg.yaml
# Or to run multiple instances of the python script(code)
#srun python3 my_code.py

