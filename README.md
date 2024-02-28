Easy Experiment Launching Tool
=

# Purpose of this repository
Provide an easy-to-customizable tool for launching a large number of experiments in a second.

# Why this tool?
To my experience, most of experiment management tools are too complicated to modify and often requires the experiment code to adapt on the launcher. Take `hydra` for example. You have to follow their decorator (e.g., `@hydra.main`) to take arguments from the CLI or use their configuration files (e.g., `OmegaConf`) to setup the experiments.  

# Getting started
We have a dummy example in this repository. Run `export PYTHONPATH=$(pwd):$PYTHONPATH`.
```
python experiments/dummy/run.py --mode local --gpus 0 # Runing jobs locally and sequentially on GPU 0
python experiments/dummy/run.py --mode sbatch --gpus 0 --n_jobs 4 # Evenlly distribute the tasks to 4 SLURM jobs and run them each on GPU 0
python experiments/dummy/run.py --mode screen --gpus 0 1 --n_jobs 4  # Evenlly distribute the tasks to 4 screen sessions and run them each on GPU 0 and 1
```