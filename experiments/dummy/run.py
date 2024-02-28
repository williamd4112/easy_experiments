import os
import sys
import copy
import itertools
import datetime
import random
from functools import reduce
from pathlib import Path

from experiments import get_launch_args, sweep, sweep_with_devices, launch, exp_id, parse_task_ids, launch_jobs, get_local_storage_path

if __name__ == '__main__':
  experiment = f"{os.path.basename(os.path.dirname(Path(__file__)))}"
  launch_args = get_launch_args(experiment)
  
  # The scripts to run
  algo_scripts = [
    {"script": "python3 dummy.py",}, 
  ]

  # Hyperparameters: The following will generate {'MNIST', 'CIFAR-10'} X {'AlexNet', 'LeNet'}
  datasets = [
    "MNIST",
    "CIFAR-10"
  ]

  models = [
    "AlexNet",
    "LeNet",
  ]
  
  # Common arguments for all processess
  common_exp_args = [
    "--lr 1e-4"
  ]

  # Debug-only arguments
  if not launch_args.debug:
    common_exp_args.append("--debug")

  all_job_args = []
  for job_idx, (n_tasks, # NOTE: number of parallel tasks in one process. no longer maintain. it'll be always 1
            device,
            algo_script,
            dataset,
            model,
          ) in enumerate(
          sweep_with_devices(itertools.product(
              algo_scripts,
              datasets,
              models
            ),
            devices=launch_args.gpus,
            n_jobs=launch_args.n_jobs,
            n_parallel_task=1, # NOTE: don't change
            shuffle=True)):

    # Start composing the command arguments          

    job_args = [] # list of all job arguments (each job arg string is a task)
    for task_idx in range(n_tasks):
      # Arguments for the current task (NOTE: now we always have only one task)
      args = [
        algo_script[task_idx]["script"].format(gpu=device),
      ] + common_exp_args

      # Append all the arguments to the `job_args`
      args.append(f"--dataset {dataset[task_idx]}")
      args.append(f"--model {model[task_idx]}")
      
      # Concatenate `args` as a string (task argument string) into `job_args` 
      job_args.append(" ".join(args))
    all_job_args.append(job_args[0]) # Append the first task argument string because we only have one task for each job

    # If debug mode is on, we only launch one job
    if launch_args.debug:
      break

  print(f"Total: {len(all_job_args)}")

  launch_jobs(
    (experiment + f"_{launch_args.run_name}" if launch_args.run_name else experiment),
    all_job_args,
    *parse_task_ids(launch_args.task_id),
    n_jobs=launch_args.n_jobs,
    mode=launch_args.mode,
    script="")

  print(f"Total: {len(all_job_args)}, num_gpus={len(launch_args.gpus)}")
