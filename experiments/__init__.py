from typing import List

import os
import math
import itertools
import random

SLURM_TASK_BASE_SCRIPT = "{entry_script} {args}"
SLURM_TASK_CPU_BASE_SCRIPT = "{entry_script} {args}"

TIG_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}
#SBATCH --exclude=tig-slurm-4

{job_cmds}
wait < <(jobs -p)
"""

TIG_SLURM_CPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}
#SBATCH --exclude=tig-slurm-4

{job_cmds}
wait < <(jobs -p)
"""

IMP_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=imp
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

{job_cmds}
wait < <(jobs -p)
"""

SUPERCLOUD_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

module load anaconda/2021a
module load cuda/11.4

mkdir -p /state/partition1/user/$USER
cp -r $HOME/mujoco-py /state/partition1/user/$USER/mujoco-py/

source ./setup_supercloud.sh
source ~/proxy.env
ts -S 3
{job_cmds}
bash wait_all.sh
"""

SUPERCLOUD_SLURM_CPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=xeon-p8
#SBATCH -c 16
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

module load anaconda/2021a

mkdir -p /state/partition1/user/$USER
cp -r $HOME/mujoco-py /state/partition1/user/zwhong/mujoco-py/

source ./setup_supercloud.sh
ts-cpu -S 2
{job_cmds}
bash wait_all_cpu.sh
"""

BASE_LOCAL_SCRIPT = """{entry_script} {args}"""

def exp_id():
  import uuid
  return uuid.uuid4()

def hostname():
  import subprocess
  cmd = 'hostname -f'
  try:
      p = subprocess.check_output(cmd, shell=True)  # Save git diff to experiment directory
      return p.decode('utf-8').strip()
  except subprocess.CalledProcessError as e:
      print(f"can not get obtain hostname via `{cmd}` due to exception: {e}")
      return None


def get_local_storage_path():
  """
  Specify local storage separately if you want to change your result directory (e.g., writing files to /scratch is faster than writing it to /data/pulkitag/models/...)
  """
  return {
    "visiongpu50.csail.mit.edu": "/mnt/scratch/zwhong",
    "improbablex001.csail.mit.edu": "/scratch/zwhong",
    "improbablex002.csail.mit.edu": "/scratch/zwhong",
    "improbablex003.csail.mit.edu": "/scratch/zwhong",
    "improbablex004.csail.mit.edu": "/scratch/zwhong",
    "improbablex005.csail.mit.edu": "/mnt/ssd1/zwhong",
    "improbablex006.csail.mit.edu": "/mnt/ssd1/zwhong",
    "improbablex007.csail.mit.edu": "/mnt/ssd1/zwhong",
    # "improbablex009.csail.mit.edu": "/mnt/ssd1/zwhong",
    "katsudon.csail.mit.edu": "/home/zwhong"
  }.get(hostname(), ".")

def _jaynes_entry(entry_script, args):
    """
    Template for launching a jaynes job
    """
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"{script}"
    print(cmd)
    os.system(cmd)

def gcp(job_name, args, entry_script, verbose=False, **kwargs):
    import jaynes

    jaynes.config(
      # "local",
      name=f"{job_name}",
      launch={'name': f"drr_{job_name}"},
      verbose=True)
    instances = jaynes.run(_jaynes_entry,
      entry_script,
      args=args
    )

def sbatch(job_name, args, entry_script, gpu=0, parallel=True, verbose=False, **kwargs):
    task_script = ""
    for idx, task_args in enumerate(args):
      if gpu == -1:
        task_script += SLURM_TASK_CPU_BASE_SCRIPT.format(args=task_args, entry_script=entry_script)
      else:
        task_script += SLURM_TASK_BASE_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        task_script += "\n"

    """
    Determine the SLURM template to use based on the hostname. You may write an new condition for your own cluster
    """
    _hostname = hostname()
    if _hostname.startswith("slurm-control"):
      script = TIG_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
      print("Use TIG Slurm script")
    elif _hostname.startswith("improbable"):
      script = IMP_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
      print("Use improbable script")
    elif _hostname.startswith("login"):
      if gpu == -1:
        script = SUPERCLOUD_SLURM_CPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
        print("Use supercloud CPU script")
      else:
        script = SUPERCLOUD_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
        print("Use supercloud GPU script")
    cmd = f'/bin/bash -c \"sbatch <<< \\"{script}\\" "'
    if verbose:
      print(cmd)
    os.system(cmd)

def screen(job_name, args, entry_script, verbose=False, **kwargs):
    """
    Template for launching a job in a screen session
    """
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"screen -S {job_name} -dm bash -c \"echo $STY; source setup.sh; {script}\""
    if verbose:
      print(cmd)
    os.system(cmd)

def docker(job_name, args, entry_script, verbose=False, **kwargs):
    """
    Template for launching a job in a docker container
    """
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"docker run --gpus=all --name={job_name} -d improbableailab/d3rlpy:latest bash -c \"{script}\""
    if verbose:
      print(cmd)
    os.system("docker pull improbableailab/d3rlpy:latest")
    os.system(cmd)

def bash(job_name, args, entry_script, verbose=False, **kwargs):
    """
    Template for generating bash commands for the job (NOTE: this mode won't run the job)
    """
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"{script}"
    if verbose:
      print(cmd)

def local(job_name, args, entry_script, verbose=False, **kwargs):
    """
    Template for launching the job locally on the current bash session
    """
    assert len(args) == 1 # Not support parallel screen jobs for now
    script = BASE_LOCAL_SCRIPT.format(args=args[0], entry_script=entry_script)
    cmd = f"{script}"

    if verbose:
      print(cmd)
    os.system(cmd)

def launch(job_name, args, mode, entry_script, verbose=False, **kwargs):
  if mode == 'sbatch': # Submit jobs via `sbatch` with GPUs
    sbatch(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == 'sbatch-cpu': # Submit jobs via `sbatch` with CPU only nodes (useful in Supercloud)
    sbatch(job_name, args, entry_script, gpu=-1, verbose=verbose, **kwargs)
  elif mode == 'screen': # Launch each job in an individual screen session
    screen(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == "docker": # Launch each job in a docker container
    docker(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == 'gcp': # Launch each job in a docker container running on GCP
    gcp(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == 'bash': # Print the bash commands for running the jobs
    return bash(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == 'local': # Launch the jobs locally sequentially
    local(job_name, args, entry_script, verbose=verbose, **kwargs)
  else:
    raise NotImplemented()

def parse_task_ids(task_ids):
  if task_ids == ":":
    start_task_id = 0
    end_task_id = None
  else:
    start_task_id, end_task_id = map(lambda x: int(x), task_ids.split(":"))

  return start_task_id, end_task_id

def launch_jobs(experiment: str, all_job_args: List[str], start_task_id, end_task_id, n_jobs, mode, script):
  import time, math
  if end_task_id is None:
    end_task_id = len(all_job_args) - 1
  job_launched = 0
  job_size = math.ceil((end_task_id + 1 - start_task_id) / n_jobs) if n_jobs >= 1 else 1
  expID = exp_id()
  for idx, i in enumerate(range(start_task_id, end_task_id + 1, job_size)):
    launch(f"{experiment}_{expID}_{idx}", all_job_args[i: min(i + job_size, end_task_id + 1)],
        mode=mode, entry_script=script,
        verbose=True)
    print(f"Run task {i}-{min(i + job_size, end_task_id + 1)}")
    job_launched += 1
    # time.sleep(3)
  print(f"Launched {job_launched} jobs. Each job runs {job_size} tasks.")


def get_launch_args(experiment):
  import argparse
  parser = argparse.ArgumentParser(description=f'{experiment}')
  parser.add_argument('--gpus', nargs="+", type=int, help="GPU Id lists (e.g., `--gpus 0 1 2 3` will amortize the workload to GPU 0 1 2 3)", default=0)
  parser.add_argument('--mode', type=str, choices=['sbatch', 'sbatch-cpu', 'screen', 'bash', 'local', "gcp", "docker"], required=True)
  parser.add_argument('--n_parallel_task', type=int, default=1, help="Number of parallel jobs in on sbatch submission (Legacy)")
  parser.add_argument('--task_id', help="Specify the task IDs (e.g., `5:10` will run the tasks of the 5th to 10th hyperparameter combinations)", type=str, required=False, default=":")
  parser.add_argument('--debug', action='store_true', default=False)
  parser.add_argument('--seq', action='store_true', help="(Legacy)", default=False)
  parser.add_argument('--n_jobs', type=int, help="Number of jobs running in sequence.", default=-1)
  parser.add_argument('--n_task_per_gpu', type=int, help="Number of tasks running on the same gpu. (Legacy)", default=1) 
  parser.add_argument('--tags', nargs="+", type=str, default=[])
  parser.add_argument('--run_name', type=str, help="(Legacy)", default=None)
  args = parser.parse_args()
  args.gpus = [args.gpus] if isinstance(args.gpus, int) else args.gpus
  return args

def to_tuple_list(list_tuple):
  tuple_lists = [[] for i in range(len(list_tuple[0]))]
  for t in list_tuple:
    for i, e in enumerate(t):
      tuple_lists[i].append(e)
  return tuple_lists

def sweep(sweep_args, n_parallel_task=1, shuffle=False):
  buffer = [] # a list of tuple, each tuple is one arg combination
  if shuffle:
    sweep_args = list(sweep_args)
    random.shuffle(sweep_args)
  n_args = len(sweep_args)
  for args in sweep_args:
    buffer.append(args)
    if len(buffer) == n_parallel_task:
      yield (len(buffer), *to_tuple_list(buffer))
      buffer = []
  if len(buffer) > 0:
    yield (len(buffer), *to_tuple_list(buffer))
    buffer = []

def sweep_with_devices(sweep_args, devices, n_jobs, n_parallel_task=1, shuffle=False):
  buffer = [] # a list of tuple, each tuple is one arg combination
  sweep_args = list(sweep_args)
  if shuffle:    
    random.shuffle(sweep_args)

  n_args = len(sweep_args)
  n_tasks_per_device = math.ceil(n_args / n_jobs)

  for idx, args in enumerate(sweep_args):
    buffer.append(args)
    device = devices[(idx // n_tasks_per_device) % len(devices)]
    if len(buffer) == n_parallel_task:
      yield (len(buffer), device, *to_tuple_list(buffer))
      buffer = []

  if len(buffer) > 0:
    device = devices[(idx // n_tasks_per_device) % len(devices)]
    yield (len(buffer), device, *to_tuple_list(buffer))
    buffer = []
