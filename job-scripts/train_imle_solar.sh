#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=imle-train
#SBATCH --partition=mars-lab-short
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/dre3/Repos/imle_policy_980/%x_%j.out
#SBATCH --error=/home/dre3/Repos/imle_policy_980/%x_%j.error
#SBATCH --mail-user=dre3@sfu.ca
#SBATCH --mail-type=END,FAIL

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n imle_policy -c conda-forge python=3.10 evdev=1.9.0 xorg-x11-proto-devel-cos6-x86_64 glew mesa-libgl-devel-cos6-x86_64 libglib
conda activate imle_policy
cd /home/dre3/Repos/imle_policy_980
pip install -e .

export WANDB_API_KEY="3edf01a34993112b3c0a356b0280f938b54e2247"
python -m torch.distributed.run --nproc_per_node=4 imle_policy/train.py --task zarr --method rs_imle
