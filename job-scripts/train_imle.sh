#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=imle-train
#SBATCH --partition=gpubase_interac
#SBATCH --time=26:00:00
#SBATCH --mem=24G
#SBATCH --mail-user=<hsd31@sfu.ca>
#SBATCH --mail-type=ALL

module load StdEnv/2023 python/3.11.5 scipy-stack cuda
source ~/py311/bin/activate
export WANDB_API_KEY="863937219a6baaf8baba21ad53db86e3ff031d8a"
cd ~/projects/aip-keli/hsd31/imle_policy_980
python -m torch.distributed.run --nproc_per_node=2 imle_policy/train.py --task zarr --method rs_imle