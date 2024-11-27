#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=fall2024-comp551
#SBATCH -e %N-%j.err # STDERR

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024

python ../../Emotion-Classification-using-LLMs/src/Distilled-GPT2/PrepareDataset.py