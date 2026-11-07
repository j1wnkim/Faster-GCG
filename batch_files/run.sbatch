#!/bin/bash
#SBATCH --job-name=faster-gcg-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --partition=a30,a100-40gb,a100-80gb
#SBATCH --qos=standby
#SBATCH --account=statdept
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=faster-gcg-verify-%j.out
#SBATCH --error=faster-gcg-verify-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiwon02@gmail.com

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/Faster-GCG}"
SIF_PATH="${SIF_PATH:-${SCRATCH}/container_files/torch.sif}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Error: project directory not found: ${PROJECT_DIR}" >&2
    exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Error: Apptainer image not found: ${SIF_PATH}" >&2
    echo "Set a different image with: sbatch --export=ALL,SIF_PATH=/path/to/torch.sif $0" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

echo "Project: ${PROJECT_DIR}"
echo "Container: ${SIF_PATH}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-assigned by Slurm}"

srun apptainer exec \
    --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${SCRATCH}:${SCRATCH}" \
    "${SIF_PATH}" \
    python tools/main.py -c configs/default.yaml
