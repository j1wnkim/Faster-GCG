# Faster-GCG

[![arXiv](https://img.shields.io/badge/arXiv-2410.15362-b31b1b.svg)](https://arxiv.org/abs/2410.15362)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![UV](https://img.shields.io/badge/uv-✓)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This repository contains the official code implementation for [*Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models*](https://arxiv.org/abs/2410.15362).

## Quickstart

### Prepare the environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To install the required packages, run:

```bash
# If you get stuck, add -v for a verbose output
uv sync
# Optionally run this to activate venv in the current shell
source .venv/bin/activate
```

*Flash attention.* Optionally, you can install the required packages for flash attention. It requires nvcc with `cuda>=11.7`.
The compilation toolchain is prepared as follows:

```bash
conda create -n nvcc && conda activate nvcc
conda install cuda-toolkit=12.8
```

Then, install flash attention by running:

```bash
# -v for verbose output. Recommended since the compilation takes a while.
uv sync --group flash-attn --no-build-isolation -v
```

### Download the resources

All models and datasets are on [Hugging Face](https://huggingface.co/). To download the resources, use the command `hf download`. To list all downloaded resources, use the command `hf cache ls`.

### Run the attack generation

`tools/main.py` is the program entry.

Verification is strongly recommended before running:

```bash
python tools/main.py -c configs/default.yaml --verify
```

To start running the attack generation (Faster-GCG++ on Llama-2), run:

```bash
python tools/main.py -c configs/default.yaml
```

**The program automatically consumes all GPUs on the current machine, with one worker per GPU.**
To limit the GPUs used, set the environment variable `CUDA_VISIBLE_DEVICES`.

You may append more configuration files via:

```bash
python tools/main.py -c configs/default.yaml -c configs/<your config 1>.yaml configs/<your config 2>.yaml -c configs/<your config 3>.yaml
```

To switch between different models, append a configuration file `configs/models/<model name>.yaml`.

To run other attacks including the basic Faster-GCG (without ++), the base configuration file is `configs/ablation/base.yaml`.

- Faster-GCG: `python tools/main.py -c configs/ablation/base.yaml`.
- GCG: `python tools/main.py -c configs/ablation/base.yaml configs/ablation/no_faster.yaml`.
- I-GCG: `python tools/main.py -c configs/ablation/base.yaml configs/ablation/i-gcg.yaml`.
- Probe sampling: `python tools/main.py -c configs/ablation/base.yaml configs/ablation/probe_sampling.yaml`.
- TAO-Attack: `python tools/main.py -c configs/ablation/base.yaml configs/ablation/tao_attack.yaml`.

For these attacks, the model is switched by appending `configs/models/<model name>_base.yaml`.

To run other datasets, append a configuration file `configs/datasets/<dataset name>.yaml`.

### Run the evaluation

During training, we use a simple string matcher to give numbers on the attack success rate (ASR). To stabilize the evaluation, we evaluate the attack with multiple judges. To run the evaluation, append `configs/eval.yaml` with the directory. For example:

```bash
python tools/main.py -c configs/default.yaml configs/eval.yaml --expr-path </path/to/experiment>
```

You may also replace `--expr-path </path/to/experiment>` with `--load </path/to/experiment>/train_raw_data.pkl`, and the results will be logged to another directory.

*Note: the base config must be correctly specified, and the vLLM services for target model and judges should be running.*

## Citation

If you find that our work is helpful to you, please star this project and consider to cite:

```bibtex
@article{li2024faster-gcg,
  title={Faster-gcg: Efficient discrete optimization jailbreak attacks against aligned large language models},
  author={Li, Xiao and Li, Zhuhong and Li, Qiongxiu and Lee, Bingze and Cui, Jinghao and Hu, Xiaolin},
  journal={arXiv preprint arXiv:2410.15362},
  year={2024}
}
```

## Acknowledgements

- [llm-attacks](https://github.com/llm-attacks/llm-attacks): original implementation of GCG.
- [nanoGCG](https://github.com/GraySwanAI/nanoGCG): simplified implementation of GCG, which our code builds upon.
- [I-GCG](https://github.com/jiaxiaojunQAQ/I-GCG)
