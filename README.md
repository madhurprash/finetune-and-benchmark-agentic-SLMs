# NVIDIA Nemotron-3-Nano-30B OpenThoughts Benchmark

Benchmarking NVIDIA Nemotron-3-Nano-30B-A3B-BF16 on the [OpenThoughts dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev) using [Harbor](https://www.openthoughts.ai/blog/agent) for agentic reasoning evaluation.

## Overview

This project hosts the Nemotron 3 Nano 30B MoE model on a P4 GPU instance and evaluates its agentic capabilities on OpenThoughts-TB-dev, a benchmark for measuring reasoning and problem-solving abilities in autonomous agents.

### Architecture

```
┌─────────────────────┐         ┌──────────────────────┐         ┌─────────────────┐
│  Harbor CLI         │────────▶│  External Agent      │────────▶│  vLLM Server    │
│  (Benchmark Runner) │         │  (Nemotron Adapter)  │         │  (localhost:8000)│
└─────────────────────┘         └──────────────────────┘         └─────────────────┘
         │                                                                  │
         │                                                                  │
         ▼                                                                  ▼
┌─────────────────────┐                                          ┌─────────────────┐
│  OpenThoughts       │                                          │  NVIDIA P4 GPU  │
│  Dataset            │                                          │  Nemotron 3 Nano│
└─────────────────────┘                                          └─────────────────┘
```

**Components:**
- **vLLM Server**: Serves Nemotron 3 Nano via OpenAI-compatible API
- **Harbor Agent**: Custom external agent implementing iterative reasoning loops
- **OpenThoughts Dataset**: Agentic task benchmark for evaluation

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install harbor-ai vllm

# Verify GPU availability
nvidia-smi
```

### 2. Start Model Server

```bash
# Launch vLLM server with Nemotron 3 Nano (Terminal 1)
python serve_api.py

# Server runs at http://localhost:8000
```

### 3. Run Benchmark

```bash
# Execute automated benchmark (Terminal 2)
chmod +x benchmark_openthoughts.sh
./benchmark_openthoughts.sh
```

The script will:
1. Download the OpenThoughts-TB-dev dataset
2. Install Harbor if needed
3. Run benchmarks using the custom external agent
4. Save results to `./benchmark_results/`

## Configuration

**Model Settings** (`config.yaml`):
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Max Context: 262K tokens
- GPU Memory: 85% utilization
- Temperature: 0.6

**Agent Settings** (`nvidia_nemotron_agent.py`):
- Iterative reasoning (max 10 iterations)
- Bash command extraction and execution
- Custom system prompts for agentic tasks

## Results

Benchmark results include:
- Task success rate
- Reasoning quality scores
- Completion accuracy
- Execution traces and logs

Results are timestamped and saved in `./benchmark_results/benchmark_nvidia-nemotron-nano_<timestamp>/`

## Design Notes

The custom Harbor external agent (`nvidia_nemotron_agent.py`) implements:
1. **Iterative Problem Solving**: Multi-turn interaction with the model
2. **Command Execution**: Extracts and runs bash commands from model responses
3. **Feedback Loop**: Feeds execution results back to the model for reflection

This design enables the model to autonomously solve complex tasks through tool use and iterative reasoning.

## Documentation

- **Detailed Setup**: See [BENCHMARK_GUIDE.md](./BENCHMARK_GUIDE.md)
- **OpenThoughts**: https://www.openthoughts.ai/blog/agent
- **Dataset**: https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev
- **Model**: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

## Requirements

- NVIDIA GPU with 48GB+ VRAM (P4, A100, etc.)
- CUDA 12.1+
- Python 3.10+
- ~100GB disk space (model + dataset)

## License

Model: [NVIDIA Open Model License](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
