# vLLM Model Fine-tuning & Benchmarking on `OpenThoughts` & `Terminal Bench 2.0`

Benchmarking any vLLM-supported model on the [OpenThoughts dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev) using [Harbor](https://www.openthoughts.ai/blog/agent) for agentic reasoning evaluation.

## Overview

This project provides a flexible framework to serve and evaluate any vLLM-supported model on OpenThoughts-TB-dev, a benchmark for measuring reasoning and problem-solving abilities in autonomous agents. Simply configure your model in `config.yaml` and run the benchmark.

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
- **vLLM Server**: Serves any vLLM-compatible model via OpenAI-compatible API
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

### 2. Configure Your Model

Edit `config.yaml` to specify your model:
```yaml
model_information:
  model_config:
    model_id: "your-org/your-model-name"  # e.g., "mistralai/Devstral-Small-2-24B-Instruct-2512"
    # ... other settings
```

### 3. Start Model Server

```bash
# Launch vLLM server with your configured model (Terminal 1)
python serve_api.py

# Server runs at http://localhost:8000
```

### 4. Run Benchmark

```bash
# Execute automated benchmark (Terminal 2)
chmod +x benchmark_openthoughts.sh
./benchmark_openthoughts.sh
```

The script will:
1. Read your model configuration from `config.yaml`
2. Download the OpenThoughts-TB-dev dataset
3. Install Harbor if needed
4. Run benchmarks using the custom external agent
5. Save results to `./benchmark_results/`

## Configuration

**Model Settings** (`config.yaml`):
- `model_id`: Your HuggingFace model identifier
- `max_model_len`: Maximum context length
- `quantization`: Quantization method (e.g., "fp8", "awq")
- `gpu_memory_utilization`: GPU memory usage (0.0-1.0)
- `temperature`: Default sampling temperature
- `max_tokens`: Default maximum tokens to generate

Example model configurations:
- Devstral Small 2: `mistralai/Devstral-Small-2-24B-Instruct-2512`
- NVIDIA Nemotron: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Any other vLLM-compatible model

**Agent Settings** (`nvidia_nemotron_agent.py`):
- Iterative reasoning (max 15 iterations)
- Bash command extraction and execution
- Custom system prompts for agentic tasks

## Results

Benchmark results include:
- Task success rate
- Reasoning quality scores
- Completion accuracy
- Execution traces and logs

Results are timestamped and saved in `./benchmark_results/benchmark_<model-name>_<timestamp>/`

## Design Notes

The custom Harbor external agent (`nvidia_nemotron_agent.py`) implements:
1. **Iterative Problem Solving**: Multi-turn interaction with the model
2. **Command Execution**: Extracts and runs bash commands from model responses
3. **Feedback Loop**: Feeds execution results back to the model for reflection

This design enables the model to autonomously solve complex tasks through tool use and iterative reasoning.

## Documentation

- **OpenThoughts**: https://www.openthoughts.ai/blog/agent
- **Dataset**: https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev
- **vLLM**: https://docs.vllm.ai/

## Supported Models

This framework works with any model supported by vLLM, including:
- Mistral/Mistral-based models (Devstral, Mistral-7B, etc.)
- NVIDIA models (Nemotron series)
- Llama models
- Qwen models
- And many more - see [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Requirements

- NVIDIA GPU (requirements vary by model size)
  - Small models (7-8B): 16GB+ VRAM
  - Medium models (20-30B): 48GB+ VRAM
  - Large models: 80GB+ VRAM
- CUDA 12.1+
- Python 3.10+
- Disk space varies by model size (typically 50-100GB for model + dataset)

## License

Check the license for your specific model on HuggingFace.
