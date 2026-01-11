# vLLM Serving Guide with LoRA Adapters

This guide explains how to serve models with LoRA adapters using vLLM's OpenAI-compatible API server.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Starting the Server](#starting-the-server)
- [Making Requests](#making-requests)
- [Dynamic Adapter Management](#dynamic-adapter-management)
- [Troubleshooting](#troubleshooting)

## Quick Start

1. Configure your model and LoRA adapters in `config.yaml`
2. Start the server: `python serve.py`
3. Make requests to `http://localhost:8000`

## Prerequisites for Devstral Model

When serving the Devstral model (especially with merged LoRA weights), you need specific versions of dependencies to avoid compatibility issues.

### 1. Install vLLM Nightly

The Devstral model card recommends vLLM nightly (or a build that includes the required commit).

```bash
# Install vLLM nightly wheels
uv pip install -U vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
```

### 2. Fix NumPy/Numba Compatibility

You may encounter this error: `"Numba needs NumPy 2.2 or less. Got NumPy 2.4."`

To fix this, pin NumPy to a Numba-compatible version (<2.3) and reinstall numba:

```bash
# Pin NumPy to a Numba-compatible version
uv pip install -U "numpy<2.3" "numba"
```

This upper-bound constraint is enforced by numba and prevents the startup failure.

### 3. Install Mistral Tooling

Devstral's HuggingFace documentation explicitly requires mistral_common >= 1.8.6 for tool-call parsing:

```bash
uv pip install -U "mistral_common>=1.8.6"
```

### 4. Verify Installation

Run these quick sanity checks to ensure everything is installed correctly:

```bash
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import numpy, numba; print('numpy', numpy.__version__, 'numba', numba.__version__)"
python -c "import mistral_common; print('mistral_common', mistral_common.__version__)"
```

Expected output:
- vllm: should show a version with nightly build date
- numpy: should show version < 2.3 (e.g., 2.2.x)
- numba: should show latest compatible version
- mistral_common: should show >= 1.8.6

## Configuration

### Base Model Configuration

Edit `config.yaml` to configure your base model:

```yaml
model_information:
  model_config:
    # Use local model or HuggingFace repo
    is_model_local: false
    model_id: "mistralai/Devstral-Small-2-24B-Instruct-2512"
    # If is_model_local is true, use this path
    model_path: "/path/to/local/model"
    trust_remote_code: true
    dtype: "auto"
```

### LoRA Adapter Configuration

Configure LoRA adapters in the `vllm_engine_config` section:

```yaml
vllm_engine_config:
  # Enable LoRA support
  enable_lora: true

  # Define LoRA modules
  lora_modules:
    # Option 1: Local path
    my-adapter-local: "/path/to/local/adapter"

    # Option 2: HuggingFace repository
    my-adapter-hf: "username/repo-name"

    # Example: Your pushed LoRA adapter
    devstral-sft: "Madhurprash/Devstral-Small-2-24B-Instruct-2512-SFT-LoRA-OpenThoughts"

  # Maximum number of LoRA adapters that can be used concurrently
  max_loras: 2

  # Maximum rank of LoRA adapters (set to match or exceed your adapter's rank)
  max_lora_rank: 16
```

**Important Notes:**
- `max_lora_rank`: Set this to at least the rank of your LoRA adapter. Check your adapter config for the rank value.
- `max_loras`: Number of adapters that can be active simultaneously. Increase if you need more concurrent adapters.
- LoRA modules can be specified as local paths or HuggingFace repository IDs (they will be auto-downloaded).

### Memory and Performance Settings

```yaml
vllm_engine_config:
  # Context window size
  max_model_len: 32768

  # GPU memory usage (0.0 to 1.0)
  gpu_memory_utilization: 0.9

  # Multi-GPU support
  tensor_parallel_size: 8
```

## Starting the Server

Run the server with:

```bash
python serve.py
```

The server will:
1. Load the base model
2. Register all configured LoRA adapters
3. Start listening on `http://0.0.0.0:8000`

You'll see output like:
```
============================================================
Starting vLLM OpenAI-Compatible API Server
============================================================
Model: mistralai/Devstral-Small-2-24B-Instruct-2512
Model source: HuggingFace Hub
Max model length: 32768
GPU memory utilization: 0.9

Server will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs
============================================================
```

## Making Requests

### Using the Base Model

To use the base model (without LoRA adapter):

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "prompt": "def fibonacci(n):",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Using a LoRA Adapter

To use a specific LoRA adapter, set the `model` parameter to the adapter name from your config:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "devstral-sft",
    "prompt": "def fibonacci(n):",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Chat Completions with LoRA

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "devstral-sft",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to calculate factorial."}
    ],
    "max_tokens": 512,
    "temperature": 0.6
  }'
```

### Using Python OpenAI Client

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require authentication
)

# Use LoRA adapter
response = client.chat.completions.create(
    model="devstral-sft",  # Your LoRA adapter name
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain async/await in Python"}
    ],
    max_tokens=1024,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Dynamic Adapter Management

vLLM supports loading and unloading LoRA adapters at runtime without restarting the server.

### Enable Dynamic Loading

Set the environment variable before starting the server:

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
python serve.py
```

### Load a New Adapter

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "new-adapter",
    "lora_path": "username/repo-name"
  }'
```

Or with a local path:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "new-adapter",
    "lora_path": "/path/to/local/adapter"
  }'
```

### Unload an Adapter

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "adapter-to-remove"
  }'
```

### Python Example for Dynamic Loading

```python
import requests

BASE_URL = "http://localhost:8000"

# Load adapter
load_response = requests.post(
    f"{BASE_URL}/v1/load_lora_adapter",
    json={
        "lora_name": "my-new-adapter",
        "lora_path": "Madhurprash/Devstral-Small-2-24B-Instruct-2512-SFT-LoRA-OpenThoughts"
    }
)
print(f"Load status: {load_response.status_code}")

# Use the adapter
# ... make inference requests with model="my-new-adapter" ...

# Unload adapter when done
unload_response = requests.post(
    f"{BASE_URL}/v1/unload_lora_adapter",
    json={"lora_name": "my-new-adapter"}
)
print(f"Unload status: {unload_response.status_code}")
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce `gpu_memory_utilization` in config.yaml:
   ```yaml
   gpu_memory_utilization: 0.85  # Try lower values like 0.8, 0.75
   ```

2. Reduce `max_model_len`:
   ```yaml
   max_model_len: 16384  # Or 8192 for very constrained memory
   ```

3. Reduce `max_loras` to load fewer adapters concurrently:
   ```yaml
   max_loras: 1
   ```

### Adapter Not Found

If you get "model not found" errors:

1. Check that the adapter name in your request matches the name in `lora_modules`
2. Verify the adapter path/repo exists and is accessible
3. For HuggingFace repos, ensure you have access and are authenticated if needed:
   ```bash
   huggingface-cli login
   ```

### Slow First Request

The first request after starting the server may be slow as the model warms up. Subsequent requests will be faster.

### Checking LoRA Rank

To find your LoRA adapter's rank, check `adapter_config.json` in your adapter directory:

```bash
cat /path/to/adapter/adapter_config.json
```

Look for the `r` or `rank` field and ensure `max_lora_rank` in your config is set to at least this value.

## API Documentation

Once the server is running, you can access interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Workflow

Here's a complete workflow for serving a fine-tuned model:

1. **Train and save your LoRA adapter** (or download from HuggingFace)

2. **Update config.yaml**:
   ```yaml
   model_config:
     is_model_local: false
     model_id: "mistralai/Devstral-Small-2-24B-Instruct-2512"

   vllm_engine_config:
     enable_lora: true
     lora_modules:
       my-finetuned-model: "Madhurprash/Devstral-Small-2-24B-Instruct-2512-SFT-LoRA-OpenThoughts"
     max_loras: 1
     max_lora_rank: 16
   ```

3. **Start the server**:
   ```bash
   python serve.py
   ```

4. **Make requests**:
   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

   response = client.chat.completions.create(
       model="my-finetuned-model",
       messages=[{"role": "user", "content": "Hello!"}],
       max_tokens=256
   )

   print(response.choices[0].message.content)
   ```

## Testing and Verification

### Verifying LoRA Adapter is Working

The best way to verify that your LoRA adapter is working correctly and not interfering with base model requests is to monitor vLLM's Prometheus metrics while making requests.

#### Setup: Monitor Metrics in Real-Time

In one terminal, start watching the LoRA-specific metrics:

```bash
watch -n 0.2 'curl -s http://localhost:8000/metrics | egrep "vllm:lora_requests_info|running_lora_adapters|waiting_lora_adapters"'
```

This command polls the metrics endpoint every 0.2 seconds and filters for LoRA-related metrics:
- `vllm:lora_requests_info`: Information about LoRA requests
- `running_lora_adapters`: Currently active LoRA adapters
- `waiting_lora_adapters`: Adapters waiting to be loaded

#### Test 1: Base Model Request (No LoRA)

In another terminal, send a long request to the base model:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "messages": [{"role":"user","content":"Write a detailed 2500-token explanation of how to implement a bash-based CI pipeline, include many command examples."}],
    "temperature": 0.0,
    "max_tokens": 2500
  }' > /dev/null
```

**Expected behavior:** In your metrics watch terminal, you should see:
- `running_lora_adapters=""` (empty)
- `waiting_lora_adapters=""` (empty)

This confirms the base model is being used without any LoRA adapter.

#### Test 2: LoRA Adapter Request

Now send a long request using your LoRA adapter:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "devstral-sft-hf",
    "messages": [{"role":"user","content":"Write a detailed 2500-token explanation of how to implement a bash-based CI pipeline, include many command examples."}],
    "temperature": 0.0,
    "max_tokens": 2500
  }' > /dev/null
```

**Expected behavior:** In your metrics watch terminal, you should see:
- `running_lora_adapters="devstral-sft-hf"` (your adapter name appears)
- Request is being processed with the LoRA adapter active

This confirms the LoRA adapter is loaded and being used for inference.

#### Why Long Requests?

We use requests with high `max_tokens` (2500) to ensure the request takes long enough to be visible in the metrics watch window. Short requests might complete before the next metrics poll, making them hard to observe.

### Important Testing Notes

1. **Run one request at a time**: Don't send both requests simultaneously, or you won't be able to clearly distinguish which adapter is active for which request.

2. **Watch for interference**: The key verification is that base model requests show empty LoRA metrics while LoRA requests show the adapter name. This proves there's no interference between the two.

3. **Model name must match config**: Ensure the model name in your request (e.g., `"devstral-sft-hf"`) exactly matches the key you used in `lora_modules` in your config.yaml.

4. **Use `/dev/null` to hide response**: Redirecting output to `/dev/null` keeps your terminal clean and focused on the metrics watch window.

### Common Gotchas

#### Config Field Warnings

You may see warnings in your server logs like:

```
The following fields were present in the config file but are not used: ...
```

This is normal and usually indicates that vLLM received some extra configuration fields that it doesn't recognize. These warnings can typically be ignored unless you're seeing actual functional issues.

#### Adapter Not Showing in Metrics

If you send a LoRA request but don't see the adapter name in metrics:

1. Check that the adapter name in your request matches the config
2. Verify the adapter loaded successfully at server startup (check logs)
3. Ensure `enable_lora: true` is set in your config
4. Confirm the request is actually reaching the server (check for errors)

### Alternative Testing: Quick Request Test

For a simpler test without metrics monitoring:

```bash
# Test base model
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "messages": [{"role":"user","content":"Generate 800 tokens of bash commands to create a project skeleton."}],
    "max_tokens": 800
  }'

# Test LoRA adapter
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "devstral-sft-hf",
    "messages": [{"role":"user","content":"Generate 800 tokens of bash commands to create a project skeleton."}],
    "max_tokens": 800
  }'
```

Compare the responses to see if the LoRA adapter produces different (hopefully improved) output for your specific use case.

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM LoRA Support](https://docs.vllm.ai/en/stable/features/lora/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
