#!/bin/bash
#
# Run TerminalBench with LoRA adapter
#
# This script runs the TerminalBench benchmark using the Devstral LoRA adapter
# served via vLLM. Make sure your vLLM server is running before executing this.
#

set -e

# Configuration - CHANGE THIS to match your LoRA adapter name from config.yaml
LORA_ADAPTER_NAME="devstral-sft-hf"
AGENT_NAME="devstral-sft-lora"
API_BASE="http://localhost:8000/v1"
DATASET="terminal-bench@2.0"
N_CONCURRENT=4
N_ATTEMPTS=5

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}TerminalBench with LoRA Adapter${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Configuration:"
echo "  LoRA Adapter:   ${LORA_ADAPTER_NAME}"
echo "  Agent Name:     ${AGENT_NAME}"
echo "  API Base:       ${API_BASE}"
echo "  Dataset:        ${DATASET}"
echo "  N-Concurrent:   ${N_CONCURRENT}"
echo "  N-Attempts:     ${N_ATTEMPTS}"
echo ""

# Check if vLLM server is running
echo -e "${YELLOW}Checking vLLM server...${NC}"
if ! curl -s --max-time 5 "${API_BASE}/models" > /dev/null 2>&1; then
    echo -e "${RED}Error: vLLM server is not responding at ${API_BASE}${NC}"
    echo "Please start the server first with: cd serve/vLLM && python serve.py"
    exit 1
fi
echo -e "${GREEN}✓ vLLM server is running${NC}"
echo ""

# Create results directory
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="./terminalbench-${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
JOB_NAME="tbench_${AGENT_NAME}"

echo "Results will be saved to: ${RESULTS_DIR}/${JOB_NAME}/"
echo ""
echo -e "${YELLOW}Starting benchmark... (This may take a while)${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENT_PATH="custom_harbor_external_agent:VLLMAgent"

# Add current directory to PYTHONPATH so harbor can import the agent
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run harbor benchmark with the LoRA adapter
harbor run \
    -d "${DATASET}" \
    --agent-import-path "${AGENT_PATH}" \
    --model "${LORA_ADAPTER_NAME}" \
    --ak "api_base=${API_BASE}" \
    --jobs-dir "${RESULTS_DIR}" \
    --job-name "${JOB_NAME}" \
    -n "${N_CONCURRENT}" \
    -k "${N_ATTEMPTS}" \
    -r 0

BENCHMARK_EXIT_CODE=$?

if [ ${BENCHMARK_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Benchmark completed successfully!${NC}"
    echo ""
    echo "Results saved to: ${RESULTS_DIR}/${JOB_NAME}/"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Benchmark failed with exit code: ${BENCHMARK_EXIT_CODE}${NC}"
    exit 1
fi
