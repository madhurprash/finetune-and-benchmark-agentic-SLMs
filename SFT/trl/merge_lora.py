import os
import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "mistralai/Devstral-Small-2-24B-Instruct-2512"
LORA_DIR   = "/home/ubuntu/devstral/benchmark-agentic-SLMs/SFT/trl/outputs/devstral-sft"
OUT_DIR    = "/home/ubuntu/devstral/merged-devstral-sft"

os.makedirs(OUT_DIR, exist_ok=True)

# Tokenizer
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tok.save_pretrained(OUT_DIR)

# Load base in bf16 (merge should happen in bf16)
base = Mistral3ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Wrap with PEFT adapter
peft_model = PeftModel.from_pretrained(base, LORA_DIR)
print("Wrapped type:", type(peft_model))
print("Has merge_and_unload:", hasattr(peft_model, "merge_and_unload"))

# Merge
merged = peft_model.merge_and_unload()

# Prevent Transformers from trying to "revert weight conversions" (fp8->bf16 path)
for attr in ("_hf_weight_conversions", "_hf_weight_conversion"):
    if hasattr(merged, attr):
        setattr(merged, attr, [])
    if hasattr(merged, "base_model") and hasattr(merged.base_model, attr):
        setattr(merged.base_model, attr, [])

# Save merged model
merged.save_pretrained(OUT_DIR, safe_serialization=True)
print(f"✅ Merged model saved to: {OUT_DIR}")
