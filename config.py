"""
config.py — All hyperparameters and paths in one place.
Edit this file, never scatter magic numbers across the codebase.
"""

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# ── Dataset ──────────────────────────────────────────────────────────────────
TRAIN_FILE = "dataset/training.json"
TEST_FILE  = "dataset/testing.json"

# ── LoRA ─────────────────────────────────────────────────────────────────────
LORA_R           = 16          # rank
LORA_ALPHA       = 32          # scaling factor (alpha/r = 2 is a safe default)
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = [        # Qwen2 attention + MLP projection layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_BIAS        = "none"
LORA_TASK_TYPE   = "CAUSAL_LM"

# ── Training ─────────────────────────────────────────────────────────────────
OUTPUT_DIR          = "output/memory-policy-lora"
ADAPTER_SAVE_DIR    = "adapter/memory-policy-lora"
DPO_ADAPTER_SAVE_DIR    = "adapter/memory-policy-dpo-v1"   # DPO v1 adapter
DPO_ADAPTER_V2_SAVE_DIR = "adapter/memory-policy-dpo-v2"  # DPO v2 adapter
DPO_ADAPTER_V3_SAVE_DIR = "adapter/memory-policy-dpo-v3"  # DPO v3 adapter (current)

NUM_TRAIN_EPOCHS    = 3
PER_DEVICE_BATCH    = 4        # increase if VRAM allows
GRADIENT_ACCUMULATION_STEPS = 4   # effective batch = 4 × 4 = 16
LEARNING_RATE       = 2e-4
LR_SCHEDULER        = "cosine"
WARMUP_RATIO        = 0.05
WEIGHT_DECAY        = 0.01
MAX_SEQ_LENGTH      = 512
FP16                = False    # set True on NVIDIA GPUs that support it
BF16                = True     # Qwen2 was trained with BF16; prefer over FP16

LOGGING_STEPS       = 10
EVAL_STEPS          = 100
SAVE_STEPS          = 100
SAVE_TOTAL_LIMIT    = 2
LOAD_BEST_AT_END    = True
METRIC_FOR_BEST     = "eval_loss"

DATALOADER_NUM_WORKERS = 0     # keep 0 on macOS to avoid multiprocessing issues
SEED                   = 42
