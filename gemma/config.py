"""FunctionGemma model configuration.

This file contains all model-specific configuration for FunctionGemma models.
Following Google's official FunctionGemma documentation:
- https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
- https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma

Now uses HuggingFace Transformers directly instead of llama.cpp for proper
tokenization and chat template handling.
"""

import os
from pathlib import Path

# Model identification - HuggingFace model ID
# Use fine-tuned model if available, otherwise fallback to base model
FINETUNED_MODEL_PATH = str(Path(__file__).parent / "fine-tune" / "functiongemma-finetuned")
MODEL_ID = os.getenv("MODEL_ID", FINETUNED_MODEL_PATH if Path(FINETUNED_MODEL_PATH).exists() else "google/functiongemma-270m-it")
MODEL_NAME = "FunctionGemma 270M Instruct (Fine-tuned)" if Path(FINETUNED_MODEL_PATH).exists() else "FunctionGemma 270M Instruct"

# Generation parameters (optimized for FunctionGemma 270M)
# Lower temperature prevents repetition in small models
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# Device configuration
DEVICE = os.getenv("DEVICE", None)  # None = auto-detect (cuda if available, else cpu)
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")  # auto, float16, bfloat16, float32

# Agent configuration
AGENT_NAME = os.getenv("AGENT_NAME", "Aria")
PROPERTY_NAME = os.getenv("PROPERTY_NAME", "Sunset Apartments")

# Paths - all outputs stored in this model's folder
MODEL_DIR = Path(__file__).parent
LOG_FILE = str(MODEL_DIR / "conversations.json")
TEST_RESULTS_DIR = MODEL_DIR / "test_results"
