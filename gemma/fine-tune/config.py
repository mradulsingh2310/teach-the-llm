"""
Configuration for FunctionGemma Fine-tuning

Based on Google's recommended settings:
https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training configuration for FunctionGemma fine-tuning.

    Based on Unsloth's recommended settings:
    https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb
    """

    # Model settings
    model_id: str = "google/functiongemma-270m-it"
    output_dir: str = "./functiongemma-finetuned"

    # Training hyperparameters (Unsloth recommended)
    # Use max_steps instead of epochs to prevent overfitting on large datasets
    num_train_epochs: int = 3  # Fallback if max_steps not used
    max_steps: int = 500  # Unsloth uses 500 steps regardless of dataset size
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2  # Effective batch size = 8

    # CRITICAL: Learning rate - Unsloth uses 2e-4, NOT 2e-5
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"  # Linear decay, not constant
    warmup_steps: int = 10  # Unsloth uses 10 warmup steps

    # Sequence settings
    max_seq_length: int = 512

    # Precision settings
    bf16: bool = True  # Use bfloat16 for training (better numerical stability)
    fp16: bool = False

    # Optimizer - adamw_torch works on all platforms (adamw_8bit requires bitsandbytes/CUDA)
    optim: str = "adamw_torch"
    weight_decay: float = 0.001  # Unsloth uses 0.001, not 0.01
    max_grad_norm: float = 1.0

    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 50  # Evaluate every 50 steps
    save_strategy: str = "steps"
    save_steps: int = 100  # Save every 100 steps
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True
    report_to: str = "none"

    # Data processing
    packing: bool = False  # Don't pack sequences (preserve conversation boundaries)
    dataset_text_field: str = "text"

    # Seed for reproducibility
    seed: int = 42


@dataclass
class LoRAConfig:
    """LoRA configuration for parameter-efficient fine-tuning.

    Unsloth uses rank=128 for FunctionGemma fine-tuning.
    Higher rank = more capacity to learn new patterns.
    """

    enabled: bool = True
    r: int = 128  # Rank - Unsloth uses 128, not 16
    lora_alpha: int = 256  # Typically 2x rank
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.0  # Unsloth uses 0 dropout
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Data configuration for training."""

    # Paths
    train_data_path: str = "./training_data.json"
    eval_split: float = 0.1  # 10% for evaluation

    # Processing
    max_samples: Optional[int] = None  # Limit samples for quick testing
    shuffle: bool = True

    # Column names (for HuggingFace datasets)
    messages_column: str = "messages"
    tools_column: str = "tools"


def get_default_config() -> dict:
    """Get default configuration as a dictionary."""
    return {
        "training": TrainingConfig().__dict__,
        "lora": LoRAConfig().__dict__,
        "data": DataConfig().__dict__,
    }
