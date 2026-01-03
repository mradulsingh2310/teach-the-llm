# -*- coding: utf-8 -*-
"""
FunctionGemma Fine-tuning Script

Based on Google's official fine-tuning guide:
https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/functiongemma/finetuning-with-functiongemma.ipynb

Usage:
    python train.py --data_path ./sample-training-data.json --output_dir ./functiongemma-finetuned
    python train.py --data_path ./sample-training-data.json --epochs 8 --learning_rate 5e-5
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# Default system message for function calling
DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"


def load_data(data_path: str) -> list:
    """Load training data from JSON file."""
    print(f"Loading data from {data_path}...")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle different data structures
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    print(f"Loaded {len(data)} examples")
    return data


def create_dataset(data: list) -> Dataset:
    """
    Create HuggingFace dataset from loaded data.

    Supports two formats:
    1. Messages format (already has "messages" and "tools" keys)
    2. Simple format (has "user_content", "tool_name", "tool_arguments")
    """
    processed = []

    for item in data:
        if "messages" in item:
            # Already in messages format - use directly
            processed.append({
                "messages": item["messages"],
                "tools": item.get("tools", [])
            })
        elif "user_content" in item:
            # Simple format - convert to messages
            # This matches Google's create_conversation function
            messages = [
                {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
                {"role": "user", "content": item["user_content"]},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": item["tool_name"],
                            "arguments": json.loads(item["tool_arguments"]) if isinstance(item["tool_arguments"], str) else item["tool_arguments"]
                        }
                    }]
                }
            ]
            processed.append({
                "messages": messages,
                "tools": item.get("tools", [])
            })

    return Dataset.from_list(processed)


def check_success_rate(model, tokenizer, dataset, tools):
    """
    Check model's tool calling success rate on dataset.

    Based on Google's check_success_rate function.
    """
    print("\n" + "=" * 60)
    print("Checking success rate...")
    print("=" * 60)

    success_count = 0
    total = len(dataset)

    for idx, item in enumerate(dataset):
        # Build prompt with only system and user message
        messages = [
            item["messages"][0],  # developer/system
            item["messages"][1],  # user
        ]

        # Get expected tool from the assistant's tool_calls
        expected_tool = None
        for msg in item["messages"]:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                if tool_calls:
                    expected_tool = tool_calls[0]["function"]["name"]
                break

        if not expected_tool:
            print(f"{idx+1}. Skipping - no expected tool found")
            continue

        # Generate response
        item_tools = item.get("tools", tools)
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=item_tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            out = model.generate(
                **inputs.to(model.device),
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128
            )

        output = tokenizer.decode(
            out[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=False
        )

        # Check if correct tool was called
        user_msg = item["messages"][1]["content"][:50]
        print(f"{idx+1}. \"{user_msg}...\"")
        print(f"   Expected: {expected_tool}")
        print(f"   Output: {output[:100]}...")

        if expected_tool in output:
            print("   -> Correct!")
            success_count += 1
        else:
            print("   -> Wrong")

    print(f"\nSuccess: {success_count}/{total} ({100*success_count/total:.1f}%)")
    return success_count / total if total > 0 else 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma for tool calling")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data JSON file")

    # Model arguments
    parser.add_argument("--model_id", type=str, default="google/functiongemma-270m-it",
                        help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="./functiongemma-finetuned",
                        help="Output directory for fine-tuned model")

    # Training arguments (Google's defaults)
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--test_split", type=float, default=0.5,
                        help="Test split ratio (Google uses 0.5 for demo)")

    # Other arguments
    parser.add_argument("--skip_eval_before", action="store_true",
                        help="Skip evaluation before training")
    parser.add_argument("--skip_eval_after", action="store_true",
                        help="Skip evaluation after training")

    args = parser.parse_args()

    print("=" * 60)
    print("FunctionGemma Fine-tuning (Google's Approach)")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)

    # Load data
    data = load_data(args.data_path)
    dataset = create_dataset(data)

    # Extract tools from first example for evaluation
    tools = dataset[0].get("tools", [])

    # Split dataset
    dataset = dataset.train_test_split(test_size=args.test_split, shuffle=True)
    print(f"Train: {len(dataset['train'])} examples")
    print(f"Test: {len(dataset['test'])} examples")

    # Load model and tokenizer
    print("\n" + "=" * 60)
    print("Loading model and tokenizer...")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Device: {model.device}")
    print(f"DType: {model.dtype}")

    # Debug: print formatted prompt
    print("\n--- Sample input ---")
    print(json.dumps(dataset["train"][0], indent=2, default=str)[:500])
    debug_msg = tokenizer.apply_chat_template(
        dataset["train"][0]["messages"],
        tools=dataset["train"][0].get("tools", tools),
        add_generation_prompt=False,
        tokenize=False
    )
    print("\n--- Formatted prompt ---")
    print(debug_msg[:1000])

    # Check success rate before training
    if not args.skip_eval_before:
        print("\n" + "=" * 60)
        print("BEFORE FINE-TUNING")
        print("=" * 60)
        check_success_rate(model, tokenizer, dataset["test"], tools)

    # Training configuration
    print("\n" + "=" * 60)
    print("Setting up training...")
    print("=" * 60)

    torch_dtype = model.dtype

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        packing=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        logging_steps=1,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        fp16=True if torch_dtype == torch.float16 else False,
        bf16=True if torch_dtype == torch.bfloat16 else False,
        lr_scheduler_type="constant",
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    trainer.train()

    # Save model
    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)

    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # Check success rate after training
    if not args.skip_eval_after:
        print("\n" + "=" * 60)
        print("AFTER FINE-TUNING")
        print("=" * 60)
        check_success_rate(model, tokenizer, dataset["test"], tools)

    # Print training metrics
    if trainer.state.log_history:
        print("\n" + "=" * 60)
        print("Training Metrics")
        print("=" * 60)

        train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        eval_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]

        if train_losses:
            print(f"Final train loss: {train_losses[-1]:.4f}")
        if eval_losses:
            print(f"Final eval loss: {eval_losses[-1]:.4f}")

    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
