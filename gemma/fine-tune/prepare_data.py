"""
Data Preparation Script for FunctionGemma Fine-tuning

This script:
1. Loads generated training data from multiple sources
2. Validates and cleans the data
3. Converts to the format required by FunctionGemma
4. Splits into train/eval sets
5. Saves to disk for training
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from datasets import Dataset
from tools import TOOLS, SYSTEM_PROMPT


def validate_tool_call(tool_call: Dict[str, Any], available_tools: List[str]) -> bool:
    """Validate a tool call has correct structure and references valid tool."""
    if not isinstance(tool_call, dict):
        return False

    if "type" not in tool_call or tool_call["type"] != "function":
        return False

    func = tool_call.get("function", {})
    if not isinstance(func, dict):
        return False

    name = func.get("name")
    if name not in available_tools:
        return False

    args = func.get("arguments")
    if args is not None and not isinstance(args, dict):
        return False

    return True


def validate_message(message: Dict[str, Any], available_tools: List[str]) -> bool:
    """Validate a single message in a conversation."""
    if not isinstance(message, dict):
        return False

    role = message.get("role")
    if role not in ["developer", "user", "assistant", "tool"]:
        return False

    # Check content or tool_calls
    if role == "assistant":
        has_content = "content" in message and message["content"]
        has_tool_calls = "tool_calls" in message and message["tool_calls"]

        if not has_content and not has_tool_calls:
            return False

        # Validate tool calls if present
        if has_tool_calls:
            for tc in message["tool_calls"]:
                if not validate_tool_call(tc, available_tools):
                    return False
    elif role in ["user", "developer"]:
        if "content" not in message:
            return False

    return True


def validate_example(example: Dict[str, Any]) -> bool:
    """Validate a complete training example."""
    if not isinstance(example, dict):
        return False

    messages = example.get("messages", [])
    if not messages or len(messages) < 2:
        return False

    # Get available tool names
    tools = example.get("tools", TOOLS)
    available_tools = [t["function"]["name"] for t in tools]

    # Validate each message
    for msg in messages:
        if not validate_message(msg, available_tools):
            return False

    # Check first message is developer role
    if messages[0].get("role") != "developer":
        return False

    return True


def fix_common_issues(example: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common issues in training examples."""
    fixed = example.copy()
    messages = fixed.get("messages", [])

    for msg in messages:
        # Ensure developer message has correct content
        if msg.get("role") == "developer":
            msg["content"] = SYSTEM_PROMPT

        # Fix assistant tool calls
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                if "type" not in tc:
                    tc["type"] = "function"

                func = tc.get("function", {})

                # Fix bedrooms parameter - ensure it's a list
                args = func.get("arguments", {})
                if "bedrooms" in args:
                    bedrooms = args["bedrooms"]
                    if isinstance(bedrooms, int):
                        args["bedrooms"] = [bedrooms]
                    elif isinstance(bedrooms, str):
                        # Try to parse as list
                        try:
                            if bedrooms.startswith("["):
                                args["bedrooms"] = json.loads(bedrooms)
                            else:
                                args["bedrooms"] = [int(bedrooms)]
                        except (json.JSONDecodeError, ValueError):
                            args["bedrooms"] = [0]  # Default to studio

                # Fix phone parameter - ensure it's a string
                if "phone" in args and not isinstance(args["phone"], str):
                    args["phone"] = str(args["phone"])

    # Add tools if not present
    if "tools" not in fixed:
        fixed["tools"] = TOOLS

    return fixed


def load_training_data(data_paths: List[str]) -> List[Dict[str, Any]]:
    """Load training data from multiple JSON files."""
    all_data = []

    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping...")
            continue

        print(f"Loading {path}...")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Try parsing as JSON array
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict) and "data" in data:
                    all_data.extend(data["data"])
                else:
                    all_data.append(data)
            except json.JSONDecodeError:
                # Try parsing as Python literal (from agent output)
                if "training_data = [" in content:
                    # Extract the list
                    start = content.find("training_data = [")
                    if start >= 0:
                        # Find matching bracket
                        bracket_count = 0
                        in_string = False
                        escape_next = False
                        end = start + len("training_data = ")

                        for i, char in enumerate(content[end:], start=end):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == "\\":
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                            if not in_string:
                                if char == "[":
                                    bracket_count += 1
                                elif char == "]":
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end = i + 1
                                        break

                        list_str = content[end:end + 1] if end > start else "[]"
                        # This is simplified - in practice we'd need safer parsing
                        print(f"  Found training_data list in {path}")

            print(f"  Loaded {len(all_data)} total examples so far")

        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    return all_data


def prepare_dataset(
    data: List[Dict[str, Any]],
    eval_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> tuple:
    """
    Prepare training and evaluation datasets.

    Args:
        data: List of training examples
        eval_split: Fraction of data for evaluation
        shuffle: Whether to shuffle data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print(f"\nPreparing dataset from {len(data)} examples...")

    # Validate and fix examples
    valid_data = []
    invalid_count = 0

    for i, example in enumerate(data):
        # Fix common issues
        fixed = fix_common_issues(example)

        # Validate
        if validate_example(fixed):
            valid_data.append(fixed)
        else:
            invalid_count += 1
            if invalid_count <= 5:
                print(f"  Invalid example {i}: {str(example)[:100]}...")

    print(f"  Valid examples: {len(valid_data)}")
    print(f"  Invalid examples: {invalid_count}")

    if not valid_data:
        raise ValueError("No valid training examples found!")

    # Shuffle
    if shuffle:
        random.seed(seed)
        random.shuffle(valid_data)

    # Split
    split_idx = int(len(valid_data) * (1 - eval_split))
    train_data = valid_data[:split_idx]
    eval_data = valid_data[split_idx:]

    print(f"  Train examples: {len(train_data)}")
    print(f"  Eval examples: {len(eval_data)}")

    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    return train_dataset, eval_dataset


def save_dataset(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str = "."
) -> None:
    """Save datasets to disk."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_dataset")
    eval_path = os.path.join(output_dir, "eval_dataset")

    print(f"\nSaving datasets...")
    train_dataset.save_to_disk(train_path)
    eval_dataset.save_to_disk(eval_path)

    # Also save as JSON for inspection
    with open(os.path.join(output_dir, "train_data.json"), "w") as f:
        json.dump([ex for ex in train_dataset], f, indent=2)

    with open(os.path.join(output_dir, "eval_data.json"), "w") as f:
        json.dump([ex for ex in eval_dataset], f, indent=2)

    print(f"  Saved to {output_dir}")


def main():
    """Main data preparation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for FunctionGemma fine-tuning")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input data files")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Evaluation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle data")

    args = parser.parse_args()

    # Load data
    data = load_training_data(args.input)

    if not data:
        print("No data loaded! Check input files.")
        return

    # Prepare datasets
    train_ds, eval_ds = prepare_dataset(
        data,
        eval_split=args.eval_split,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )

    # Save
    save_dataset(train_ds, eval_ds, args.output)

    print("\nData preparation complete!")
    print(f"  Total examples: {len(train_ds) + len(eval_ds)}")
    print(f"  Train: {len(train_ds)}")
    print(f"  Eval: {len(eval_ds)}")


if __name__ == "__main__":
    main()
