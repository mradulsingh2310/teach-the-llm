"""
Test inference with fine-tuned FunctionGemma model.

Usage:
    python test_model.py --model_path ./functiongemma-finetuned
    python test_model.py --model_path ./functiongemma-finetuned --prompt "I need a 2 bedroom apartment"
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from tools import TOOLS, SYSTEM_PROMPT


def test_inference(model_path: str, prompt: str):
    """Run inference with the fine-tuned model."""
    print("=" * 60)
    print(f"Loading model from: {model_path}")
    print("=" * 60)

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_path)

    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )

    if device != "cuda":
        model = model.to(device)

    print(f"Model loaded on {device} with {dtype}")

    # Build messages
    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    print("\n" + "=" * 60)
    print(f"User: {prompt}")
    print("=" * 60)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.eos_token_id,
        )

    # Decode response (only the generated part)
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    print("\n" + "=" * 60)
    print("Assistant Response:")
    print("=" * 60)
    print(response)

    # Try to parse tool call if present
    if "tool_calls" in response.lower() or "function" in response.lower():
        print("\n" + "=" * 60)
        print("Detected tool call in response")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned FunctionGemma model")
    parser.add_argument("--model_path", type=str, default="./functiongemma-finetuned",
                        help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str,
                        default="Hey, I'm looking for a 2 bedroom apartment under $2000",
                        help="Test prompt")

    args = parser.parse_args()

    test_inference(args.model_path, args.prompt)


if __name__ == "__main__":
    main()
