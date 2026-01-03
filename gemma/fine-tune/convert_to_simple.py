"""
Convert complex training data to Google's simple format.

Google's format:
{
    "user_content": "Show me 2 bedroom apartments",
    "tool_name": "get_available_listings",
    "tool_arguments": "{\"bedrooms\": [2]}"
}

This removes all the schema noise and focuses on:
user says X -> call tool Y with args Z
"""

import json
from pathlib import Path


def convert_example(example: dict) -> dict | None:
    """Convert a complex example to simple format."""
    messages = example.get("messages", [])

    # Find user message and assistant tool call
    user_content = None
    tool_name = None
    tool_arguments = None

    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")

        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            if tool_calls:
                tc = tool_calls[0]  # Take first tool call
                func = tc.get("function", {})
                tool_name = func.get("name")
                tool_arguments = json.dumps(func.get("arguments", {}))

    if user_content and tool_name:
        return {
            "user_content": user_content,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments
        }
    return None


def main():
    input_path = Path("training_data.json")
    output_path = Path("training_data_simple.json")

    print(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    print(f"Converting {len(data)} examples...")

    simple_data = []
    skipped = 0

    for ex in data:
        converted = convert_example(ex)
        if converted:
            simple_data.append(converted)
        else:
            skipped += 1

    print(f"Converted: {len(simple_data)}")
    print(f"Skipped (no tool calls): {skipped}")

    # Save
    with open(output_path, "w") as f:
        json.dump(simple_data, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Show sample
    print("\n=== Sample entries ===")
    for ex in simple_data[:3]:
        print(json.dumps(ex, indent=2))
        print()


if __name__ == "__main__":
    main()
