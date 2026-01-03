"""Debug script to see what the training data looks like after processing."""
import json
from transformers import AutoProcessor

from train import convert_simple_to_messages, load_dataset_from_json
from tools import TOOLS, SYSTEM_PROMPT

# Load a few examples
with open("./training_data.json") as f:
    data = json.load(f)

print("=" * 60)
print("RAW TRAINING DATA (first example):")
print("=" * 60)
print(json.dumps(data[0], indent=2))

# Convert to messages format
converted = convert_simple_to_messages(data[0])
print("\n" + "=" * 60)
print("CONVERTED TO MESSAGES FORMAT:")
print("=" * 60)
print(json.dumps(converted, indent=2, default=str))

# Load processor and apply chat template
processor = AutoProcessor.from_pretrained("google/functiongemma-270m-it")

# Apply chat template
text = processor.apply_chat_template(
    converted["messages"],
    tools=converted.get("tools", TOOLS),
    add_generation_prompt=False,
    tokenize=False,
)

print("\n" + "=" * 60)
print("AFTER CHAT TEMPLATE (what model sees):")
print("=" * 60)
print(text)

print("\n" + "=" * 60)
print("KEY MARKERS TO LOOK FOR:")
print("=" * 60)
print("- <start_of_turn>model should precede the assistant response")
print("- <start_function_call> should be the tool call, not declarations")
print("- Tool declarations should be BEFORE user message")
