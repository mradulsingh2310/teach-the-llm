"""
Fine-tune Models Project

This project contains model-specific agent implementations.

Available modules:
- gemma/  - FunctionGemma agent (270M, 2B models)
- agent/  - Shared tools and test framework

To run the FunctionGemma agent:
    cd gemma && python main.py

To run tests:
    cd gemma && pytest tests/ -v

Requirements:
    Start llama-server with a FunctionGemma model first:
    llama-server -hf unsloth/functiongemma-270m-it-GGUF:BF16
"""


def main():
    print("Fine-tune Models Project")
    print("=" * 40)
    print()
    print("Available agents:")
    print("  - gemma/main.py  (FunctionGemma agent)")
    print()
    print("To run the FunctionGemma agent:")
    print("  cd gemma && python main.py")
    print()
    print("Make sure llama-server is running with a FunctionGemma model:")
    print("  llama-server -hf unsloth/functiongemma-270m-it-GGUF:BF16")


if __name__ == "__main__":
    main()
