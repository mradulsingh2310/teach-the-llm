#!/usr/bin/env python3
"""
FunctionGemma Property Agent CLI

A command-line interface for interacting with the Property Agent
using FunctionGemma models via HuggingFace Transformers.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for agent package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gemma_agent import GemmaAgent
import config


def read_multiline_input(prompt: str = "You: ") -> str:
    """Read input until user submits with two consecutive blank lines."""
    print(prompt, end="", flush=True)
    lines = []
    consecutive_blanks = 0

    while True:
        line = sys.stdin.readline()
        if not line:  # EOF
            raise EOFError()

        line = line.rstrip('\n')

        if line == "":
            consecutive_blanks += 1
            if consecutive_blanks >= 2 and lines:
                while lines and lines[-1] == "":
                    lines.pop()
                break
            if not lines:
                consecutive_blanks = 0
                print(prompt, end="", flush=True)
                continue
            lines.append(line)
        else:
            consecutive_blanks = 0
            lines.append(line)

    return '\n'.join(lines)


def print_banner():
    """Print the welcome banner."""
    print("\n" + "=" * 60)
    print(f"  Property Agent - {config.MODEL_NAME}")
    print("  (HuggingFace Transformers)")
    print("=" * 60)
    print("\nWelcome! I'm here to help you find your perfect home.")
    print("(Type/paste message, then press Enter 3x to send)")
    print("\nCommands: quit | end | new | help | debug")
    print("-" * 60 + "\n")


def print_help():
    """Print help information."""
    print("\n" + "-" * 40)
    print("Available Commands:")
    print("  quit  - Exit the application")
    print("  end   - End the current session normally")
    print("  new   - Start a new conversation session")
    print("  help  - Show this help message")
    print("  debug - Toggle debug mode")
    print("\nHow to send messages:")
    print("  Type or paste, then press Enter 3 times")
    print("  (two blank lines in a row to submit)")
    print("-" * 40 + "\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description=f"Property Agent CLI ({config.MODEL_NAME})"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=config.MODEL_ID,
        help="HuggingFace model ID (e.g., google/functiongemma-270m-it)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.MAX_NEW_TOKENS,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device to use (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=config.TORCH_DTYPE,
        help="Torch dtype (auto, float16, bfloat16, float32)",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default=config.AGENT_NAME,
        help="Name of the AI assistant",
    )
    parser.add_argument(
        "--property-name",
        type=str,
        default=config.PROPERTY_NAME,
        help="Name of the property",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=config.LOG_FILE,
        help="Path to the conversation log file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (shows raw model output)",
    )

    args = parser.parse_args()

    # Print welcome banner
    print_banner()
    print(f"Loading model: {args.model_id}")
    print("This may take a moment on first run (downloading model)...")
    print()

    # Initialize the Gemma-specific agent
    try:
        agent = GemmaAgent(
            model_id=args.model_id,
            agent_name=args.agent_name,
            property_name=args.property_name,
            log_file=args.log_file,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            torch_dtype=args.dtype,
        )
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Start a session
    session_id = agent.start_session()
    print(f"\nModel loaded successfully!")
    print(f"Session started: {session_id[:8]}...")
    print(f"Agent: {args.agent_name} | Property: {args.property_name}")
    print(f"Model: {args.model_id}")
    print(f"Temperature: {args.temperature} | Max tokens: {args.max_new_tokens}")
    print("-" * 60 + "\n")

    debug_mode = args.debug

    try:
        while True:
            try:
                user_input = read_multiline_input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            lower_input = user_input.lower()

            if lower_input == "quit":
                agent.end_session(session_id, "abandoned")
                print("\nSession abandoned. Goodbye!")
                break

            if lower_input == "end":
                agent.end_session(session_id, "completed")
                print("\nSession completed. Goodbye!")
                break

            if lower_input == "new":
                agent.end_session(session_id, "completed")
                session_id = agent.start_session()
                print(f"\nNew session started: {session_id[:8]}...")
                print("-" * 60 + "\n")
                continue

            if lower_input == "help":
                print_help()
                continue

            if lower_input == "debug":
                debug_mode = not debug_mode
                print(f"\nDebug mode: {'ON' if debug_mode else 'OFF'}\n")
                continue

            try:
                response = agent.process_message(user_input, session_id, debug=debug_mode)
                print(f"\n{args.agent_name}: {response}\n")
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                print("The conversation will continue. Please try again.\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Ending session...")
        agent.end_session(session_id, "abandoned")
        print("Session abandoned. Goodbye!")

    print(f"\nConversation log saved to: {args.log_file}")


if __name__ == "__main__":
    main()
