#!/usr/bin/env python3
"""
Quick test script for the HuggingFace Transformers-based GemmaAgent.

Run this to verify the new implementation works correctly.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from gemma_agent import GemmaAgent


def main():
    print("=" * 70)
    print("FunctionGemma Agent - Quick Test")
    print("Using HuggingFace Transformers (proper tokenization)")
    print("=" * 70)
    print()

    # Initialize agent
    print("Loading model... (this may take a moment)")
    agent = GemmaAgent(
        model_id="google/functiongemma-270m-it",
        temperature=0.3,
        max_new_tokens=256,
    )
    print()

    # Start session
    session_id = agent.start_session()
    print(f"Session started: {session_id}")
    print()

    # Test queries
    test_queries = [
        "Show me 2 bedroom apartments",
        "What times are available on 2025-02-15?",
        "What is the pet policy?",
    ]

    for query in test_queries:
        print("-" * 70)
        print(f"USER: {query}")
        print("-" * 70)

        response = agent.process_message(query, session_id, debug=True)

        print()
        print(f"AGENT: {response}")
        print()

        # Reset conversation for next independent test
        agent.reset_conversation()

    # End session
    agent.end_session(session_id)
    print("=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
