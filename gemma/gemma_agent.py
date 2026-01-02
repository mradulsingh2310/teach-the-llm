"""
FunctionGemma Agent Implementation using HuggingFace Transformers

Uses Google's EXACT pipeline from:
- https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma

Key insight: Google uses processor.apply_chat_template() which handles ALL the
<start_function_declaration>, <escape> tokens, etc. automatically. Manual string
building leads to tokenization mismatches.
"""

import json
import re
import time
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import torch

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.lead_tools import clear_session_leads
from agent.conversation_logger import ConversationLogger
from agent.models import ConversationTurn, ToolCall, Usage

# Import the actual tool implementations (we'll call these directly)
from agent.tools.listing_tools import get_available_listings as _get_available_listings
from agent.tools.scheduling_tools import get_availability as _get_availability, create_appointment as _create_appointment
from agent.tools.handoff_tools import escalate_conversation as _escalate_conversation
from agent.tools.knowledge_base_tools import search_property_knowledge as _search_property_knowledge
from agent.tools.lead_tools import create_lead as _create_lead


# =============================================================================
# Plain Python functions for FunctionGemma
#
# Google's processor.apply_chat_template() parses Python function signatures
# and docstrings. We create clean wrapper functions that:
# 1. Have clear type hints
# 2. Have well-structured docstrings
# 3. Call our actual LangChain tool implementations
# =============================================================================

def get_available_listings(bedrooms: list[int], max_rent: float = None) -> dict:
    """
    Search for available apartment listings.

    Args:
        bedrooms: List of bedroom counts to search for. Use [0] for studios, [1] for 1BR, [2] for 2BR, etc.
        max_rent: Maximum monthly rent in dollars. Optional filter.

    Returns:
        listings: List of available apartments matching criteria
        total_count: Total number of matches found
        message: Status message
    """
    return _get_available_listings.invoke({"bedrooms": bedrooms, "max_rent": max_rent})


def get_availability(date_str: str) -> dict:
    """
    Get available tour time slots for a specific date.

    Args:
        date_str: Date to check availability for, in YYYY-MM-DD format (e.g., "2025-02-15")

    Returns:
        available_slots: List of available time slots
        formatted_date: Human-readable date
        message: Status message
    """
    return _get_availability.invoke({"date_str": date_str})


def create_appointment(
    appointment_date: str,
    appointment_start_time: str,
    first_name: str,
    email: str,
    phone: str,
    last_name: str = None
) -> dict:
    """
    Book a property tour appointment.

    Args:
        appointment_date: Tour date in YYYY-MM-DD format (e.g., "2025-02-15")
        appointment_start_time: Tour time in HH:MM 24-hour format (e.g., "10:00" or "14:30")
        first_name: Visitor's first name
        email: Visitor's email address for confirmation
        phone: Visitor's phone number
        last_name: Visitor's last name (optional)

    Returns:
        confirmation_id: Unique confirmation number
        appointment_details: Date, time, and duration info
        message: Confirmation message
    """
    return _create_appointment.invoke({
        "appointment_date": appointment_date,
        "appointment_start_time": appointment_start_time,
        "first_name": first_name,
        "email": email,
        "phone": phone,
        "last_name": last_name
    })


def escalate_conversation(reason_for_escalation: str) -> dict:
    """
    Transfer the conversation to a human leasing agent.

    Args:
        reason_for_escalation: Brief description of why human assistance is needed

    Returns:
        ticket_id: Escalation ticket number
        message: Confirmation that conversation was escalated
        expected_response_time: When to expect a response
    """
    return _escalate_conversation.invoke({"reason_for_escalation": reason_for_escalation})


def search_property_knowledge(search_text: str, category: str = None) -> dict:
    """
    Search property information about amenities, policies, and services.

    Args:
        search_text: What to search for (e.g., "pet policy", "parking", "gym hours")
        category: Optional filter - one of: amenities, neighborhood, leasing, rent_payment, policies, utilities, maintenance, move_in_out

    Returns:
        results: List of relevant knowledge base entries
        total_results: Number of matches found
        message: Status message
    """
    return _search_property_knowledge.invoke({"search_text": search_text, "category": category})


def create_lead(email: str, phone: str, first_name: str, last_name: str) -> dict:
    """
    Save prospect contact information for follow-up.

    Args:
        email: Prospect's email address
        phone: Prospect's phone number
        first_name: Prospect's first name
        last_name: Prospect's last name

    Returns:
        lead_id: Unique lead identifier
        status: Whether lead was created or already exists
        message: Confirmation message
    """
    return _create_lead.invoke({
        "email": email,
        "phone": phone,
        "first_name": first_name,
        "last_name": last_name
    })


# All tools as plain Python functions
ALL_TOOL_FUNCTIONS = [
    get_available_listings,
    get_availability,
    create_appointment,
    escalate_conversation,
    search_property_knowledge,
    create_lead,
]

# Tool name to function mapping
TOOL_MAPPING: Dict[str, Callable] = {
    func.__name__: func for func in ALL_TOOL_FUNCTIONS
}


def extract_tool_calls(text: str) -> list[dict]:
    """
    Extract function calls from FunctionGemma output.

    Expected format:
    <start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>
    """
    def cast(v: str) -> Any:
        """Cast string value to appropriate Python type."""
        v = v.strip()
        # Try integer
        try:
            return int(v)
        except ValueError:
            pass
        # Try float
        try:
            return float(v)
        except ValueError:
            pass
        # Try boolean
        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        # Try JSON (for lists/dicts)
        if v.startswith('[') or v.startswith('{'):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        # Return as string
        return v.strip("'\"")

    results = []

    # Pattern: <start_function_call>call:NAME{args}<end_function_call>
    # Also handle truncated calls (stop sequence may cut off end token)
    pattern = r"<start_function_call>call:(\w+)\{(.*?)\}(?:<end_function_call>|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    for name, args_str in matches:
        arguments = {}
        # Parse: key:<escape>value<escape> or key:value
        arg_pattern = r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))"
        for key, escaped_val, plain_val in re.findall(arg_pattern, args_str):
            value = escaped_val if escaped_val else plain_val
            if value:
                arguments[key] = cast(value.strip())

        results.append({
            "name": name,
            "arguments": arguments
        })

    return results


class GemmaAgent:
    """
    Property Agent using HuggingFace Transformers with FunctionGemma.

    This implementation uses Google's exact pipeline:
    - AutoProcessor for tokenization and chat template formatting
    - AutoModelForCausalLM for generation
    - processor.apply_chat_template() handles all special token formatting
    """

    def __init__(
        self,
        model_id: str = "google/functiongemma-270m-it",
        agent_name: str = "Aria",
        property_name: str = "Sunset Apartments",
        log_file: str = "conversations.json",
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        device: str = None,
        torch_dtype: str = "auto",
    ):
        """
        Initialize the GemmaAgent with HuggingFace Transformers.

        Args:
            model_id: HuggingFace model ID (default: google/functiongemma-270m-it)
            agent_name: Name of the assistant
            property_name: Name of the property
            log_file: Path to conversation log file
            temperature: Generation temperature (0.0-1.0)
            max_new_tokens: Maximum tokens to generate per turn
            device: Device to use ('cuda', 'cpu', or None for auto)
            torch_dtype: Torch dtype ('auto', 'float16', 'bfloat16', 'float32')
        """
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.model_id = model_id
        self.agent_name = agent_name
        self.property_name = property_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Determine dtype
        if torch_dtype == "auto":
            if self.device == "cuda":
                self._torch_dtype = torch.float16
            else:
                self._torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self._torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self._torch_dtype = torch.bfloat16
        else:
            self._torch_dtype = torch.float32

        print(f"Loading FunctionGemma model: {model_id}")
        print(f"Device: {self.device}, Dtype: {self._torch_dtype}")

        # Load processor and model using Google's exact approach
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self._torch_dtype,
            device_map=self.device if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print(f"Model loaded successfully!")

        # Tools - plain Python functions
        self.tools = ALL_TOOL_FUNCTIONS
        self.tool_mapping = TOOL_MAPPING

        # Conversation state
        self._messages: List[Dict[str, Any]] = []

        # Logging
        self.logger = ConversationLogger(log_file)

    def _generate(self, add_generation_prompt: bool = True) -> str:
        """
        Generate a response using the current message history.

        Uses processor.apply_chat_template() which handles ALL formatting:
        - Converts Python functions to <start_function_declaration>...<end_function_declaration>
        - Adds <escape> tokens correctly
        - Formats <start_of_turn>developer/user/model<end_of_turn> correctly

        Returns:
            Generated text from the model
        """
        # Apply chat template - this is the KEY step that Google's pipeline uses
        inputs = self.processor.apply_chat_template(
            self._messages,
            tools=self.tools,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.processor.eos_token_id,
            )

        # Decode only the new tokens (not the prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        return output_text

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        if tool_name not in self.tool_mapping:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            func = self.tool_mapping[tool_name]
            result = func(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}

    def start_session(self) -> str:
        """Start a new conversation session."""
        self._messages = []
        clear_session_leads()
        return self.logger.create_session(
            model_id=self.model_id,
            agent_type="functiongemma_agent",
            channel="cli",
        )

    def process_message(self, user_input: str, session_id: str, debug: bool = False) -> str:
        """
        Process a user message with function calling loop.

        Follows Google's exact conversation flow:
        1. Developer message activates function calling
        2. User provides query
        3. Model generates function call OR text response
        4. If function call: execute, add result, let model continue
        5. Return final text response

        Args:
            user_input: User's message
            session_id: Session ID for logging
            debug: If True, print debug information

        Returns:
            Agent's text response
        """
        # Initialize conversation with developer message if empty
        # This is CRITICAL - it activates function calling mode
        if not self._messages:
            self._messages.append({
                "role": "developer",
                "content": "You are a model that can do function calling with the following functions"
            })

        # Add user message
        self._messages.append({
            "role": "user",
            "content": user_input
        })

        # Log user turn
        self.logger.add_turn(session_id, ConversationTurn(
            turn_id=self.logger.get_next_turn_id(session_id),
            role="user",
            content=user_input,
        ))

        # Agent loop - allow multiple tool calls
        max_iterations = 5
        final_response = ""

        for iteration in range(max_iterations):
            start_time = time.time()

            if debug:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}")
                print(f"{'='*60}")
                print(f"Messages: {len(self._messages)}")

            try:
                output_text = self._generate()
            except Exception as e:
                self.logger.add_error(session_id, "GenerationError", str(e))
                raise RuntimeError(f"Generation error: {e}") from e

            latency_ms = int((time.time() - start_time) * 1000)

            if debug:
                print(f"\n--- RAW MODEL OUTPUT ---")
                print(output_text)
                print(f"--- END RAW OUTPUT ---")
                print(f"Latency: {latency_ms}ms\n")

            # Extract function calls
            function_calls = extract_tool_calls(output_text)

            if debug:
                if function_calls:
                    print(f"--- DETECTED FUNCTION CALLS ---")
                    for fc in function_calls:
                        print(f"  Tool: {fc['name']}")
                        print(f"  Args: {fc['arguments']}")
                    print()
                else:
                    print("--- NO FUNCTION CALLS DETECTED ---\n")

            # Clean response (remove function call tokens)
            clean_response = re.sub(
                r"<start_function_call>.*?(<end_function_call>|$)",
                "", output_text, flags=re.DOTALL
            ).strip()

            logged_tools: List[ToolCall] = []

            if function_calls:
                # Process each function call
                tool_results = []

                for i, fc in enumerate(function_calls):
                    tool_name = fc["name"]
                    tool_args = fc["arguments"]

                    if debug:
                        print(f"--- EXECUTING TOOL ---")
                        print(f"  Tool: {tool_name}")
                        print(f"  Args: {tool_args}")

                    # Execute tool
                    result = self._execute_tool(tool_name, tool_args)
                    status = "error" if "error" in result else "success"

                    if debug:
                        print(f"  Status: {status}")
                        print(f"  Result: {json.dumps(result, indent=2)[:500]}...")
                        print()

                    # Log tool call
                    logged_tools.append(ToolCall(
                        tool_name=tool_name,
                        tool_use_id=f"call_{iteration}_{i}",
                        input=tool_args,
                        output=result,
                        status=status,
                        error_message=result.get("error") if status == "error" else None,
                    ))

                    # Collect result for message
                    tool_results.append({
                        "name": tool_name,
                        "response": result
                    })

                # Add assistant message with tool calls (Google's format)
                self._messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {"type": "function", "function": fc}
                        for fc in function_calls
                    ]
                })

                # Add tool results (Google's format)
                # For single result, use dict; for multiple, use list
                if len(tool_results) == 1:
                    tool_content = tool_results[0]
                else:
                    tool_content = tool_results

                self._messages.append({
                    "role": "tool",
                    "content": tool_content
                })

            # Log assistant turn
            self.logger.add_turn(session_id, ConversationTurn(
                turn_id=self.logger.get_next_turn_id(session_id),
                role="assistant",
                content=clean_response,
                stop_reason="tool_use" if function_calls else "end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                latency_ms=latency_ms,
                tools_called=logged_tools,
            ))

            # If no function calls, we're done
            if not function_calls:
                final_response = clean_response
                # Add final response to message history
                self._messages.append({
                    "role": "assistant",
                    "content": output_text
                })
                break

            # If we have a text response along with tool calls, save it
            if clean_response:
                final_response = clean_response

        return final_response or "I'm having trouble processing your request. Please try again."

    def end_session(self, session_id: str, status: str = "completed") -> None:
        """End a conversation session."""
        self._messages = []
        clear_session_leads()
        self.logger.end_session(session_id, status)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self._messages.copy()

    def reset_conversation(self) -> None:
        """Reset conversation history without ending session."""
        self._messages = []


# =============================================================================
# Convenience function for quick testing
# =============================================================================

def create_agent(**kwargs) -> GemmaAgent:
    """Create a GemmaAgent with default settings."""
    return GemmaAgent(**kwargs)


if __name__ == "__main__":
    # Quick test
    print("Initializing FunctionGemma Agent...")
    agent = GemmaAgent()

    session_id = agent.start_session()
    print(f"Session started: {session_id}\n")

    # Test query
    test_query = "Show me 2 bedroom apartments"
    print(f"User: {test_query}")

    response = agent.process_message(test_query, session_id, debug=True)
    print(f"\nAgent: {response}")

    agent.end_session(session_id)
