"""Property Agent using LangChain + ChatOpenAI via OpenAI-compatible server."""

import json
import re
import time
from typing import Any, Callable, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from .conversation_logger import ConversationLogger
from .models import ConversationTurn, ToolCall, Usage

# Import tools from the tools module
from .tools import (
    create_appointment,
    create_lead,
    escalate_conversation,
    get_availability,
    get_available_listings,
    search_property_knowledge,
    ALL_TOOLS,
)

# Import lead session management
from .tools.lead_tools import clear_session_leads, get_session_leads

# Import prompt renderer
from .prompt.agent_prompt import render_property_agent_prompt

# Type for prompt format
PromptFormat = Literal["default", "functiongemma"]


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from FunctionGemma output.

    FunctionGemma uses the format:
    <start_function_call>call:function_name{param1:<escape>value1<escape>,param2:<escape>value2<escape>}<end_function_call>

    Args:
        text: The model output text to parse.

    Returns:
        List of tool call dictionaries with 'name' and 'arguments' keys.
    """
    def cast(v: str) -> Any:
        """Cast string value to appropriate Python type."""
        v = v.strip()
        # Try integer first
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
        # Handle list values like [1, 2] or <escape>[1, 2]<escape>
        clean_v = v.strip("<escape>").strip()
        if clean_v.startswith('[') and clean_v.endswith(']'):
            try:
                return json.loads(clean_v)
            except json.JSONDecodeError:
                pass
        # Return as string, stripping quotes and escape tags
        return v.strip("'\"").replace("<escape>", "")

    results = []

    # Handle both complete and truncated function calls (stop sequence may cut off end token)
    # Pattern 1: Complete with end token
    # Pattern 2: Truncated (stop sequence removed end token)
    patterns = [
        r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>",
        r"<start_function_call>call:(\w+)\{(.*?)\}$",  # Truncated at end
        r"<start_function_call>call:(\w+)\{(.*?)\}\s*$",  # With trailing whitespace
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            break

    for name, args_str in matches:
        arguments = {}
        # Parse arguments - handle both escaped and non-escaped values
        # Pattern: key:<escape>value<escape> or key:value
        arg_pattern = r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))"
        arg_matches = re.findall(arg_pattern, args_str)

        for key, escaped_val, plain_val in arg_matches:
            # Use escaped value if present, otherwise plain value
            value = escaped_val if escaped_val else plain_val
            if value:
                arguments[key] = cast(value.strip())

        results.append({
            "name": name,
            "arguments": arguments
        })

    return results


def get_tool_json_schema(tool: Any) -> dict[str, Any]:
    """Convert a LangChain tool to JSON Schema format.

    Args:
        tool: A LangChain tool object.

    Returns:
        JSON Schema representation of the tool.
    """
    # Get the input schema from the tool (using Pydantic V2 API)
    schema = tool.args_schema.model_json_schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}

    # Build the function definition
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    }


def build_tools_prompt(tools: list[Any]) -> str:
    """Build the tools section of the system prompt for text-based function calling.

    Args:
        tools: List of LangChain tools.

    Returns:
        Formatted string with tool definitions.
    """
    tool_schemas = [get_tool_json_schema(tool) for tool in tools]
    return json.dumps(tool_schemas, indent=2)


def format_functiongemma_tool_declaration(tool: Any, minimal: bool = True) -> str:
    """Format a LangChain tool as a FunctionGemma tool declaration.

    Args:
        tool: A LangChain tool object.
        minimal: If True, use ultra-short format for small models.

    Returns:
        Formatted tool declaration string.
    """
    schema = {}
    if hasattr(tool, 'args_schema') and tool.args_schema:
        schema = tool.args_schema.model_json_schema()

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    if minimal:
        # Ultra-minimal format for 270M model
        # Just param names and types, no descriptions
        props_parts = []
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get("type", "string").upper()
            props_parts.append(f"{prop_name}:{prop_type}")
        params_str = ",".join(props_parts)

        # Short description (first sentence only, max 50 chars)
        short_desc = tool.description.split('.')[0][:50]

        return f"{tool.name}({params_str}) - {short_desc}"
    else:
        # Full FunctionGemma format
        props_parts = []
        for prop_name, prop_info in properties.items():
            prop_desc = prop_info.get("description", "")
            prop_type = prop_info.get("type", "string").upper()
            props_parts.append(
                f"{prop_name}:{{description:<escape>{prop_desc}<escape>,type:<escape>{prop_type}<escape>}}"
            )

        properties_str = ",".join(props_parts)
        required_str = ",".join(f"<escape>{r}<escape>" for r in required)

        declaration = (
            f"<start_function_declaration>declaration:{tool.name}{{"
            f"description:<escape>{tool.description}<escape>,"
            f"parameters:{{properties:{{{properties_str}}},required:[{required_str}],type:<escape>OBJECT<escape>}}"
            f"}}<end_function_declaration>"
        )

        return declaration


def format_functiongemma_response(tool_name: str, response: Any) -> str:
    """Format a tool response for FunctionGemma.

    Args:
        tool_name: Name of the tool that was called.
        response: The response from the tool (dict or string).

    Returns:
        Formatted function response string.
    """
    if isinstance(response, dict):
        parts = []
        for key, value in response.items():
            if isinstance(value, str):
                parts.append(f"{key}:<escape>{value}<escape>")
            elif isinstance(value, (list, dict)):
                parts.append(f"{key}:<escape>{json.dumps(value)}<escape>")
            else:
                parts.append(f"{key}:{value}")
        response_str = ",".join(parts)
    else:
        response_str = f"value:<escape>{response}<escape>"

    return f"<start_function_response>response:{tool_name}{{{response_str}}}<end_function_response>"


# Tool calling mode type
ToolCallingMode = Literal["native", "text_based"]


class PropertyAgent:
    """Property Agent that uses LangChain + ChatOpenAI with OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model_id: str = "local-model",
        agent_name: str = "Aria",
        property_name: str = "Sunset Apartments",
        log_file: str = "conversations.json",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        tool_calling_mode: ToolCallingMode = "text_based",
        prompt_format: PromptFormat = "default",
    ):
        """Initialize the PropertyAgent.

        Args:
            base_url: Base URL for the OpenAI-compatible API server.
            model_id: Model identifier for logging purposes.
            agent_name: Name of the AI assistant.
            property_name: Name of the property.
            log_file: Path to the conversation log file.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.
            tool_calling_mode: How to handle tool calling:
                - "native": Use OpenAI-style bind_tools() for models like Llama 3.1,
                            Qwen 2.5, Mistral, etc. Requires server started with --jinja.
                - "text_based": Parse tool calls from text output for models like
                                FunctionGemma that use custom formats.
            prompt_format: Which prompt format to use:
                - "default": Full prompt with JSON tool schemas (for larger models).
                - "functiongemma": Optimized short prompt with FunctionGemma's
                                   declaration format (for FunctionGemma 270M).
        """
        self.base_url = base_url
        self.model_id = model_id
        self.agent_name = agent_name
        self.property_name = property_name
        self.temperature = temperature
        self.tool_calling_mode = tool_calling_mode
        self.prompt_format = prompt_format

        # Initialize ChatOpenAI pointing to OpenAI-compatible server
        base_llm = ChatOpenAI(
            base_url=base_url,
            api_key="not-needed",  # Local server doesn't require an API key
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.logger = ConversationLogger(log_file)

        # Available tools list
        self._tools = ALL_TOOLS

        # Build tool mapping (name -> function)
        self.tool_mapping: dict[str, Callable] = {
            tool.name: tool for tool in self._tools
        }

        # Set up LLM based on tool calling mode
        if tool_calling_mode == "native":
            # Bind tools for models with native OpenAI-style function calling
            self.llm = base_llm.bind_tools(self._tools)
        elif prompt_format == "functiongemma":
            # FunctionGemma: add stop sequences to prevent repetition
            self.llm = base_llm.bind(stop=["<end_function_call>", "\n\nUser:", "\n\nYou:"])
        else:
            # Use base LLM for text-based tool calling
            self.llm = base_llm

        # Conversation history for the current session
        self._conversation_history: list[dict[str, str]] = []

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Input parameters for the tool.

        Returns:
            Dictionary containing the tool result or error.
        """
        if tool_name not in self.tool_mapping:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            tool_func = self.tool_mapping[tool_name]
            result = tool_func.invoke(tool_input)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def _build_system_prompt(self, base_prompt: str) -> str:
        """Build the complete system prompt.

        Args:
            base_prompt: The base agent prompt (used for 'default' format).

        Returns:
            Complete system prompt (with tool definitions for text-based mode).
        """
        if self.tool_calling_mode == "native":
            # Native mode: tools are handled by bind_tools()
            return base_prompt

        # Text-based mode: include tool definitions in prompt
        if self.prompt_format == "functiongemma":
            # Ultra-minimal prompt for FunctionGemma 270M
            tool_list = "\n".join(
                format_functiongemma_tool_declaration(tool, minimal=True) for tool in self._tools
            )
            return f"""You are {self.agent_name}, a leasing assistant. You can call these functions:

{tool_list}

To call a function: <start_function_call>call:name{{param:<escape>value<escape>}}<end_function_call>

Examples:
- "2 bedroom" -> call:get_available_listings{{bedrooms:<escape>[2]<escape>}}
- "schedule tour" -> first ask for date, then call get_availability"""
        else:
            # Default: Full prompt with JSON schemas
            tools_json = build_tools_prompt(self._tools)
            return f"""You are a model that can do function calling with the following functions:

{tools_json}

When you need to use a tool, respond with:
<start_function_call>call:function_name{{param1:<escape>value1<escape>,param2:<escape>value2<escape>}}<end_function_call>

---

{base_prompt}"""

    def start_session(self) -> str:
        """Start a new conversation session.

        Returns:
            The session ID for the new session.
        """
        self._conversation_history = []
        # Clear session-level lead tracking
        clear_session_leads()
        session_id = self.logger.create_session(
            model_id=self.model_id,
            agent_type="property_agent",
            channel="cli",
        )
        return session_id

    def process_message(self, user_input: str, session_id: str) -> str:
        """Process a user message and return the agent's response.

        This implements the main agent loop:
        1. Add user message to conversation
        2. Call LLM
        3. If there are tool calls, execute tools and continue
        4. Log all turns and tool calls
        5. Return final response when no more tool calls

        Args:
            user_input: The user's input message.
            session_id: The session ID for logging.

        Returns:
            The agent's final text response.
        """
        # Generate base system prompt using Jinja template
        base_prompt = render_property_agent_prompt(
            agent_name=self.agent_name,
            property_name=self.property_name,
            output_channel="cli",
        )

        # Build complete system prompt
        system_prompt = self._build_system_prompt(base_prompt)

        # Add user message to conversation history
        self._conversation_history.append({"role": "user", "content": user_input})

        # Log user turn
        user_turn = ConversationTurn(
            turn_id=self.logger.get_next_turn_id(session_id),
            role="user",
            content=user_input,
        )
        self.logger.add_turn(session_id, user_turn)

        # Track messages for this agent loop iteration
        messages: list = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in self._conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "tool_result":
                # Tool results for text-based mode
                messages.append(HumanMessage(content=f"Tool result: {msg['content']}"))

        # Agent loop
        while True:
            start_time = time.time()

            try:
                # Call LLM
                response = self.llm.invoke(messages)
            except Exception as e:
                error_msg = f"LLM error: {str(e)}"
                self.logger.add_error(
                    session_id,
                    error_type="LLMError",
                    error_message=error_msg,
                )
                raise RuntimeError(error_msg) from e

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract text content
            text_content = response.content if isinstance(response.content, str) else ""

            # Get tool calls based on mode
            if self.tool_calling_mode == "native":
                # Native mode: get tool calls from response object
                tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
                clean_response = text_content
            else:
                # Text-based mode: parse tool calls from text
                tool_calls = extract_tool_calls(text_content)
                # Remove tool call tokens from the displayed response
                clean_response = re.sub(
                    r"<start_function_call>.*?<end_function_call>",
                    "",
                    text_content,
                    flags=re.DOTALL
                ).strip()

            # Add assistant response to messages and conversation history
            if self.tool_calling_mode == "native":
                messages.append(response)
            else:
                messages.append(AIMessage(content=text_content))
            self._conversation_history.append({"role": "assistant", "content": text_content})

            # Build tool calls for logging
            logged_tool_calls: list[ToolCall] = []

            if tool_calls:
                # Process tool calls
                for i, tool_call in enumerate(tool_calls):
                    if self.tool_calling_mode == "native":
                        tool_name = tool_call.get("name", "")
                        tool_input = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id", f"call_{session_id}_{time.time()}_{i}")
                    else:
                        tool_name = tool_call.get("name", "")
                        tool_input = tool_call.get("arguments", {})
                        tool_call_id = f"call_{session_id}_{time.time()}_{i}"

                    # Execute the tool
                    result = self._execute_tool(tool_name, tool_input)

                    # Determine status
                    status = "error" if "error" in result else "success"
                    error_message = result.get("error") if status == "error" else None

                    # Create tool call record for logging
                    logged_tool_call = ToolCall(
                        tool_name=tool_name,
                        tool_use_id=tool_call_id,
                        input=tool_input,
                        output=result,
                        status=status,
                        error_message=error_message,
                    )
                    logged_tool_calls.append(logged_tool_call)

                    # Add tool result based on mode
                    if status == "success":
                        tool_result_content = json.dumps(result.get("result", {}))
                    else:
                        tool_result_content = f"Error: {error_message}"

                    if self.tool_calling_mode == "native":
                        # Native mode: use ToolMessage with name for FunctionGemma compatibility
                        messages.append(ToolMessage(
                            content=tool_result_content,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        ))
                    else:
                        # Text-based mode: format based on prompt_format
                        if self.prompt_format == "functiongemma":
                            # FunctionGemma expects specific response format
                            tool_response = result.get("result", {}) if status == "success" else {"error": error_message}
                            formatted_response = format_functiongemma_response(tool_name, tool_response)
                            self._conversation_history.append({
                                "role": "tool_result",
                                "content": formatted_response
                            })
                            messages.append(HumanMessage(content=formatted_response))
                        else:
                            # Default text format
                            self._conversation_history.append({
                                "role": "tool_result",
                                "content": f"Result of {tool_name}: {tool_result_content}"
                            })
                            messages.append(HumanMessage(
                                content=f"Tool result for {tool_name}: {tool_result_content}"
                            ))

            # Estimate token usage
            usage_data = {}
            if hasattr(response, 'response_metadata'):
                usage_data = response.response_metadata.get('usage', {})

            # Log assistant turn
            stop_reason = "tool_use" if tool_calls else "end_turn"
            assistant_turn = ConversationTurn(
                turn_id=self.logger.get_next_turn_id(session_id),
                role="assistant",
                content=clean_response,
                stop_reason=stop_reason,
                usage=Usage(
                    input_tokens=usage_data.get("prompt_tokens", 0),
                    output_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                ),
                latency_ms=latency_ms,
                tools_called=logged_tool_calls,
            )
            self.logger.add_turn(session_id, assistant_turn)

            # Check if we should exit the loop (no more tool calls)
            if not tool_calls:
                return clean_response

            # Continue the loop to process tool results

    def end_session(self, session_id: str, status: str = "completed") -> None:
        """End a conversation session.

        Args:
            session_id: The session ID to end.
            status: The final status (completed, abandoned, error).
        """
        self._conversation_history = []
        # Clear session-level lead tracking
        clear_session_leads()
        self.logger.end_session(session_id, status)

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the current conversation history.

        Returns:
            The conversation history list.
        """
        return self._conversation_history.copy()
