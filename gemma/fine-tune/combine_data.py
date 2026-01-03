"""
Combine and enhance training data from data/ directory.

This script:
1. Loads all JSON files from data/ directory (simple format)
2. Uses AWS Bedrock to generate full conversations including:
   - Enhanced user content (natural/conversational)
   - Reasoning for <think> blocks
   - Realistic tool responses
   - Final assistant answers
3. Outputs in messages format for FunctionGemma training

Usage:
    python combine_data.py
    python combine_data.py --data-dir ./data --output ./training_data.json
    python combine_data.py --skip-bedrock  # Skip Bedrock, use templates
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_PROFILE = os.environ.get("AWS_PROFILE")
MODEL_ID = "moonshot.kimi-k2-thinking"

# Rate limiting: 6000 requests/min = 100/sec
# With ~2 min response time, concurrency is the bottleneck, not rate
REQUESTS_PER_SECOND = 50
MAX_CONCURRENT = 100  # Higher concurrency since each request takes ~2 min
MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0  # Exponential backoff base (1, 2, 4, 8, 16 seconds)

# Default system message
DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"

# Tool definitions (same as tools.py)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_listings",
            "description": "Search for available apartment listings based on bedroom count and optional maximum rent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bedrooms": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of bedroom counts to search for"
                    },
                    "max_rent": {
                        "type": "number",
                        "description": "Maximum monthly rent in dollars"
                    }
                },
                "required": ["bedrooms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_availability",
            "description": "Get available tour appointment time slots for a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_str": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format"
                    }
                },
                "required": ["date_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_appointment",
            "description": "Book a property tour appointment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_date": {"type": "string", "description": "Tour date in YYYY-MM-DD format"},
                    "appointment_start_time": {"type": "string", "description": "Tour time in HH:MM 24-hour format"},
                    "first_name": {"type": "string", "description": "Visitor's first name"},
                    "last_name": {"type": "string", "description": "Visitor's last name"},
                    "email": {"type": "string", "description": "Visitor's email address"},
                    "phone": {"type": "string", "description": "Visitor's phone number"}
                },
                "required": ["appointment_date", "appointment_start_time", "first_name", "email", "phone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_property_knowledge",
            "description": "Search the property knowledge base for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_text": {"type": "string", "description": "Query text to search for"},
                    "category": {"type": "string", "description": "Category filter"}
                },
                "required": ["search_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_lead",
            "description": "Save prospect contact information for follow-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string", "description": "Prospect's first name"},
                    "last_name": {"type": "string", "description": "Prospect's last name"},
                    "email": {"type": "string", "description": "Prospect's email address"},
                    "phone": {"type": "string", "description": "Prospect's phone number"}
                },
                "required": ["email", "phone", "first_name", "last_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_conversation",
            "description": "Transfer the conversation to a human leasing agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason_for_escalation": {
                        "type": "string",
                        "description": "Brief description of why human assistance is needed"
                    }
                },
                "required": ["reason_for_escalation"]
            }
        }
    }
]


def create_bedrock_client():
    """Create AWS Bedrock runtime client with configured profile."""
    from botocore.config import Config

    if not AWS_PROFILE:
        raise ValueError("AWS_PROFILE environment variable is not set")

    # Increase connection pool and configure retries
    config = Config(
        max_pool_connections=MAX_CONCURRENT + 10,  # Slightly more than concurrent
        retries={
            "max_attempts": 0,  # We handle retries manually with exponential backoff
            "mode": "standard"
        },
        connect_timeout=30,
        read_timeout=120
    )

    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client("bedrock-runtime", region_name="us-east-1", config=config)


def load_json_file(path: str) -> list[dict[str, Any]]:
    """Load a JSON file and return list of examples."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data:
            return data["data"]
        else:
            return [data]
    except Exception as e:
        logger.error("Error loading %s: %s", path, e)
        return []


def load_all_data(data_dir: str) -> list[dict[str, Any]]:
    """Load all JSON files from data directory."""
    data_path = Path(data_dir)
    all_examples = []

    json_files = sorted(data_path.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", data_dir)
        return []

    logger.info("Found %d data files in %s", len(json_files), data_dir)

    for json_file in json_files:
        examples = load_json_file(str(json_file))
        logger.info("  Loaded %d examples from %s", len(examples), json_file.name)
        all_examples.extend(examples)

    logger.info("Total examples loaded: %d", len(all_examples))
    return all_examples


def extract_from_messages(example: dict[str, Any]) -> dict[str, Any] | None:
    """Extract user_content, tool_name, tool_arguments from messages format."""
    messages = example.get("messages", [])

    user_content = None
    tool_name = None
    tool_arguments = None

    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")

        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            if tool_calls:
                tc = tool_calls[0]
                func = tc.get("function", {})
                tool_name = func.get("name")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    tool_arguments = args
                else:
                    tool_arguments = json.dumps(args)

    if user_content and tool_name:
        return {
            "user_content": user_content,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments or "{}"
        }
    return None


def get_tool_for_name(tool_name: str) -> dict | None:
    """Get tool definition by name."""
    for tool in TOOLS:
        if tool["function"]["name"] == tool_name:
            return tool
    return None


def build_generation_prompt(simple_example: dict[str, Any]) -> str:
    """Build prompt for Bedrock to generate full conversation."""
    user_content = simple_example["user_content"]
    tool_name = simple_example["tool_name"]
    tool_arguments = simple_example["tool_arguments"]

    return f"""You are generating training data for a property leasing AI assistant.

Given this interaction:
- Original user message: "{user_content}"
- Tool to call: {tool_name}
- Tool arguments: {tool_arguments}

Generate a complete conversation with these components:

1. ENHANCED_USER_CONTENT: Make the user message more natural and conversational. Add greetings, politeness, or indirect phrasing. Keep the same intent.

2. THINK_BEFORE_TOOL: Brief 1-2 sentence reasoning explaining WHY this tool should be called with these arguments. Focus on understanding the user's need.

3. TOOL_RESPONSE: Generate a realistic JSON response that the tool would return. Make it specific and useful. Examples:
   - For get_available_listings: Return 1-3 listings with unit_number, bedrooms, bathrooms, rent, square_feet, available_date, pet_friendly
   - For get_availability: Return available_slots array with times like ["10:00", "14:00", "15:30"]
   - For create_appointment: Return success with appointment_id
   - For search_property_knowledge: Return results array with title and content
   - For create_lead: Return success with lead_id
   - For escalate_conversation: Return success with estimated_wait

4. THINK_AFTER_TOOL: Brief reasoning about the tool response and how to present it to the user.

5. FINAL_RESPONSE: Natural, helpful response to the user based on the tool result.

IMPORTANT: Keep the tool_name and tool_arguments EXACTLY as provided. Return ONLY this JSON:

{{"enhanced_user_content": "...", "think_before_tool": "...", "tool_response": {{...}}, "think_after_tool": "...", "final_response": "..."}}"""


def extract_json_from_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from model response, handling thinking blocks."""
    original_text = text
    text = text.strip()

    # Required key to validate we found the right JSON (not an example inside reasoning)
    required_key = "enhanced_user_content"

    def is_valid_result(obj: dict) -> bool:
        """Check if extracted JSON has required fields."""
        return isinstance(obj, dict) and required_key in obj

    # Try direct JSON parse first
    try:
        result = json.loads(text)
        if is_valid_result(result):
            return result
    except json.JSONDecodeError:
        pass

    # Remove thinking model's <think> tags (Kimi K2 Thinking uses <think>...</think>)
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip()

    # Remove <reasoning> tags - handle both closed and unclosed tags
    if "<reasoning>" in text:
        if "</reasoning>" in text:
            text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
        text = text.strip()

    # Try parsing after removing tags
    try:
        result = json.loads(text)
        if is_valid_result(result):
            return result
    except json.JSONDecodeError:
        pass

    # Clean markdown code blocks from remaining text
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    try:
        result = json.loads(text)
        if is_valid_result(result):
            return result
    except json.JSONDecodeError:
        pass

    # Find JSON object by looking for { after </reasoning> or at end
    search_text = original_text

    # Look for JSON after </reasoning>
    reasoning_end = search_text.find("</reasoning>")
    if reasoning_end != -1:
        after_reasoning = search_text[reasoning_end + len("</reasoning>"):].strip()
        after_reasoning = re.sub(r"```json\s*", "", after_reasoning)
        after_reasoning = re.sub(r"```\s*$", "", after_reasoning)
        after_reasoning = re.sub(r"```", "", after_reasoning)
        after_reasoning = after_reasoning.strip()
        try:
            result = json.loads(after_reasoning)
            if is_valid_result(result):
                return result
        except json.JSONDecodeError:
            pass

    # Look for JSON after </think>
    think_end = search_text.find("</think>")
    if think_end != -1:
        after_think = search_text[think_end + len("</think>"):].strip()
        after_think = re.sub(r"```json\s*", "", after_think)
        after_think = re.sub(r"```\s*$", "", after_think)
        after_think = re.sub(r"```", "", after_think)
        after_think = after_think.strip()
        try:
            result = json.loads(after_think)
            if is_valid_result(result):
                return result
        except json.JSONDecodeError:
            pass

    # Look for JSON containing required key by searching from END of text
    # This handles unclosed <reasoning> tags with example JSON inside
    search_from = original_text

    # Find all { positions and try brace matching from each, starting from the last
    brace_positions = [i for i, c in enumerate(search_from) if c == '{']

    for start in reversed(brace_positions):
        depth = 0
        for i, char in enumerate(search_from[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(search_from[start:i+1])
                        if is_valid_result(result):
                            return result
                    except json.JSONDecodeError:
                        pass
                    break

    # Fallback: find any JSON with brace matching (first occurrence)
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    return None


def call_bedrock(client, prompt: str) -> dict[str, Any] | None:
    """Call Bedrock API with exponential backoff retry logic."""
    import random

    for attempt in range(MAX_RETRIES):
        try:
            # Kimi K2 uses OpenAI-compatible format
            request_body = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16384,
                "temperature": 0.7
            })

            response = client.invoke_model(
                modelId=MODEL_ID,
                body=request_body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            text = ""

            # OpenAI-compatible format (choices[].message.content)
            if "choices" in response_body:
                choices = response_body["choices"]
                if choices and isinstance(choices, list):
                    message = choices[0].get("message", {})
                    text = message.get("content", "")

            # Anthropic format (content[].text) - fallback
            elif "content" in response_body:
                content = response_body["content"]
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                break
                            elif "text" in block:
                                text = block.get("text", "")
                                break
                else:
                    text = str(content)

            # Legacy completion format
            elif "completion" in response_body:
                text = response_body["completion"]

            # Outputs format
            elif "outputs" in response_body:
                outputs = response_body["outputs"]
                text = outputs[0].get("text", "") if outputs else ""

            else:
                text = str(response_body)

            result = extract_json_from_response(text)
            if result:
                return result

            # Log warning with sample of raw response for debugging
            logger.warning(
                "Attempt %d: Could not extract JSON from response (len=%d): %s...",
                attempt + 1, len(text), text[:150].replace('\n', ' ')
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_msg = e.response.get("Error", {}).get("Message", "")

            # Check for throttling
            if error_code == "ThrottlingException" or "throttl" in error_msg.lower():
                # Exponential backoff with jitter for throttling
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Attempt %d: Throttled, waiting %.1fs before retry",
                    attempt + 1, delay
                )
                time.sleep(delay)
                continue

            logger.warning("Attempt %d: Bedrock error (%s): %s", attempt + 1, error_code, error_msg)
            if error_code in ["ValidationException", "AccessDeniedException"]:
                return None

        except Exception as e:
            logger.warning("Attempt %d: Error: %s", attempt + 1, e)

        if attempt < MAX_RETRIES - 1:
            # Exponential backoff with jitter
            import random
            delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

    return None


def convert_to_messages_format(
    simple_example: dict[str, Any],
    bedrock_result: dict[str, Any] | None
) -> dict[str, Any]:
    """Convert simple example + Bedrock result to messages format."""
    tool_name = simple_example["tool_name"]
    tool_arguments = simple_example["tool_arguments"]

    # Parse tool arguments
    if isinstance(tool_arguments, str):
        try:
            tool_args = json.loads(tool_arguments)
        except json.JSONDecodeError:
            tool_args = {}
    else:
        tool_args = tool_arguments

    # Use Bedrock results or fallback to defaults
    if bedrock_result:
        user_content = bedrock_result.get("enhanced_user_content", simple_example["user_content"])
        think_before = bedrock_result.get("think_before_tool", f"User needs help with {tool_name}.")
        tool_response = bedrock_result.get("tool_response", {"success": True})
        think_after = bedrock_result.get("think_after_tool", "Got the result, presenting to user.")
        final_response = bedrock_result.get("final_response", "Here's what I found.")
    else:
        user_content = simple_example["user_content"]
        think_before = f"User needs help. Calling {tool_name}."
        tool_response = generate_default_tool_response(tool_name, tool_args)
        think_after = "Got the result."
        final_response = "Here's what I found."

    # Ensure tool_response is a dict
    if isinstance(tool_response, str):
        try:
            tool_response = json.loads(tool_response)
        except json.JSONDecodeError:
            tool_response = {"success": True, "message": tool_response}

    # Build messages
    messages = [
        {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": f"<think>{think_before}</think>",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_args
                }
            }]
        },
        {
            "role": "tool",
            "name": tool_name,
            "tool_call_id": "call_1",
            "content": json.dumps(tool_response)
        },
        {
            "role": "assistant",
            "content": f"<think>{think_after}</think>\n{final_response}"
        }
    ]

    # Get tool definition for this example
    tool_def = get_tool_for_name(tool_name)
    tools = [tool_def] if tool_def else TOOLS

    return {
        "messages": messages,
        "tools": tools
    }


def generate_default_tool_response(tool_name: str, tool_args: dict) -> dict:
    """Generate default tool response without Bedrock."""
    if tool_name == "get_available_listings":
        bedrooms = tool_args.get("bedrooms", [1])
        br = bedrooms[0] if bedrooms else 1
        return {
            "success": True,
            "listings": [{
                "unit_number": "201",
                "bedrooms": br,
                "bathrooms": 1,
                "rent": 1500 + (br * 300),
                "square_feet": 600 + (br * 200),
                "available_date": "2025-02-01",
                "pet_friendly": True
            }],
            "total_count": 1
        }
    elif tool_name == "get_availability":
        return {
            "success": True,
            "date": tool_args.get("date_str", "2025-02-15"),
            "available_slots": ["10:00", "11:00", "14:00", "15:30"]
        }
    elif tool_name == "create_appointment":
        return {
            "success": True,
            "appointment_id": "APT-12345",
            "message": f"Appointment confirmed for {tool_args.get('appointment_date', 'the requested date')}"
        }
    elif tool_name == "search_property_knowledge":
        return {
            "success": True,
            "results": [{
                "title": "Property Information",
                "content": f"Information about {tool_args.get('search_text', 'your query')}."
            }]
        }
    elif tool_name == "create_lead":
        return {
            "success": True,
            "lead_id": "LEAD-789",
            "message": "Lead created successfully"
        }
    elif tool_name == "escalate_conversation":
        return {
            "success": True,
            "message": "Conversation escalated to human agent",
            "estimated_wait": "2 minutes"
        }
    return {"success": True}


class RateLimiter:
    """Token bucket rate limiter for controlling request rate."""

    def __init__(self, rate: float):
        """Initialize with requests per second."""
        self.rate = rate
        self.tokens = rate
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made."""
        async with self._lock:
            now = time.monotonic()
            # Add tokens based on time passed
            time_passed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + time_passed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                # Wait for token to be available
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


def process_single_example_sync(
    client,
    simple_example: dict[str, Any],
    index: int
) -> dict[str, Any]:
    """Process a single example using Bedrock (synchronous)."""
    prompt = build_generation_prompt(simple_example)
    result = call_bedrock(client, prompt)

    if not result:
        logger.debug("Example %d: Bedrock failed, using defaults", index)

    return convert_to_messages_format(simple_example, result)


async def process_single_example_async(
    client,
    simple_example: dict[str, Any],
    index: int,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    executor: ThreadPoolExecutor
) -> dict[str, Any]:
    """Process a single example with rate limiting."""
    async with semaphore:
        await rate_limiter.acquire()

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                executor,
                process_single_example_sync,
                client,
                simple_example,
                index
            )
            return result
        except Exception as e:
            logger.error("Example %d failed: %s", index, str(e))
            return convert_to_messages_format(simple_example, None)


async def process_all_examples(
    examples: list[dict[str, Any]],
    use_bedrock: bool = True
) -> list[dict[str, Any]]:
    """Process all examples with rate limiting."""
    if use_bedrock:
        client = create_bedrock_client()
        logger.info("Using Bedrock for generation (model: %s)", MODEL_ID)
        logger.info(
            "Rate limit: %d req/sec, max concurrent: %d",
            REQUESTS_PER_SECOND, MAX_CONCURRENT
        )
    else:
        client = None
        logger.info("Skipping Bedrock, using default templates")

    total = len(examples)

    if use_bedrock and client:
        # Create rate limiter and semaphore
        rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        # Use thread pool for blocking Bedrock calls
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            # Create all tasks
            tasks = [
                process_single_example_async(
                    client, ex, i, semaphore, rate_limiter, executor
                )
                for i, ex in enumerate(examples)
            ]

            # Process with progress logging
            processed_examples = []
            start_time = time.time()

            # Process in batches for progress reporting
            batch_size = 100
            for batch_start in range(0, len(tasks), batch_size):
                batch_tasks = tasks[batch_start:batch_start + batch_size]
                batch_results = await asyncio.gather(*batch_tasks)
                processed_examples.extend(batch_results)

                elapsed = time.time() - start_time
                rate = len(processed_examples) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f%%) - %.1f req/sec",
                    len(processed_examples), total,
                    len(processed_examples) / total * 100,
                    rate
                )
    else:
        # Generate without Bedrock
        processed_examples = []
        for ex in examples:
            processed_examples.append(convert_to_messages_format(ex, None))

    return processed_examples


def save_training_data(data: list[dict[str, Any]], output_path: str) -> None:
    """Save training data to JSON file."""
    logger.info("Saving %d examples to %s", len(data), output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Print statistics
    tool_counts: dict[str, int] = {}

    for ex in data:
        for msg in ex.get("messages", []):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "unknown")
                    tool_counts[name] = tool_counts.get(name, 0) + 1

    logger.info("=" * 50)
    logger.info("Dataset Statistics")
    logger.info("=" * 50)
    logger.info("Total examples: %d", len(data))
    logger.info("Tool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", tool, count)


async def main_async(
    data_dir: str,
    output_path: str,
    use_bedrock: bool,
    limit: int | None = None,
    debug: bool = False
) -> None:
    """Main async entry point."""
    start_time = time.time()

    # Step 1: Load data
    logger.info("=" * 50)
    logger.info("Loading data from %s", data_dir)
    logger.info("=" * 50)

    all_data = load_all_data(data_dir)
    if not all_data:
        logger.error("No data found in %s", data_dir)
        return

    # Step 2: Extract to simple format
    logger.info("=" * 50)
    logger.info("Extracting to simple format")
    logger.info("=" * 50)

    simple_examples = []
    skipped = 0

    for ex in all_data:
        # Handle both formats
        if "user_content" in ex:
            simple_examples.append({
                "user_content": ex["user_content"],
                "tool_name": ex["tool_name"],
                "tool_arguments": ex["tool_arguments"]
            })
        elif "messages" in ex:
            extracted = extract_from_messages(ex)
            if extracted:
                simple_examples.append(extracted)
            else:
                skipped += 1
        else:
            skipped += 1

    logger.info("Extracted %d examples, skipped %d", len(simple_examples), skipped)

    # Apply limit if specified
    if limit and limit > 0:
        original_count = len(simple_examples)
        simple_examples = simple_examples[:limit]
        logger.info("Limited to %d examples (from %d)", len(simple_examples), original_count)

    if not simple_examples:
        logger.error("No valid examples extracted")
        return

    # Step 3: Process with Bedrock (or defaults)
    logger.info("=" * 50)
    logger.info("Generating full conversations")
    logger.info("=" * 50)

    processed_examples = await process_all_examples(simple_examples, use_bedrock)

    # Step 4: Save
    logger.info("=" * 50)
    logger.info("Saving training data")
    logger.info("=" * 50)

    save_training_data(processed_examples, output_path)

    # Debug: Check for failures (examples that fell back to defaults)
    if debug:
        failures = []
        for i, ex in enumerate(processed_examples):
            msgs = ex.get("messages", [])
            for msg in msgs:
                # Check if it used default template (indicates Bedrock failure)
                content = msg.get("content", "")
                if msg.get("role") == "assistant" and "User needs help. Calling" in content:
                    failures.append({
                        "index": i,
                        "tool": simple_examples[i]["tool_name"],
                        "user_content": simple_examples[i]["user_content"][:100]
                    })
                    break

        if failures:
            failures_path = output_path.replace(".json", "_failures.json")
            with open(failures_path, "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=2)
            logger.warning("Found %d failures (used defaults), saved to %s", len(failures), failures_path)
        else:
            logger.info("No failures detected - all examples processed successfully!")

    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info("Complete in %.1f seconds", elapsed)
    logger.info("Output: %s", output_path)
    logger.info("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate FunctionGemma training data in messages format"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="./data",
        help="Directory containing JSON data files"
    )
    parser.add_argument(
        "--output", "-o",
        default="./training_data.json",
        help="Output file path"
    )
    parser.add_argument(
        "--skip-bedrock",
        action="store_true",
        help="Skip Bedrock API calls, use default templates"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save failures to separate file"
    )

    args = parser.parse_args()

    asyncio.run(main_async(
        data_dir=args.data_dir,
        output_path=args.output,
        use_bedrock=not args.skip_bedrock,
        limit=args.limit,
        debug=args.debug
    ))


if __name__ == "__main__":
    main()
