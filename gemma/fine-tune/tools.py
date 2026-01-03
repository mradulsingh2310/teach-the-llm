"""
Tool Definitions for FunctionGemma Fine-tuning

This module provides tool definitions in the format required by
Google's FunctionGemma for supervised fine-tuning.
"""

from typing import Any, Dict, List


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions in JSON schema format for FunctionGemma.

    These definitions are used:
    1. During training - to teach the model the available tools
    2. During inference - to enable tool calling

    Returns:
        List of tool definitions in OpenAI-compatible format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_available_listings",
                "description": "Search for available apartment listings based on bedroom count and optional maximum rent. Returns top 5 matching listings sorted by rent (lowest first).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bedrooms": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "REQUIRED: List of bedroom counts to search for. Use [0] for studios, [1] for 1-bedroom, [2] for 2-bedroom, etc. Can include multiple options like [1, 2] for 1BR or 2BR."
                        },
                        "max_rent": {
                            "type": "number",
                            "description": "OPTIONAL: Maximum monthly rent in dollars. Only return listings at or below this price."
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
                "description": "Get available tour appointment time slots for a specific date. Returns list of available times.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_str": {
                            "type": "string",
                            "description": "REQUIRED: Date to check availability for, in YYYY-MM-DD format (e.g., '2025-02-15', '2025-03-01')"
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
                "description": "Book a property tour appointment. Requires date, time, and visitor contact information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_date": {
                            "type": "string",
                            "description": "REQUIRED: Tour date in YYYY-MM-DD format (e.g., '2025-02-15')"
                        },
                        "appointment_start_time": {
                            "type": "string",
                            "description": "REQUIRED: Tour time in HH:MM 24-hour format (e.g., '10:00' for 10am, '14:30' for 2:30pm)"
                        },
                        "first_name": {
                            "type": "string",
                            "description": "REQUIRED: Visitor's first name"
                        },
                        "email": {
                            "type": "string",
                            "description": "REQUIRED: Visitor's email address for confirmation"
                        },
                        "phone": {
                            "type": "string",
                            "description": "REQUIRED: Visitor's phone number as a string (e.g., '555-123-4567')"
                        },
                        "last_name": {
                            "type": "string",
                            "description": "OPTIONAL: Visitor's last name"
                        }
                    },
                    "required": ["appointment_date", "appointment_start_time", "first_name", "email", "phone"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_property_knowledge",
                "description": "Search the property knowledge base for information about amenities, policies, leasing, rent payments, utilities, maintenance, neighborhood, and move-in/move-out procedures.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_text": {
                            "type": "string",
                            "description": "REQUIRED: Query text to search for (e.g., 'pet policy', 'parking options', 'gym hours')"
                        },
                        "category": {
                            "type": "string",
                            "enum": ["amenities", "neighborhood", "leasing", "rent_payment", "policies", "utilities", "maintenance", "move_in_out"],
                            "description": "OPTIONAL: Category filter to narrow search results"
                        }
                    },
                    "required": ["search_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_lead",
                "description": "Save prospect contact information for follow-up by the leasing team. Creates a lead record with the visitor's details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "REQUIRED: Prospect's email address"
                        },
                        "phone": {
                            "type": "string",
                            "description": "REQUIRED: Prospect's phone number as a string"
                        },
                        "first_name": {
                            "type": "string",
                            "description": "REQUIRED: Prospect's first name"
                        },
                        "last_name": {
                            "type": "string",
                            "description": "REQUIRED: Prospect's last name"
                        }
                    },
                    "required": ["email", "phone", "first_name", "last_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "escalate_conversation",
                "description": "Transfer the conversation to a human leasing agent when the user requests human assistance, expresses frustration, or has complex inquiries that require human judgment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason_for_escalation": {
                            "type": "string",
                            "description": "REQUIRED: Brief description of why human assistance is needed (e.g., 'User frustrated and requesting human agent', 'Complex lease negotiation question')"
                        }
                    },
                    "required": ["reason_for_escalation"]
                }
            }
        }
    ]


# System prompt that activates function calling mode
SYSTEM_PROMPT = "You are a model that can do function calling with the following functions"


# For backwards compatibility and easier imports
TOOLS = get_tool_definitions()
