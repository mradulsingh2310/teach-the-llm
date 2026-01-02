"""
Property Agent Tools Package

This package contains all the tools for the property agent, including:
- Listing tools: Search and retrieve property listings
- Scheduling tools: Manage appointments and availability
- Handoff tools: Escalate conversations to human agents
- Knowledge base tools: Search property-related knowledge
- Lead tools: Create and manage leads
"""

from langchain_core.tools import tool
from .listing_tools import get_available_listings
from .scheduling_tools import get_availability, create_appointment
from .handoff_tools import escalate_conversation
from .knowledge_base_tools import search_property_knowledge
from .lead_tools import (
    create_lead,
    clear_session_leads,
    get_session_leads,
    has_lead_in_session,
    get_lead_by_email,
)

# Export all tools for easy access
ALL_TOOLS = [
    get_available_listings,
    get_availability,
    create_appointment,
    escalate_conversation,
    search_property_knowledge,
    create_lead,
]


__all__ = [
    "tool",
    "get_available_listings",
    "get_availability",
    "create_appointment",
    "escalate_conversation",
    "search_property_knowledge",
    "create_lead",
    "clear_session_leads",
    "get_session_leads",
    "has_lead_in_session",
    "get_lead_by_email",
    "ALL_TOOLS",
]
