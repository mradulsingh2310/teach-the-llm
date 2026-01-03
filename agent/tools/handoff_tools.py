"""
Handoff Tools for Property Agent

This module contains tools for escalating conversations to human agents.
"""

from typing import Annotated, Dict
from langchain_core.tools import tool
import uuid
from datetime import datetime


@tool
def escalate_conversation(
    reason_for_escalation: Annotated[str, "REQUIRED: Reason for escalating the conversation"],
) -> Dict[str, str]:
    """Escalate to leasing team for complex queries.

    This is a DUMMY implementation that returns an escalation confirmation.
    """
    # Generate dummy escalation ticket
    ticket_id = f"ESC-{str(uuid.uuid4())[:6].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "success": True,
        "ticket_id": ticket_id,
        "status": "escalated",
        "reason": reason_for_escalation,
        "timestamp": timestamp,
        "message": f"Your conversation has been escalated to our leasing team. Ticket ID: {ticket_id}",
        "expected_response_time": "A team member will reach out within 1-2 business hours",
        "next_steps": "Our leasing specialist will contact you via phone or email to assist with your inquiry"
    }
