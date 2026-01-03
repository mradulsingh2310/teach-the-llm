"""
Lead Tools for Property Agent

This module contains tools for creating and managing leads.
Supports session-level lead tracking to prevent duplicate lead creation.
Logs all lead creation to leads.txt for tracking and verification.
"""

from typing import Annotated, Dict, Optional
from langchain_core.tools import tool
import uuid
import os
import warnings
from datetime import datetime


# Session-level lead storage (injected by PropertyAgent)
_session_leads: Dict[str, Dict] = {}

# Path for leads log file (relative to agent/data/)
LEADS_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "leads.txt"
)


def _log_lead_creation(
    lead_id: str,
    first_name: str,
    last_name: str,
    email: str,
    phone: str,
    status: str,
    timestamp: str
) -> None:
    """
    Log a lead creation to the leads.txt file.

    Creates the data/ directory and file if they don't exist.
    If logging fails, a warning is issued but no exception is raised.
    """
    try:
        # Get the directory path
        dir_path = os.path.dirname(LEADS_LOG_PATH)

        # Create directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Generate log timestamp
        log_timestamp = datetime.now().isoformat()

        # Format the log entry
        full_name = f"{first_name} {last_name}"
        log_entry = f"[{log_timestamp}] Lead {status}: {lead_id} - {full_name} (email: {email}, phone: {phone}) - Created: {timestamp}\n"

        # Append to file (creates file if it doesn't exist)
        with open(LEADS_LOG_PATH, "a") as f:
            f.write(log_entry)

    except Exception as e:
        # Log warning but don't raise - lead should still be created
        warnings.warn(f"Failed to log lead creation to file: {e}")


def set_session_leads(leads: Dict[str, Dict]) -> None:
    """Set the session leads reference (called by PropertyAgent)."""
    global _session_leads
    _session_leads = leads


def get_session_leads() -> Dict[str, Dict]:
    """Get the current session leads."""
    return _session_leads


def clear_session_leads() -> None:
    """Clear all session leads (called when session ends)."""
    global _session_leads
    _session_leads = {}


def _normalize_email(email: str) -> str:
    """Normalize email for comparison."""
    return email.lower().strip()


def _find_existing_lead(email: str, phone: str) -> Optional[Dict]:
    """
    Check if a lead already exists in the session by email or phone.

    Args:
        email: Email address to check
        phone: Phone number to check

    Returns:
        Existing lead dict if found, None otherwise
    """
    normalized_email = _normalize_email(email)

    for lead_id, lead_data in _session_leads.items():
        # Check by email (primary identifier)
        if _normalize_email(lead_data.get("email", "")) == normalized_email:
            return {"lead_id": lead_id, **lead_data}
        # Also check by phone as backup
        if lead_data.get("phone") == phone:
            return {"lead_id": lead_id, **lead_data}

    return None


@tool
def create_lead(
    email: Annotated[str, "REQUIRED: Email address"],
    phone: Annotated[str, "REQUIRED: Phone number"],
    first_name: Annotated[str, "REQUIRED: First name"],
    last_name: Annotated[str, "REQUIRED: Last name"],
) -> Dict[str, str]:
    """Create a lead from the conversation with prospect information.

    If a lead with the same email already exists in this session, returns the existing lead instead of creating a duplicate.
    """
    # Check for existing lead in session
    existing_lead = _find_existing_lead(email, phone)

    if existing_lead:
        lead_id = existing_lead["lead_id"]
        full_name = f"{existing_lead.get('first_name', '')} {existing_lead.get('last_name', '')}"

        # Log the duplicate attempt
        _log_lead_creation(
            lead_id=lead_id,
            first_name=existing_lead.get('first_name', ''),
            last_name=existing_lead.get('last_name', ''),
            email=existing_lead.get('email', ''),
            phone=existing_lead.get('phone', ''),
            status="already_exists",
            timestamp=existing_lead.get("created_at", "")
        )

        return {
            "success": True,
            "lead_id": lead_id,
            "status": "already_exists",
            "created_at": existing_lead.get("created_at", ""),
            "prospect_info": f"{full_name} | Email: {existing_lead.get('email', '')} | Phone: {existing_lead.get('phone', '')}",
            "message": f"Lead for {full_name} already exists in this session (Lead ID: {lead_id}). No duplicate created.",
            "next_steps": "Contact information is already captured. Our leasing team will follow up within 24 hours."
        }

    # Generate new lead ID
    lead_id = f"LEAD-{str(uuid.uuid4())[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    full_name = f"{first_name} {last_name}"

    # Store in session leads
    _session_leads[lead_id] = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "created_at": timestamp,
    }

    # Log the new lead creation to file
    _log_lead_creation(
        lead_id=lead_id,
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone=phone,
        status="created",
        timestamp=timestamp
    )

    return {
        "success": True,
        "lead_id": lead_id,
        "status": "created",
        "created_at": timestamp,
        "prospect_info": f"{full_name} | Email: {email} | Phone: {phone}",
        "message": f"Lead successfully created for {full_name}. Lead ID: {lead_id}",
        "next_steps": "Our leasing team will follow up within 24 hours"
    }


def get_lead_by_email(email: str) -> Optional[Dict]:
    """
    Get a lead from the session by email address.

    Args:
        email: Email address to look up

    Returns:
        Lead dict if found, None otherwise
    """
    normalized_email = _normalize_email(email)

    for lead_id, lead_data in _session_leads.items():
        if _normalize_email(lead_data.get("email", "")) == normalized_email:
            return {"lead_id": lead_id, **lead_data}

    return None


def has_lead_in_session() -> bool:
    """Check if any lead has been created in this session."""
    return len(_session_leads) > 0
