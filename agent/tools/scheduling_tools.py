"""
Scheduling Tools for Property Agent

This module contains tools for managing appointments and availability.
"""

from typing import Annotated, Any, Dict, Optional
from langchain_core.tools import tool
import uuid
import os
import warnings
from datetime import datetime


# Dummy availability data
DUMMY_AVAILABILITY = {
    "slots": [
        {"time": "09:00", "available": True},
        {"time": "10:00", "available": True},
        {"time": "11:00", "available": False},
        {"time": "12:00", "available": False},
        {"time": "13:00", "available": True},
        {"time": "14:00", "available": True},
        {"time": "15:00", "available": True},
        {"time": "16:00", "available": False},
        {"time": "17:00", "available": True},
    ]
}

# Path for tour bookings log file (relative to agent/data/)
TOUR_BOOKINGS_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "tour_bookings.txt"
)


def _log_tour_booking(
    date: str,
    time: str,
    name: str,
    email: str,
    phone: str,
    confirmation_number: str
) -> None:
    """
    Log a tour booking to the tour_bookings.txt file.

    Creates the data/ directory and file if they don't exist.
    If logging fails, a warning is issued but no exception is raised.
    """
    try:
        # Get the directory path
        dir_path = os.path.dirname(TOUR_BOOKINGS_LOG_PATH)

        # Create directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().isoformat()

        # Format the log entry
        log_entry = f"[{timestamp}] Tour booked for {date} at {time} by {name} (email: {email}, phone: {phone}) - Confirmation: {confirmation_number}\n"

        # Append to file (creates file if it doesn't exist)
        with open(TOUR_BOOKINGS_LOG_PATH, "a") as f:
            f.write(log_entry)

    except Exception as e:
        # Log warning but don't raise - appointment should still be created
        warnings.warn(f"Failed to log tour booking to file: {e}")


@tool
def get_availability(
    date_str: Annotated[str, "REQUIRED: Date in YYYY-MM-DD format"],
) -> Dict[str, Any]:
    """Get appointment slots for a property on a given date.

    This is a DUMMY implementation that returns static availability slots.
    """
    # Validate date format
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = parsed_date.strftime("%A, %B %d, %Y")
    except ValueError:
        return {
            "success": False,
            "error": "Invalid date format. Please use YYYY-MM-DD format.",
            "available_slots": []
        }

    available_slots = [
        slot for slot in DUMMY_AVAILABILITY["slots"] if slot["available"]
    ]

    return {
        "success": True,
        "date": date_str,
        "formatted_date": formatted_date,
        "available_slots": available_slots,
        "total_available": len(available_slots),
        "message": f"Found {len(available_slots)} available time slots for {formatted_date}"
    }


@tool
def create_appointment(
    appointment_date: Annotated[str, "REQUIRED: Date in YYYY-MM-DD format"],
    appointment_start_time: Annotated[str, "REQUIRED: Start time in 24-hour HH:MM format"],
    first_name: Annotated[str, "REQUIRED: First name of the person"],
    email: Annotated[str, "REQUIRED: Email address"],
    phone: Annotated[str, "REQUIRED: Phone number"],
    last_name: Annotated[Optional[str], "OPTIONAL: Last name"] = None,
) -> Dict[str, Any]:
    """Create an appointment for a property tour.

    This is a DUMMY implementation that returns a confirmation.
    """
    # Validate date format
    try:
        parsed_date = datetime.strptime(appointment_date, "%Y-%m-%d")
        formatted_date = parsed_date.strftime("%A, %B %d, %Y")
    except ValueError:
        return {
            "success": False,
            "error": "Invalid date format. Please use YYYY-MM-DD format."
        }

    # Validate time format
    try:
        parsed_time = datetime.strptime(appointment_start_time, "%H:%M")
        formatted_time = parsed_time.strftime("%I:%M %p")
    except ValueError:
        return {
            "success": False,
            "error": "Invalid time format. Please use HH:MM format (24-hour)."
        }

    # Generate dummy confirmation
    confirmation_id = str(uuid.uuid4())[:8].upper()
    full_name = f"{first_name} {last_name}" if last_name else first_name

    # Log the tour booking to file
    _log_tour_booking(
        date=appointment_date,
        time=appointment_start_time,
        name=full_name,
        email=email,
        phone=phone,
        confirmation_number=confirmation_id
    )

    return {
        "success": True,
        "confirmation_id": confirmation_id,
        "appointment_details": {
            "date": appointment_date,
            "formatted_date": formatted_date,
            "time": appointment_start_time,
            "formatted_time": formatted_time,
            "duration": "30 minutes"
        },
        "contact_info": {
            "name": full_name,
            "email": email,
            "phone": phone
        },
        "message": f"Appointment confirmed for {full_name} on {formatted_date} at {formatted_time}. Confirmation ID: {confirmation_id}",
        "next_steps": [
            "A confirmation email has been sent to your email address",
            "Please arrive 5 minutes before your scheduled time",
            "Bring a valid photo ID for the tour",
            "Feel free to prepare any questions about the property"
        ]
    }
