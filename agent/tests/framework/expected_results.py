"""
Expected Results Generator for Property Agent Tests.

This module generates expected results (keywords, tool parameters, file writes)
based on the actual data in agent/data/. This ensures test expectations are
grounded in real data.

Data Sources:
- listings.json: Property listings with bedrooms, rent, amenities
- knowledge_base.json: Property FAQ with categories and keywords
- tour_bookings.txt: Tour booking log file
- leads.txt: Lead creation log file
"""

import json
import os
import re
from typing import Any, Optional

from .test_metrics import (
    ExpectedKeywords,
    ExpectedToolParameters,
    TurnExpectation,
)


# =============================================================================
# Data File Paths
# =============================================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data"
)

LISTINGS_PATH = os.path.join(DATA_DIR, "listings.json")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.json")


# =============================================================================
# Data Loaders
# =============================================================================

def load_listings() -> list[dict]:
    """Load property listings from JSON file."""
    if not os.path.exists(LISTINGS_PATH):
        return []
    with open(LISTINGS_PATH, "r") as f:
        data = json.load(f)
    return data.get("listings", [])


def load_knowledge_base() -> list[dict]:
    """Load knowledge base entries from JSON file."""
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        return []
    with open(KNOWLEDGE_BASE_PATH, "r") as f:
        data = json.load(f)
    return data.get("entries", [])


# =============================================================================
# Listings Analysis
# =============================================================================

def get_listings_by_criteria(
    bedrooms: Optional[list[int]] = None,
    max_rent: Optional[int] = None,
    min_rent: Optional[int] = None,
    pet_friendly: Optional[bool] = None,
    parking_included: Optional[bool] = None,
) -> list[dict]:
    """
    Filter listings by given criteria.

    Args:
        bedrooms: List of acceptable bedroom counts
        max_rent: Maximum rent
        min_rent: Minimum rent
        pet_friendly: Must be pet friendly
        parking_included: Must include parking

    Returns:
        List of matching listings
    """
    listings = load_listings()
    results = []

    for listing in listings:
        # Filter by bedrooms
        if bedrooms is not None and listing.get("bedrooms") not in bedrooms:
            continue

        # Filter by rent
        rent = listing.get("rent", 0)
        if max_rent is not None and rent > max_rent:
            continue
        if min_rent is not None and rent < min_rent:
            continue

        # Filter by pet friendly
        if pet_friendly is not None and listing.get("pet_friendly") != pet_friendly:
            continue

        # Filter by parking
        if parking_included is not None and listing.get("parking_included") != parking_included:
            continue

        results.append(listing)

    return results


def get_rent_range_for_bedrooms(bedrooms: int) -> tuple[int, int]:
    """Get min and max rent for a given bedroom count."""
    listings = [l for l in load_listings() if l.get("bedrooms") == bedrooms]
    if not listings:
        return (0, 0)
    rents = [l.get("rent", 0) for l in listings]
    return (min(rents), max(rents))


def get_available_bedroom_counts() -> list[int]:
    """Get all available bedroom counts."""
    listings = load_listings()
    return sorted(set(l.get("bedrooms", 0) for l in listings))


# =============================================================================
# Knowledge Base Analysis
# =============================================================================

def get_keywords_for_category(category: str) -> list[str]:
    """Get all keywords for a knowledge base category."""
    entries = load_knowledge_base()
    keywords = []

    for entry in entries:
        if entry.get("category") == category:
            keywords.extend(entry.get("keywords", []))

    return list(set(keywords))


def get_answer_keywords(search_text: str) -> list[str]:
    """
    Get keywords expected in response for a knowledge base search.

    Args:
        search_text: The search query

    Returns:
        List of keywords that should appear in the answer
    """
    entries = load_knowledge_base()
    search_lower = search_text.lower()

    # Find matching entries
    matched_keywords = []
    for entry in entries:
        entry_keywords = [k.lower() for k in entry.get("keywords", [])]

        # Check if search text matches any keywords
        if any(search_lower in k or k in search_lower for k in entry_keywords):
            # Add some keywords from the answer
            answer = entry.get("answer", "")
            answer_words = answer.split()

            # Extract key words (numbers, proper nouns, specific terms)
            for word in answer_words:
                # Include numbers (prices, times, etc.)
                if re.match(r'^\$?\d+', word):
                    matched_keywords.append(word.strip('.,'))
                # Include specific terms
                elif len(word) > 4 and word[0].isupper():
                    matched_keywords.append(word.strip('.,'))

            # Include category-specific keywords
            matched_keywords.extend(entry.get("keywords", [])[:3])

    return list(set(matched_keywords))[:10]


# =============================================================================
# Expected Keywords Generation
# =============================================================================

def generate_listing_search_keywords(
    bedrooms: Optional[list[int]] = None,
    max_rent: Optional[int] = None,
    should_find_results: bool = True,
) -> ExpectedKeywords:
    """
    Generate expected keywords for a listing search response.

    Args:
        bedrooms: Bedroom counts being searched
        max_rent: Maximum rent in search
        should_find_results: Whether results are expected

    Returns:
        ExpectedKeywords for the response
    """
    required = []
    optional = []
    forbidden = []

    if should_find_results:
        # Check if listings exist
        matches = get_listings_by_criteria(bedrooms=bedrooms, max_rent=max_rent)

        if matches:
            # Should mention unit numbers or availability
            required.extend(["unit", "available", "bedroom"])

            # Add specific unit numbers as optional
            for listing in matches[:3]:
                optional.append(listing.get("unit_number", ""))

            # Add rent information
            if max_rent:
                optional.append(str(max_rent))

            # Add bedroom count
            if bedrooms:
                for b in bedrooms:
                    if b == 0:
                        optional.append("studio")
                    else:
                        optional.append(f"{b}")
        else:
            # No matches - should apologize or suggest alternatives
            required.extend(["sorry", "no", "available"])
            optional.extend(["budget", "alternative", "different"])
    else:
        required.extend(["no", "unavailable", "sorry"])
        forbidden.extend(["booked", "confirmed", "scheduled"])

    return ExpectedKeywords(
        required=list(set(required)),
        optional=list(set(optional)),
        forbidden=list(set(forbidden)),
    )


def generate_availability_keywords(date_str: str) -> ExpectedKeywords:
    """
    Generate expected keywords for availability check response.

    Args:
        date_str: Date being checked (YYYY-MM-DD)

    Returns:
        ExpectedKeywords for the response
    """
    # Parse date for display format
    try:
        from datetime import datetime
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        month_name = date_obj.strftime("%B")
        day = str(date_obj.day)
    except ValueError:
        month_name = ""
        day = ""

    return ExpectedKeywords(
        required=["available", "time", "slot"],
        optional=[month_name, day, "AM", "PM", "morning", "afternoon", "9:00", "10:00", "14:00", "15:00"],
        forbidden=[],
    )


def generate_appointment_keywords(
    first_name: str,
    time: str,
) -> ExpectedKeywords:
    """
    Generate expected keywords for appointment booking response.

    Args:
        first_name: Person's first name
        time: Appointment time

    Returns:
        ExpectedKeywords for the response
    """
    return ExpectedKeywords(
        required=["confirmed", "appointment", "booked"],
        optional=[first_name, time, "tour", "scheduled", "confirmation"],
        forbidden=["error", "failed", "unavailable"],
    )


def generate_lead_keywords(first_name: str) -> ExpectedKeywords:
    """
    Generate expected keywords for lead creation response.

    Args:
        first_name: Lead's first name

    Returns:
        ExpectedKeywords for the response
    """
    return ExpectedKeywords(
        required=["contact", "information", "saved"],
        optional=[first_name, "team", "follow", "reach"],
        forbidden=["error", "failed"],
    )


def generate_knowledge_search_keywords(search_text: str) -> ExpectedKeywords:
    """
    Generate expected keywords for knowledge base search response.

    Args:
        search_text: The search query

    Returns:
        ExpectedKeywords for the response
    """
    # Get keywords from the knowledge base based on search
    answer_keywords = get_answer_keywords(search_text)

    # Category-specific required keywords
    category_keywords = {
        "pet": ["pet", "dog", "cat", "deposit", "policy"],
        "parking": ["parking", "garage", "spot", "fee"],
        "smoking": ["smoke", "prohibited", "free"],
        "utilities": ["utilities", "water", "electric", "included"],
        "pool": ["pool", "swim", "heated"],
        "gym": ["gym", "fitness", "workout"],
        "laundry": ["laundry", "washer", "dryer"],
    }

    required = []
    optional = answer_keywords

    # Find matching category keywords
    search_lower = search_text.lower()
    for key, keywords in category_keywords.items():
        if key in search_lower:
            required.extend(keywords[:2])  # Add first 2 as required
            optional.extend(keywords[2:])  # Rest as optional

    return ExpectedKeywords(
        required=list(set(required)) if required else ["information"],
        optional=list(set(optional)),
        forbidden=[],
    )


# =============================================================================
# Expected Tool Parameters Generation
# =============================================================================

def generate_listing_tool_params(
    bedrooms: Optional[list[int]] = None,
    max_rent: Optional[int] = None,
    pet_friendly: Optional[bool] = None,
) -> ExpectedToolParameters:
    """
    Generate expected parameters for get_available_listings tool.

    Args:
        bedrooms: Expected bedroom filter
        max_rent: Expected max rent filter
        pet_friendly: Expected pet friendly filter

    Returns:
        ExpectedToolParameters for the tool call
    """
    params = {}
    required = ["bedrooms"]

    if bedrooms is not None:
        params["bedrooms"] = bedrooms

    if max_rent is not None:
        params["max_rent"] = max_rent

    if pet_friendly is not None:
        params["pet_friendly"] = pet_friendly

    return ExpectedToolParameters(
        tool_name="get_available_listings",
        parameters=params,
        required_params=required,
    )


def generate_availability_tool_params(date_str: str) -> ExpectedToolParameters:
    """
    Generate expected parameters for get_availability tool.

    Args:
        date_str: Expected date parameter

    Returns:
        ExpectedToolParameters for the tool call
    """
    return ExpectedToolParameters(
        tool_name="get_availability",
        parameters={"date_str": date_str},
        required_params=["date_str"],
        param_validators={"date_str": r"\d{4}-\d{2}-\d{2}"},
    )


def generate_appointment_tool_params(
    first_name: str,
    email: str,
    phone: str,
    time: str,
    last_name: Optional[str] = None,
    date: Optional[str] = None,
) -> ExpectedToolParameters:
    """
    Generate expected parameters for create_appointment tool.

    Args:
        first_name: Expected first name
        email: Expected email
        phone: Expected phone
        time: Expected time (HH:MM format)
        last_name: Expected last name (optional)
        date: Expected date (optional)

    Returns:
        ExpectedToolParameters for the tool call
    """
    params = {
        "first_name": first_name,
        "email": email,
        "phone": phone,
        "appointment_start_time": time,
    }

    if last_name:
        params["last_name"] = last_name

    if date:
        params["appointment_date"] = date

    return ExpectedToolParameters(
        tool_name="create_appointment",
        parameters=params,
        required_params=["first_name", "email", "phone", "appointment_start_time"],
        param_validators={
            "email": r".+@.+\..+",
            "appointment_start_time": r"\d{2}:\d{2}",
        },
    )


def generate_lead_tool_params(
    first_name: str,
    last_name: str,
    email: str,
    phone: str,
) -> ExpectedToolParameters:
    """
    Generate expected parameters for create_lead tool.

    Args:
        first_name: Expected first name
        last_name: Expected last name
        email: Expected email
        phone: Expected phone

    Returns:
        ExpectedToolParameters for the tool call
    """
    return ExpectedToolParameters(
        tool_name="create_lead",
        parameters={
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
        },
        required_params=["first_name", "last_name", "email", "phone"],
        param_validators={"email": r".+@.+\..+"},
    )


def generate_knowledge_tool_params(search_text: str) -> ExpectedToolParameters:
    """
    Generate expected parameters for search_property_knowledge tool.

    Args:
        search_text: Expected search text

    Returns:
        ExpectedToolParameters for the tool call
    """
    return ExpectedToolParameters(
        tool_name="search_property_knowledge",
        parameters={"search_text": search_text},
        required_params=["search_text"],
    )


# =============================================================================
# Turn Expectation Generation from YAML Scenarios
# =============================================================================

def generate_turn_expectation_from_yaml(
    turn_data: dict[str, Any],
    turn_number: int,
) -> TurnExpectation:
    """
    Generate TurnExpectation from YAML scenario turn data.

    Args:
        turn_data: Turn data from YAML (expected_tools, expected_response_patterns, etc.)
        turn_number: The turn number

    Returns:
        TurnExpectation for the turn
    """
    expected_tools = []
    expected_keywords = ExpectedKeywords()
    should_write_to_file = None
    file_search_pattern = None

    # Parse expected tools
    yaml_tools = turn_data.get("expected_tools", [])
    for tool_data in yaml_tools:
        tool_name = tool_data.get("tool_name", "")
        params = tool_data.get("parameters", {})

        expected_tools.append(ExpectedToolParameters(
            tool_name=tool_name,
            parameters=params,
            required_params=list(params.keys()),
        ))

        # Check if tool writes to file
        if tool_name == "create_appointment":
            should_write_to_file = "tour_bookings"
            # Build search pattern from params
            if "email" in params:
                file_search_pattern = params["email"]
        elif tool_name == "create_lead":
            should_write_to_file = "leads"
            if "email" in params:
                file_search_pattern = params["email"]

    # Parse expected response patterns
    response_patterns = turn_data.get("expected_response_patterns", [])
    for pattern in response_patterns:
        # Pattern might be regex like "bedroom|unit|available"
        keywords = pattern.replace("|", " ").split()
        expected_keywords.required.extend(keywords[:2])  # First 2 as required
        expected_keywords.optional.extend(keywords[2:])  # Rest as optional

    return TurnExpectation(
        turn_number=turn_number,
        expected_tools=expected_tools,
        expected_keywords=expected_keywords,
        should_write_to_file=should_write_to_file,
        file_search_pattern=file_search_pattern,
    )


def enrich_scenario_with_expectations(scenario: dict[str, Any]) -> dict[str, Any]:
    """
    Enrich a YAML scenario with generated expectations based on data.

    Args:
        scenario: Original YAML scenario

    Returns:
        Enriched scenario with turn_expectations field
    """
    turn_expectations = []

    # Handle conversation format
    if "conversation" in scenario:
        turn_num = 0
        for msg in scenario["conversation"]:
            if msg.get("role") == "assistant":
                turn_num += 1
                expectation = generate_turn_expectation_from_yaml(msg, turn_num)

                # Enrich with data-driven keywords
                expected_tools = msg.get("expected_tools", [])
                for tool_data in expected_tools:
                    tool_name = tool_data.get("tool_name", "")
                    params = tool_data.get("parameters", {})

                    # Generate data-driven keywords
                    if tool_name == "get_available_listings":
                        bedrooms = params.get("bedrooms")
                        max_rent = params.get("max_rent")
                        data_keywords = generate_listing_search_keywords(bedrooms, max_rent)
                        expectation.expected_keywords.required.extend(data_keywords.required)
                        expectation.expected_keywords.optional.extend(data_keywords.optional)

                    elif tool_name == "get_availability":
                        date_str = params.get("date_str", "")
                        data_keywords = generate_availability_keywords(date_str)
                        expectation.expected_keywords.required.extend(data_keywords.required)
                        expectation.expected_keywords.optional.extend(data_keywords.optional)

                    elif tool_name == "create_appointment":
                        first_name = params.get("first_name", "")
                        time = params.get("appointment_start_time", "")
                        data_keywords = generate_appointment_keywords(first_name, time)
                        expectation.expected_keywords.required.extend(data_keywords.required)
                        expectation.expected_keywords.optional.extend(data_keywords.optional)

                    elif tool_name == "create_lead":
                        first_name = params.get("first_name", "")
                        data_keywords = generate_lead_keywords(first_name)
                        expectation.expected_keywords.required.extend(data_keywords.required)
                        expectation.expected_keywords.optional.extend(data_keywords.optional)

                    elif tool_name == "search_property_knowledge":
                        search_text = params.get("search_text", "")
                        data_keywords = generate_knowledge_search_keywords(search_text)
                        expectation.expected_keywords.required.extend(data_keywords.required)
                        expectation.expected_keywords.optional.extend(data_keywords.optional)

                # Deduplicate keywords
                expectation.expected_keywords.required = list(set(expectation.expected_keywords.required))
                expectation.expected_keywords.optional = list(set(expectation.expected_keywords.optional))

                turn_expectations.append(expectation)

    # Handle single-turn scenarios (user_input format)
    elif "user_input" in scenario:
        expectation = generate_turn_expectation_from_yaml(scenario, 1)

        # Enrich with data-driven keywords
        for tool_data in scenario.get("expected_tools", []):
            tool_name = tool_data.get("tool_name", "")
            params = tool_data.get("parameters", {})

            if tool_name == "get_available_listings":
                bedrooms = params.get("bedrooms")
                max_rent = params.get("max_rent")
                data_keywords = generate_listing_search_keywords(bedrooms, max_rent)
                expectation.expected_keywords.required.extend(data_keywords.required)
                expectation.expected_keywords.optional.extend(data_keywords.optional)

        expectation.expected_keywords.required = list(set(expectation.expected_keywords.required))
        expectation.expected_keywords.optional = list(set(expectation.expected_keywords.optional))

        turn_expectations.append(expectation)

    scenario["turn_expectations"] = [exp.model_dump() for exp in turn_expectations]
    return scenario


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data loaders
    "load_listings",
    "load_knowledge_base",
    # Listings analysis
    "get_listings_by_criteria",
    "get_rent_range_for_bedrooms",
    "get_available_bedroom_counts",
    # Knowledge base analysis
    "get_keywords_for_category",
    "get_answer_keywords",
    # Keyword generation
    "generate_listing_search_keywords",
    "generate_availability_keywords",
    "generate_appointment_keywords",
    "generate_lead_keywords",
    "generate_knowledge_search_keywords",
    # Tool params generation
    "generate_listing_tool_params",
    "generate_availability_tool_params",
    "generate_appointment_tool_params",
    "generate_lead_tool_params",
    "generate_knowledge_tool_params",
    # Turn expectation
    "generate_turn_expectation_from_yaml",
    "enrich_scenario_with_expectations",
]
