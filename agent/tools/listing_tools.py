"""
Listing Tools for Property Agent

This module contains tools for searching and retrieving property listings.
Listings are loaded from a JSON data file.
"""

import json
import os
from typing import Annotated, Any, Dict, List, Optional
from langchain_core.tools import tool


# Path to the listings JSON file
LISTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "listings.json"
)


def load_listings() -> List[Dict[str, Any]]:
    """
    Load listings from the JSON file.

    Returns:
        List of listing dictionaries, or empty list if file not found or error.
    """
    try:
        if not os.path.exists(LISTINGS_FILE):
            return []

        with open(LISTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("listings", [])
    except (json.JSONDecodeError, IOError, KeyError):
        return []


@tool
def get_available_listings(
    bedrooms: Annotated[List[int], "REQUIRED: List of bedroom counts to filter by (e.g., [1, 2] for 1BR or 2BR units). Use [0] for studios."],
    max_rent: Annotated[Optional[float], "OPTIONAL: Maximum monthly rent in dollars"] = None,
) -> Dict[str, Any]:
    """Get available listings for a property based on search criteria.

    Loads listings from the JSON data file and filters based on:
    - Bedroom count (required): List of acceptable bedroom counts
    - Maximum rent (optional): Filter out listings above this price

    Returns top 5 matching listings sorted by rent (lowest first).
    """
    # Load listings from JSON file
    all_listings = load_listings()

    # Handle file not found or empty listings
    if not all_listings:
        return {
            "success": False,
            "message": "Unable to load listings data. Please try again later.",
            "listings": [],
            "total_count": 0
        }

    # Filter listings based on criteria
    results = []

    # Convert bedrooms to integers for comparison
    bedroom_filter = [int(b) for b in bedrooms]

    for listing in all_listings:
        # Check if bedroom count matches
        if listing.get("bedrooms") not in bedroom_filter:
            continue

        # Check max rent filter if specified
        if max_rent is not None and listing.get("rent", 0) > max_rent:
            continue

        results.append(listing)

    # Handle no matches found
    if not results:
        bedroom_str = ", ".join(str(b) if b > 0 else "Studio" for b in bedroom_filter)
        rent_str = f" under ${max_rent:,.0f}/month" if max_rent else ""
        return {
            "success": True,
            "message": f"No listings found matching your criteria ({bedroom_str}{rent_str}). Try adjusting your search.",
            "listings": [],
            "total_count": 0
        }

    # Sort by rent (lowest first) and return top 5
    results.sort(key=lambda x: x.get("rent", 0))
    top_results = results[:5]

    return {
        "success": True,
        "message": f"Found {len(results)} available listing(s). Showing top {len(top_results)} results.",
        "listings": top_results,
        "total_count": len(results)
    }
