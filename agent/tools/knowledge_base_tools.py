"""
Knowledge Base Tools for Property Agent

This module contains tools for searching property-related knowledge.
Loads knowledge base entries from a JSON file and provides keyword-based search
with relevance scoring.
"""

import json
import os
import re
from typing import Annotated, Any, Dict, List, Optional
from langchain_core.tools import tool


# Path to the knowledge base JSON file
KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "knowledge_base.json"
)

# Cache for loaded knowledge base
_knowledge_base_cache: Optional[Dict[str, Any]] = None


def load_knowledge_base() -> Dict[str, Any]:
    """
    Load the knowledge base from the JSON file.
    Uses caching to avoid repeated file reads.

    Returns:
        Dictionary containing the knowledge base data
    """
    global _knowledge_base_cache

    if _knowledge_base_cache is not None:
        return _knowledge_base_cache

    try:
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            # Return empty structure if file doesn't exist
            return {
                "property_name": "Unknown Property",
                "entries": []
            }

        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            _knowledge_base_cache = json.load(f)
            return _knowledge_base_cache

    except json.JSONDecodeError as e:
        print(f"Error parsing knowledge base JSON: {e}")
        return {"property_name": "Unknown Property", "entries": []}
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return {"property_name": "Unknown Property", "entries": []}


def reload_knowledge_base() -> Dict[str, Any]:
    """
    Force reload the knowledge base from disk.
    Useful when the JSON file has been updated.

    Returns:
        Dictionary containing the knowledge base data
    """
    global _knowledge_base_cache
    _knowledge_base_cache = None
    return load_knowledge_base()


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase words, removing punctuation.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase tokens
    """
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter empty strings
    tokens = [t.strip() for t in text.split() if t.strip()]
    return tokens


def calculate_relevance_score(search_tokens: List[str], entry: Dict[str, Any]) -> float:
    """
    Calculate a relevance score for an entry based on keyword matching.

    Scoring factors:
    - Keyword matches (highest weight)
    - Question text matches
    - Answer text matches
    - Category matches

    Args:
        search_tokens: List of tokenized search terms
        entry: Knowledge base entry to score

    Returns:
        Relevance score between 0.0 and 1.0
    """
    if not search_tokens:
        return 0.0

    score = 0.0
    max_possible_score = len(search_tokens) * 4  # 4 scoring dimensions

    # Get entry components
    keywords = [kw.lower() for kw in entry.get("keywords", [])]
    question_tokens = tokenize(entry.get("question", ""))
    answer_tokens = tokenize(entry.get("answer", ""))
    category = entry.get("category", "").lower()

    for token in search_tokens:
        # Keyword matches (weight: 1.5)
        if token in keywords:
            score += 1.5
        else:
            # Partial keyword match (weight: 0.75)
            for kw in keywords:
                if token in kw or kw in token:
                    score += 0.75
                    break

        # Question matches (weight: 1.0)
        if token in question_tokens:
            score += 1.0

        # Answer matches (weight: 0.5)
        if token in answer_tokens:
            score += 0.5

        # Category match (weight: 1.0)
        if token == category or token in category:
            score += 1.0

    # Normalize to 0-1 range
    normalized_score = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0

    # Apply a minimum threshold - if there were any matches, ensure a minimum score
    if score > 0:
        normalized_score = max(normalized_score, 0.1)

    return round(normalized_score, 3)


def search_entries(
    search_text: str,
    category_filter: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search knowledge base entries by text query.

    Args:
        search_text: The search query
        category_filter: Optional category to filter by
        limit: Maximum number of results to return

    Returns:
        List of matching entries with relevance scores
    """
    kb = load_knowledge_base()
    entries = kb.get("entries", [])

    if not entries:
        return []

    # Tokenize search text
    search_tokens = tokenize(search_text)

    if not search_tokens:
        return []

    # Score all entries
    scored_entries = []

    for entry in entries:
        # Apply category filter if specified
        if category_filter:
            if entry.get("category", "").lower() != category_filter.lower():
                continue

        score = calculate_relevance_score(search_tokens, entry)

        if score > 0:
            scored_entries.append({
                "id": entry.get("id"),
                "category": entry.get("category"),
                "question": entry.get("question"),
                "answer": entry.get("answer"),
                "relevance_score": score
            })

    # Sort by relevance score (highest first)
    scored_entries.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Return top results
    return scored_entries[:limit]


# Default response for queries that don't match
DEFAULT_KNOWLEDGE = {
    "id": 0,
    "category": "general",
    "question": "General Property Information",
    "answer": "Welcome to our property! For specific questions about our amenities, policies, leasing, or other topics, please contact our leasing office at (555) 123-4567 or email leasing@sunsetheights.com. We're happy to help!",
    "relevance_score": 0.1
}


@tool
def search_property_knowledge(
    search_text: Annotated[str, "REQUIRED: User's query text to search for relevant property information"],
    category: Annotated[Optional[str], "Optional category filter: amenities, neighborhood, leasing, rent_payment, policies, utilities, maintenance, move_in_out"] = None,
) -> Dict[str, Any]:
    """Search the property knowledge base for information about amenities, policies, leasing, rent payments, utilities, maintenance, neighborhood, and move-in/move-out procedures.

    This implementation:
    - Loads knowledge base from JSON file
    - Performs keyword-based matching with relevance scoring
    - Returns top 5 most relevant results
    - Handles edge cases (empty search, no matches, etc.)
    """
    # Handle empty or whitespace-only search
    if not search_text or not search_text.strip():
        return {
            "success": False,
            "query": search_text,
            "category_filter": category,
            "results": [DEFAULT_KNOWLEDGE],
            "total_results": 1,
            "message": "Please provide a search query to find relevant information."
        }

    # Validate category if provided
    valid_categories = [
        "amenities", "neighborhood", "leasing", "rent_payment",
        "policies", "utilities", "maintenance", "move_in_out"
    ]

    if category and category.lower() not in valid_categories:
        return {
            "success": False,
            "query": search_text,
            "category_filter": category,
            "results": [],
            "total_results": 0,
            "message": f"Invalid category '{category}'. Valid categories are: {', '.join(valid_categories)}"
        }

    # Perform the search
    results = search_entries(
        search_text=search_text.strip(),
        category_filter=category,
        limit=5
    )

    # If no results found, return default response
    if not results:
        kb = load_knowledge_base()
        property_name = kb.get("property_name", "our property")

        return {
            "success": True,
            "query": search_text,
            "category_filter": category,
            "results": [{
                **DEFAULT_KNOWLEDGE,
                "answer": f"I couldn't find specific information about '{search_text}' in our knowledge base. For detailed questions about {property_name}, please contact our leasing office at (555) 123-4567 or email leasing@sunsetheights.com."
            }],
            "total_results": 1,
            "message": f"No exact matches found for your query. Showing general information."
        }

    return {
        "success": True,
        "query": search_text,
        "category_filter": category,
        "results": results,
        "total_results": len(results),
        "message": f"Found {len(results)} relevant knowledge base entries"
    }


def get_all_categories() -> List[str]:
    """
    Get all unique categories from the knowledge base.

    Returns:
        List of category names
    """
    kb = load_knowledge_base()
    entries = kb.get("entries", [])
    categories = set(entry.get("category", "") for entry in entries)
    return sorted(list(categories))


def get_entries_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Get all entries for a specific category.

    Args:
        category: The category to filter by

    Returns:
        List of entries in the specified category
    """
    kb = load_knowledge_base()
    entries = kb.get("entries", [])
    return [
        entry for entry in entries
        if entry.get("category", "").lower() == category.lower()
    ]


def get_entry_count() -> int:
    """
    Get the total number of entries in the knowledge base.

    Returns:
        Total entry count
    """
    kb = load_knowledge_base()
    return len(kb.get("entries", []))
