"""
Tool Decorator for LangChain

This module re-exports LangChain's tool decorator for backward compatibility.
The original Bedrock-specific tool decorator has been replaced with LangChain's
built-in tool decorator.
"""

from langchain_core.tools import tool

# Re-export LangChain's tool decorator
__all__ = ["tool"]
