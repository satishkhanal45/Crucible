"""RAG system under test, the adapter boundary, and the canary system."""

from crucible.target.adapter import (
    BehaviorSpec,
    Document,
    OutputContract,
    TargetAdapter,
    TargetCapabilities,
    TargetResponse,
    ToolCall,
    ToolSpec,
)
from crucible.target.contract import ContractCheck, validate_output_contract

__all__ = [
    "BehaviorSpec",
    "ContractCheck",
    "Document",
    "OutputContract",
    "TargetAdapter",
    "TargetCapabilities",
    "TargetResponse",
    "ToolCall",
    "ToolSpec",
    "validate_output_contract",
]
