"""The five defense layers, implemented against any target.

Each layer is a pure function over data so that Phase 8 can point the loop at a
second target without reimplementing the stack.
"""

from crucible.defenses.layers.context_layer import RenderedContext, render_context
from crucible.defenses.layers.input_layer import InputVerdict, inspect_input
from crucible.defenses.layers.output_layer import OutputVerdict, inspect_output
from crucible.defenses.layers.prompt_layer import harden_system_prompt
from crucible.defenses.layers.structural import ToolDecision, decide_tool_call

__all__ = [
    "InputVerdict",
    "OutputVerdict",
    "RenderedContext",
    "ToolDecision",
    "decide_tool_call",
    "harden_system_prompt",
    "inspect_input",
    "inspect_output",
    "render_context",
]
