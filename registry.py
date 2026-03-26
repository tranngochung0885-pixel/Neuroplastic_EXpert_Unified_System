"""
nexus/tools/registry.py
========================
ToolRegistry: extensible tool dispatch with schema introspection.

Built-in tools:
  calculator     — safe math expression evaluation
  memory_search  — query episodic memory
  system_status  — runtime introspection
  web_search     — optional web search (via requests/httpx)

Tools are registered async callables. Schema is auto-generated for LLM use.
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any, Callable, Dict, List, Optional

from nexus.core.observability import LOG, METRICS


class ToolRegistry:
    """
    Async tool dispatcher with:
    - Named registration with description + parameter schema
    - Metrics tracking per-tool
    - Error isolation (exceptions → error dict, not re-raise)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tools[name] = {
            "fn": fn,
            "description": description,
            "parameters": parameters or {},
        }

    def has(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name!r}")
        METRICS.inc_tool(name)
        fn = self._tools[name]["fn"]
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(**kwargs)
            else:
                return await asyncio.to_thread(fn, **kwargs)
        except Exception as exc:
            LOG.warning("tool.call_error", tool=name, error=str(exc))
            METRICS.inc_error("tool")
            return {"error": str(exc)}

    async def safe_call(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """Call tool, always returning a dict (error on failure)."""
        try:
            result = await self.call(name, **kwargs)
            if isinstance(result, dict):
                return result
            return {"result": result}
        except Exception as exc:
            return {"error": str(exc)}

    def schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "description": meta["description"],
                "parameters": meta["parameters"],
            }
            for name, meta in sorted(self._tools.items())
        ]

    def detect_tool_call(self, text: str) -> Optional[tuple]:
        """Detect inline tool call syntax: [TOOL:name args]."""
        calc_m = re.search(r"\[CALC:\s*([^\]]+)\]", text)
        if calc_m:
            return ("calculator", {"expression": calc_m.group(1).strip()})
        mem_m = re.search(r"\[MEMORY:\s*([^\]]+)\]", text)
        if mem_m:
            return ("memory_search", {"query": mem_m.group(1).strip()})
        return None


# ---------------------------------------------------------------------------
# Default tool implementations
# ---------------------------------------------------------------------------

_ALLOWED_MATH_NAMES = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
}

_SAFE_CHARS = re.compile(r"^[0-9\+\-\*/\(\)\.\s\%\,a-zA-Z_]+$")


def _calculator(expression: str) -> Dict[str, Any]:
    """Safe math expression evaluator. Allows only whitelisted names."""
    expr = expression.strip()
    if not _SAFE_CHARS.match(expr):
        return {"error": "Unsafe characters in expression.", "expression": expr}
    try:
        result = eval(expr, _ALLOWED_MATH_NAMES)  # noqa: S307
        return {"expression": expr, "result": result}
    except Exception as exc:
        return {"error": str(exc), "expression": expr}


async def _system_status_stub() -> Dict[str, Any]:
    """Placeholder — replaced with runtime.status() at startup."""
    return {"status": "system_status not yet connected"}


def build_default_registry() -> ToolRegistry:
    """
    Build and return a ToolRegistry with default built-in tools.
    memory_search and system_status will be monkey-patched by NexusBrain.
    """
    reg = ToolRegistry()

    reg.register(
        "calculator",
        _calculator,
        description="Evaluate a safe mathematical expression.",
        parameters={
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        },
    )

    reg.register(
        "system_status",
        _system_status_stub,
        description="Return current system status and memory stats.",
        parameters={},
    )

    return reg
