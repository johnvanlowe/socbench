"""Model Context Protocol (MCP) server for SOCBench tools.

Exposes SOCBench's content-addressed dataset indexes and read-only tool catalog
over the standard JSON-RPC 2.0 Model Context Protocol (MCP).

Allows external agents (Claude Code, OpenHands, SWE-agent, Cline, Cursor, etc.)
to connect directly to SOCBench environments as an MCP server over stdio.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

from socbench._version import __version__
from socbench.tools import ToolContext, ToolRegistry, ToolSchemaViolation
from socbench.tools.catalog import build_default_registry

log = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPServer:
    """JSON-RPC 2.0 Model Context Protocol (MCP) server for SOCBench tools."""

    def __init__(
        self,
        index_dir: Path,
        registry: ToolRegistry | None = None,
        persona: str | None = None,
        persona_allowlist: dict[str, set[str]] | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(f"index directory not found at {self.index_dir}")

        self.registry = registry or build_default_registry(persona_allowlist=persona_allowlist)
        self.persona = persona
        self.tool_context = ToolContext(index_dir=self.index_dir)

    def available_tools(self) -> list[Any]:
        """Return the list of tools accessible in this server session."""
        if self.persona is not None:
            return self.registry.tools_for_persona(self.persona)
        return [self.registry.get(name) for name in self.registry.names()]

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:  # noqa: PLR0911, PLR0912
        """Process a single JSON-RPC 2.0 message."""
        jsonrpc = message.get("jsonrpc")
        if jsonrpc != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32600, "message": "Invalid Request: jsonrpc must be '2.0'"},
            }

        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {}) or {}

        # Notifications (no id)
        if msg_id is None:
            if method in {"notifications/initialized", "initialized"}:
                log.info("MCP client initialized notification received")
            return None

        # RPC Requests
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "socbench",
                        "version": __version__,
                    },
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        if method == "tools/list":
            tools_data = []
            for tool in self.available_tools():
                tools_data.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.args_schema,
                    }
                )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": tools_data},
            }

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {}) or {}

            if not tool_name or not isinstance(tool_name, str):
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": "Invalid params: 'name' is required"},
                }

            if self.persona is not None and not self.registry.is_allowed(self.persona, tool_name):
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Error: Tool {tool_name!r} is not allowed "
                                    f"for persona {self.persona!r}."
                                ),
                            }
                        ],
                        "isError": True,
                    },
                }

            try:
                tool = self.registry.get(tool_name)
            except KeyError:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: unknown tool {tool_name!r}",
                    },
                }

            try:
                payload = tool(dict(arguments), self.tool_context)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(payload, indent=2, sort_keys=True),
                            }
                        ],
                        "isError": False,
                    },
                }
            except ToolSchemaViolation as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"ToolSchemaViolation: {exc}",
                            }
                        ],
                        "isError": True,
                    },
                }
            except Exception as exc:
                log.exception("Error executing tool %s", tool_name)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"ToolExecutionError: {type(exc).__name__}: {exc}",
                            }
                        ],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method!r}"},
        }

    def run_stdio(self, in_stream: TextIO | None = None, out_stream: TextIO | None = None) -> None:
        """Run the JSON-RPC line-delimited message loop over standard streams."""
        reader = in_stream or sys.stdin
        writer = out_stream or sys.stdout

        for raw_line in reader:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                writer.write(json.dumps(err_resp) + "\n")
                writer.flush()
                continue

            response = self.handle_message(msg)
            if response is not None:
                writer.write(json.dumps(response) + "\n")
                writer.flush()


__all__ = ["MCP_PROTOCOL_VERSION", "MCPServer"]
