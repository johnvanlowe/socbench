"""Tests for Model Context Protocol (MCP) server."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from socbench.config import load_benchmark_config
from socbench.mcp import MCP_PROTOCOL_VERSION, MCPServer
from socbench.tools.catalog import build_default_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "benchmark_config.yaml"


@pytest.fixture
def mcp_server(built_index) -> MCPServer:
    cfg = load_benchmark_config(CONFIG_PATH)
    registry = build_default_registry(
        persona_allowlist=cfg.agent.persona_tool_allowlist()
    )
    return MCPServer(
        index_dir=built_index.index_dir,
        registry=registry,
    )


def test_mcp_initialize(mcp_server: MCPServer) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = mcp_server.handle_message(req)
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert "tools" in resp["result"]["capabilities"]
    assert resp["result"]["serverInfo"]["name"] == "socbench"


def test_mcp_ping(mcp_server: MCPServer) -> None:
    req = {"jsonrpc": "2.0", "id": 42, "method": "ping"}
    resp = mcp_server.handle_message(req)
    assert resp == {"jsonrpc": "2.0", "id": 42, "result": {}}


def test_mcp_notification_initialized(mcp_server: MCPServer) -> None:
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = mcp_server.handle_message(req)
    assert resp is None


def test_mcp_tools_list_all(mcp_server: MCPServer) -> None:
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = mcp_server.handle_message(req)
    assert resp is not None
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "list_pairs" in tool_names
    assert "get_flows" in tool_names
    assert "submit_assessment" in tool_names
    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "inputSchema" in t


def test_mcp_tools_list_persona_scoped(built_index) -> None:
    cfg = load_benchmark_config(CONFIG_PATH)
    registry = build_default_registry(
        persona_allowlist=cfg.agent.persona_tool_allowlist()
    )
    server = MCPServer(
        index_dir=built_index.index_dir,
        registry=registry,
        persona="soc_analyst",
    )
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    resp = server.handle_message(req)
    assert resp is not None
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    # soc_analyst has 5 tools in benchmark_config.yaml
    expected = set(cfg.agent.personas["soc_analyst"].tools)
    assert tool_names == expected
    assert "rarity_stats" not in tool_names


def test_mcp_tools_call_success(mcp_server: MCPServer) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "list_pairs",
            "arguments": {"limit": 5},
        },
    }
    resp = mcp_server.handle_message(req)
    assert resp is not None
    assert resp["id"] == 4
    assert resp["result"]["isError"] is False
    content = resp["result"]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    payload = json.loads(content[0]["text"])
    assert "items" in payload
    assert isinstance(payload["items"], list)
    assert payload["returned"] == 5


def test_mcp_tools_call_persona_denied(built_index) -> None:
    cfg = load_benchmark_config(CONFIG_PATH)
    registry = build_default_registry(
        persona_allowlist=cfg.agent.persona_tool_allowlist()
    )
    server = MCPServer(
        index_dir=built_index.index_dir,
        registry=registry,
        persona="soc_analyst",
    )
    # rarity_stats is NOT allowed for soc_analyst
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "rarity_stats",
            "arguments": {},
        },
    }
    resp = server.handle_message(req)
    assert resp is not None
    assert resp["result"]["isError"] is True
    assert "not allowed for persona" in resp["result"]["content"][0]["text"]


def test_mcp_tools_call_schema_violation(mcp_server: MCPServer) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "get_flows",
            "arguments": {"flow_ids": "not-a-list"},
        },
    }
    resp = mcp_server.handle_message(req)
    assert resp is not None
    assert resp["result"]["isError"] is True
    assert "ToolSchemaViolation" in resp["result"]["content"][0]["text"]


def test_mcp_run_stdio(mcp_server: MCPServer) -> None:
    input_lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]
    in_stream = io.StringIO("\n".join(input_lines) + "\n")
    out_stream = io.StringIO()

    mcp_server.run_stdio(in_stream=in_stream, out_stream=out_stream)
    output_lines = [
        json.loads(line) for line in out_stream.getvalue().splitlines() if line.strip()
    ]
    # ping response and tools/list response (notification yields no response)
    assert len(output_lines) == 2
    assert output_lines[0]["id"] == 1
    assert output_lines[1]["id"] == 2
