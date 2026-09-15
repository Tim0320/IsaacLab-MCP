from unittest.mock import MagicMock

import pytest

from isaaclab_mcp import server
from isaaclab_mcp.tools import register_all_tools


class FakeMcp:
    def __init__(self):
        self.tools = {}

    def tool(self, name):
        def decorator(function):
            self.tools[name] = function
            return function

        return decorator


def test_registers_initial_named_tools():
    mcp = FakeMcp()

    register_all_tools(mcp)

    assert set(mcp.tools) == {
        "create_lifting_training_project",
        "design_lifting_training",
        "design_people_rl_program",
        "get_isaac_lab_capabilities",
        "get_isaac_lab_status",
        "list_isaac_lab_tasks",
        "submit_dofbot_training_run",
        "get_training_run_status",
        "cancel_training_run",
        "validate_environment_contract",
        "validate_evidence_bundle",
        "validate_lifting_training_project",
        "validate_scene_change_request",
    }


@pytest.mark.parametrize("transport", [None, "stdio"])
def test_stdio_is_default(monkeypatch, transport):
    if transport is None:
        monkeypatch.delenv("ISAACLAB_MCP_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("ISAACLAB_MCP_TRANSPORT", transport)
    run = MagicMock()
    monkeypatch.setattr(server.mcp, "run", run)

    server.main()

    run.assert_called_once_with()


@pytest.mark.parametrize("transport", ["http", "streamable-http"])
def test_http_transport_is_normalized(monkeypatch, transport):
    monkeypatch.setenv("ISAACLAB_MCP_TRANSPORT", transport)
    run = MagicMock()
    monkeypatch.setattr(server.mcp, "run", run)

    server.main()

    run.assert_called_once_with(transport="streamable-http")


def test_unknown_transport_is_rejected(monkeypatch):
    monkeypatch.setenv("ISAACLAB_MCP_TRANSPORT", "websocket")

    with pytest.raises(ValueError, match="Unsupported ISAACLAB_MCP_TRANSPORT"):
        server.main()


@pytest.mark.parametrize("value", ["*", "*.example.com", ".example.com", "https://example.com"])
def test_http_host_wildcards_and_urls_are_rejected(monkeypatch, value):
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", value)

    with pytest.raises(ValueError, match="exact host values"):
        server._transport_security_settings()
