"""IsaacLab-MCP server entry point."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from isaaclab_mcp.settings import load_settings
from isaaclab_mcp.tools import register_all_tools

_INSTRUCTIONS = """
Use this server for Isaac Lab environment inspection and workflows.
The current skeleton provides read-only status and static task discovery.
Do not claim that a simulation or training job ran unless a future runtime tool returns execution evidence.
""".strip()

_LOCAL_HTTP_HOSTS = ("localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*", "[::1]", "[::1]:*")
_LOCAL_HTTP_ORIGINS = ("http://localhost:*", "http://127.0.0.1:*", "http://[::1]:*")


def _transport_security_settings() -> TransportSecuritySettings:
    allowed_hosts = list(_LOCAL_HTTP_HOSTS)
    for value in os.getenv("MCP_ALLOWED_HOSTS", "").split(","):
        host = value.strip()
        if not host:
            continue
        if "*" in host or host.startswith(".") or "://" in host or "/" in host:
            raise ValueError("MCP_ALLOWED_HOSTS must contain exact host values without schemes, paths, or wildcards")
        allowed_hosts.append(host)
        if ":" not in host or (host.startswith("[") and host.endswith("]")):
            allowed_hosts.append(f"{host}:*")

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(dict.fromkeys(allowed_hosts)),
        allowed_origins=list(_LOCAL_HTTP_ORIGINS),
    )


def create_mcp() -> FastMCP:
    """Create the configured MCP server."""
    settings = load_settings()
    return FastMCP(
        "IsaacLabMCP",
        instructions=_INSTRUCTIONS,
        host=settings.http_host,
        port=settings.http_port,
        streamable_http_path="/mcp",
        transport_security=_transport_security_settings(),
    )


mcp = create_mcp()
register_all_tools(mcp)


def main() -> None:
    """Run stdio or Streamable HTTP transport."""
    transport = load_settings().transport
    if transport == "stdio":
        mcp.run()
    elif transport in {"http", "streamable-http"}:
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(
            f"Unsupported ISAACLAB_MCP_TRANSPORT: {transport!r}; expected 'stdio', 'http', or 'streamable-http'"
        )


if __name__ == "__main__":
    main()
