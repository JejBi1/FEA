"""Starting the REST API: python -m fea_core.rest [--host 127.0.0.1] [--port 8765] [--workdir …] [--token … | --no-token].

A token is required by default even on the loopback: without --token / FEA_REST_TOKEN the per-user token from
`<user data>/FEAStudio/tokens/rest.token` (0600, created on the first start) is used – the CAD adapters read it from there.
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import sys

LOCAL = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(host: str) -> bool:
    if host in LOCAL:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fea_core.rest", description="FEA Studio REST API (for CAD adapters)")
    ap.add_argument("--host", default="127.0.0.1", help="address (default 127.0.0.1 = this computer only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workdir", help="root of the work folders (default the user data folder, FEAStudio/mcp, or FEA_MCP_WORKDIR)")
    ap.add_argument("--token", default=os.environ.get("FEA_REST_TOKEN") or None,
                    help="require the header Authorization: Bearer <token> (default variable FEA_REST_TOKEN, "
                         "otherwise the per-user token file FEAStudio/tokens/rest.token, created on first start)")
    ap.add_argument("--no-token", action="store_true", default=_env_flag("FEA_REST_NO_TOKEN"),
                    help="loopback only: no token (any local process may call the tools; variable FEA_REST_NO_TOKEN=1)")
    ap.add_argument("--log-level", default="warning", choices=("critical", "error", "warning", "info", "debug"))
    a = ap.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if any(x == "--token" or x.startswith("--token=") for x in (argv if argv is not None else sys.argv[1:])):
        print("Warning: other processes see the token on the command line (process list) – prefer the variable FEA_REST_TOKEN.", file=sys.stderr)
    loopback = _is_loopback(a.host)
    if not loopback and not a.token:
        print(f"Refused: address {a.host} is reachable from other computers – give --token (or FEA_REST_TOKEN). The server runs calculations and reads files, without a token it may run only on 127.0.0.1.", file=sys.stderr)
        return 2
    token_note = ""
    if a.no_token and not a.token:
        token_note = ", no token (--no-token)"
    elif not a.token:
        from fea_core.ipc import ensure_server_token, server_token_path
        a.token = ensure_server_token("rest")
        token_note = f", token required – in {server_token_path('rest')}"
    else:
        token_note = ", token required"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from fea_core import modules
    try:
        modules.require("mcp")
    except modules.ModuleMissing as e:
        print(e, file=sys.stderr)
        return 3
    import uvicorn
    from fea_core.rest.app import create_app
    # on the loopback the Host header stays checked also with a token (DNS rebinding); outside it the token protects
    app = create_app(base_dir=a.workdir, token=a.token, allowed_hosts=({a.host} if loopback else None))
    import threading                      # cleaning of old automatic working folders (FEA_MCP_KEEP_DAYS)
    from fea_core.mcp.session import cleanup_workdirs
    threading.Thread(target=cleanup_workdirs, args=(app.state.session.base_dir,), daemon=True,
                     name="fea-rest cleanup").start()
    print(f"FEA Studio REST API: http://{a.host}:{a.port}/api/status  (documentation /docs, work folder {app.state.session.base_dir}{token_note})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level=a.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
