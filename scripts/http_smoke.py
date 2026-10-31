#!/usr/bin/env python3
"""Loopback socket smoke test for the actual HTTP transport, CPU fixture only."""
import asyncio
import argparse
import json
from pathlib import Path
import tempfile

from kvsnap.gateway.engine import DeterministicEngine, fixture_identity
from kvsnap.gateway.http import GatewayHTTP
from kvsnap.gateway.service import SessionGateway
from kvsnap.store import SnapshotStore


async def main():
    with tempfile.TemporaryDirectory(prefix="kvsnap-http-") as root, SnapshotStore(root) as store:
        http = GatewayHTTP(SessionGateway(DeterministicEngine(), store), fixture_identity())
        server = await asyncio.start_server(http.handle, "127.0.0.1", 0, limit=16384)
        port = server.sockets[0].getsockname()[1]

        async def call(method, path, body):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            payload = json.dumps(body).encode()
            writer.write(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload)
            await writer.drain()
            raw = await asyncio.wait_for(reader.read(), 10)
            writer.close(); await writer.wait_closed()
            headers, data = raw.split(b"\r\n\r\n", 1)
            status = int(headers.split(b" ", 2)[1])
            result = [json.loads(line) for line in data.splitlines()] if b"x-ndjson" in headers else json.loads(data)
            return status, result

        async with server:
            status, session = await call("POST", "/sessions", {"max_tokens": 8, "durability": "durable", "seed": 13})
            assert status == 201
            sid = session["session_id"]
            status, outputs = await call("POST", f"/sessions/{sid}/turns", {"expected_generation": 0, "token_ids": [1, 2, 3], "max_tokens": 3})
            assert status == 200 and len(outputs) == 3 and outputs[-1]["seq"] == 3
            status, snapshot = await call("POST", f"/sessions/{sid}/pause", {"expected_generation": 1})
            assert status == 200 and snapshot["state"] == "Durable"
            status, child = await call("POST", f"/snapshots/{snapshot['snapshot_id']}/fork", {"expected_generation": 1})
            assert status == 201
            status, resumed = await call("POST", f"/sessions/{sid}/resume", {"expected_generation": 1, "ack_seq": 1})
            assert status == 200 and [x["seq"] for x in resumed] == list(range(2, 9))
            status, conflict = await call("POST", f"/sessions/{sid}/pause", {"expected_generation": 1})
            assert status == 409
            return {"status": "passed", "evidence_tier": "cpu_fixture_loopback_http", "operations": ["create", "stream", "pause", "fork", "resume_ack_replay", "stale_generation_rejection"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(asyncio.run(main()), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")
