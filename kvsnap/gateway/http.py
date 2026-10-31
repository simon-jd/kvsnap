"""Small bounded HTTP/1.1 transport with NDJSON token streams.

Designed for a local single-user gateway, not an Internet-facing HTTP server.
All sessions and engine tasks stay on one asyncio event loop.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

from kvsnap.models import ContractError, Durability, KVIdentity, SamplingConfig
from .engine import fixture_identity
from .service import GenerationConflict, SessionGateway, StateConflict, TurnHandle


class GatewayHTTP:
    MAX_BODY = 1024 * 1024

    def __init__(self, gateway: SessionGateway, default_identity: KVIdentity | None = None):
        self.gateway = gateway
        self.default_identity = default_identity

    async def dispatch(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        parts = [p for p in urlsplit(path).path.split("/") if p]
        if method == "GET" and parts == ["health"]:
            return 200, {"status": "ok", "evidence_tier": getattr(self.gateway.engine, "evidence_tier", "unverified")}
        if method == "POST" and parts == ["sessions"]:
            identity = KVIdentity.from_dict(body["identity"]) if "identity" in body else self.default_identity
            if identity is None:
                raise ContractError("identity is required")
            session = await self.gateway.create(identity, sampling=SamplingConfig.from_dict(body.get("sampling", {})),
                                                seed=body.get("seed"), max_tokens=body.get("max_tokens", 256),
                                                durability=body.get("durability", "volatile"))
            return 201, session.to_dict()
        if len(parts) == 3 and parts[0] == "snapshots" and method == "POST":
            if parts[2] == "fork":
                session = await self.gateway.fork(parts[1], expected_generation=body["expected_generation"])
                return 201, session.to_dict()
            if parts[2] == "recover":
                # Register a committed session after restart; this never claims
                # an execution generation or emits tokens until resume.
                return 200, self.gateway.recover(parts[1]).to_dict()
        if len(parts) >= 2 and parts[0] == "sessions":
            session_id = parts[1]
            if len(parts) == 2:
                if method == "GET":
                    return 200, self.gateway.get(session_id).to_dict()
                if method == "DELETE":
                    await self.gateway.close(session_id, expected_generation=body["expected_generation"])
                    return 200, self.gateway.get(session_id).to_dict()
            if len(parts) == 3 and method == "POST":
                generation = body["expected_generation"]
                if parts[2] == "turns":
                    return 200, await self.gateway.start_turn(session_id, body["token_ids"],
                                                              expected_generation=generation, max_tokens=body.get("max_tokens"))
                if parts[2] == "pause":
                    manifest = await self.gateway.pause(session_id, expected_generation=generation,
                                                        durability=body.get("durability"))
                    return 200, manifest.to_dict()
                if parts[2] == "resume":
                    return 200, await self.gateway.resume(session_id, expected_generation=generation,
                                                         ack_seq=body["ack_seq"], append_tokens=body.get("append_tokens", []),
                                                         max_tokens=body.get("max_tokens"),
                                                         identity=KVIdentity.from_dict(body["identity"]) if "identity" in body else None)
        return 404, {"error": "route not found"}

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        started = False
        try:
            headers_raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15)
            lines = headers_raw.decode("ascii").split("\r\n")
            method, path, version = lines[0].split(" ")
            if version != "HTTP/1.1":
                raise ContractError("HTTP/1.1 is required")
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    if key in headers:
                        raise ContractError("duplicate headers are not accepted")
                    headers[key] = value.strip()
            if "transfer-encoding" in headers:
                raise ContractError("chunked request bodies are not supported")
            length = int(headers.get("content-length", "0"))
            if not 0 <= length <= self.MAX_BODY:
                raise ContractError("request body exceeds 1 MiB")
            raw = await asyncio.wait_for(reader.readexactly(length), timeout=15)
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise ContractError("request body must be an object")
            status, response = await self.dispatch(method, path, body)
            if isinstance(response, TurnHandle):
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n"
                             + f"X-Kvsnap-Generation: {response.generation}\r\n\r\n".encode())
                started = True
                await writer.drain()
                async for token in response:
                    writer.write(json.dumps(token.to_dict(), separators=(",", ":")).encode() + b"\n")
                    await writer.drain()
            else:
                await self._respond(writer, status, response)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # Engine output remains in the replay log after disconnect.
        except Exception as exc:
            if started:
                writer.write(json.dumps({"error": str(exc)}).encode() + b"\n")
                try:
                    await writer.drain()
                except ConnectionError:
                    pass
            else:
                status = 409 if isinstance(exc, (GenerationConflict, StateConflict)) else 400
                if isinstance(exc, KeyError) and "unknown session" in str(exc):
                    status = 404
                await self._respond(writer, status, {"error": str(exc), "type": type(exc).__name__})
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, status: int, body: Any) -> None:
        payload = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        reason = {200: "OK", 201: "Created", 400: "Bad Request", 404: "Not Found", 409: "Conflict"}.get(status, "Error")
        writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
        await writer.drain()

    async def serve(self, host: str = "127.0.0.1", port: int = 8030) -> None:
        server = await asyncio.start_server(self.handle, host, port, limit=16384)
        async with server:
            await server.serve_forever()
