import asyncio
import json
import tempfile
import unittest

from kvsnap.gateway import DeterministicEngine, SessionGateway, fixture_identity
from kvsnap.gateway.http import GatewayHTTP
from kvsnap.store import SnapshotStore


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(self.tmp.name)
        self.gateway = SessionGateway(DeterministicEngine(), self.store)
        self.app = GatewayHTTP(self.gateway, fixture_identity())

    async def asyncTearDown(self):
        self.store.close()
        self.tmp.cleanup()

    async def request(self, method, path, data=None):
        class CapturingWriter:
            def __init__(self):
                self.data = bytearray()
            def write(self, data):
                self.data.extend(data)
            async def drain(self):
                await asyncio.sleep(0)
            def close(self):
                pass
            async def wait_closed(self):
                pass
        reader, writer = asyncio.StreamReader(), CapturingWriter()
        payload = json.dumps(data or {}).encode()
        reader.feed_data(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload)
        reader.feed_eof()
        await self.app.handle(reader, writer)
        raw = bytes(writer.data)
        headers, body = raw.split(b"\r\n\r\n", 1)
        return headers.decode(), body.decode()

    async def test_http_pause_resume_flow_and_conflict(self):
        headers, body = await self.request("POST", "/sessions", {"seed": 1, "max_tokens": 4, "durability": "durable"})
        self.assertIn("201 Created", headers)
        session = json.loads(body)
        sid = session["session_id"]
        headers, body = await self.request("POST", f"/sessions/{sid}/turns", {"token_ids": [1, 2], "expected_generation": 0, "max_tokens": 2})
        self.assertIn("application/x-ndjson", headers)
        self.assertEqual([json.loads(line)["seq"] for line in body.splitlines()], [1, 2])
        headers, body = await self.request("POST", f"/sessions/{sid}/pause", {"expected_generation": 1})
        self.assertEqual(json.loads(body)["state"], "Durable")
        headers, body = await self.request("POST", f"/sessions/{sid}/resume", {"expected_generation": 0, "ack_seq": 2})
        self.assertIn("409 Conflict", headers)
        headers, body = await self.request("POST", f"/sessions/{sid}/resume", {"expected_generation": 1, "ack_seq": 1})
        self.assertEqual([json.loads(line)["seq"] for line in body.splitlines()], [2, 3, 4])

    async def test_unsupported_sampling_and_missing_generation(self):
        headers, body = await self.request("POST", "/sessions", {"sampling": {"presence_penalty": 1}})
        self.assertIn("400 Bad Request", headers)
        self.assertIn("penalties", body)
        headers, body = await self.request("GET", "/health")
        self.assertEqual(json.loads(body)["evidence_tier"], "cpu_fixture")

    async def test_session_flow_over_a_loopback_socket(self):
        # The same handler behind a real asyncio server, with requests parsed off the wire.
        server = await asyncio.start_server(self.app.handle, "127.0.0.1", 0, limit=16384)
        port = server.sockets[0].getsockname()[1]

        async def call(method, path, body):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            payload = json.dumps(body).encode()
            writer.write(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
                         f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            await writer.drain()
            raw = await asyncio.wait_for(reader.read(), 10)
            writer.close()
            await writer.wait_closed()
            headers, data = raw.split(b"\r\n\r\n", 1)
            status = int(headers.split(b" ", 2)[1])
            return status, [json.loads(line) for line in data.splitlines()] if b"x-ndjson" in headers else json.loads(data)

        async with server:
            status, session = await call("POST", "/sessions", {"max_tokens": 8, "durability": "durable", "seed": 13})
            self.assertEqual(status, 201)
            sid = session["session_id"]
            status, outputs = await call("POST", f"/sessions/{sid}/turns",
                                         {"expected_generation": 0, "token_ids": [1, 2, 3], "max_tokens": 3})
            self.assertEqual((status, len(outputs), outputs[-1]["seq"]), (200, 3, 3))
            status, snapshot = await call("POST", f"/sessions/{sid}/pause", {"expected_generation": 1})
            self.assertEqual((status, snapshot["state"]), (200, "Durable"))
            status, _ = await call("POST", f"/snapshots/{snapshot['snapshot_id']}/fork", {"expected_generation": 1})
            self.assertEqual(status, 201)
            status, resumed = await call("POST", f"/sessions/{sid}/resume", {"expected_generation": 1, "ack_seq": 1})
            self.assertEqual((status, [x["seq"] for x in resumed]), (200, list(range(2, 9))))
            status, _ = await call("POST", f"/sessions/{sid}/pause", {"expected_generation": 1})
            self.assertEqual(status, 409)


if __name__ == "__main__":
    unittest.main()
