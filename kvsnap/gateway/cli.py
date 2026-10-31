"""Local CLI. CPU fixture is an explicit choice and labelled in every report."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kvsnap.store import SnapshotStore
from kvsnap.models import KVIdentity
from .engine import DeterministicEngine, fixture_identity
from .http import GatewayHTTP
from .service import SessionGateway, freeze_startup_heap


async def demo(root: str) -> dict:
    store = SnapshotStore(root, host_capacity_bytes=64 * 1024 * 1024, disk_capacity_bytes=64 * 1024 * 1024)
    engine = DeterministicEngine()
    gateway = SessionGateway(engine, store)
    identity = fixture_identity()
    session = await gateway.create(identity, seed=12345, max_tokens=12, durability="durable")
    initial = [o.token_id async for o in await gateway.start_turn(session.session_id, [1, 2, 3], expected_generation=0, max_tokens=4)]
    manifest = await gateway.pause(session.session_id, expected_generation=1)
    child = await gateway.fork(manifest.snapshot_id, expected_generation=1)
    resumed = [o.token_id async for o in await gateway.resume(session.session_id, expected_generation=1, ack_seq=4)]
    branch = [o.token_id async for o in await gateway.resume(child.session_id, expected_generation=0, ack_seq=4, append_tokens=[777])]
    reference = await gateway.create(identity, seed=12345, max_tokens=12)
    oracle = [o.token_id async for o in await gateway.start_turn(reference.session_id, [1, 2, 3], expected_generation=0)]
    result = {"evidence_tier": "cpu_fixture", "snapshot_id": manifest.snapshot_id,
              "snapshot": manifest.to_dict(), "reference_token_ids": oracle,
              "resumed_token_ids": initial + resumed, "branch_token_ids": initial + branch,
              "token_equal": initial + resumed == oracle, "events": engine.events}
    if not result["token_equal"]:
        raise AssertionError("CPU fixture resume differs from uninterrupted oracle")
    store.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="kvsnap token-ID session gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("demo", help="run deterministic CPU fixture pause/resume/fork; no GPU claims")
    run.add_argument("--root", default=".kvsnap-demo")
    run.add_argument("--output", type=Path)
    server = sub.add_parser("serve", help="serve the local HTTP API")
    server.add_argument("--engine", choices=["cpu-fixture", "vllm"], required=True)
    server.add_argument("--identity-file", type=Path,
                        help="optional expected resolved KVIdentity; worker handshake supplies the actual identity")
    server.add_argument("--model", default="Qwen/Qwen3-1.7B")
    server.add_argument("--revision", default="70d244cc")
    server.add_argument("--max-model-len", type=int, default=32768)
    server.add_argument("--block-size", type=int, default=16)
    server.add_argument("--pinned-pool-bytes", type=int, default=24 * 1024**3)
    server.add_argument("--disk-capacity-bytes", type=int, default=200 * 1024**3)
    server.add_argument("--root", default=".kvsnap-store")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8030)
    args = parser.parse_args(argv)
    if args.command == "demo":
        result = asyncio.run(demo(args.root))
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
    else:
        store = SnapshotStore(args.root,
                              host_capacity_bytes=args.pinned_pool_bytes if args.engine == "vllm" else 64 * 1024**2,
                              disk_capacity_bytes=args.disk_capacity_bytes)
        try:
            async def serve():
                if args.engine == "cpu-fixture":
                    identity = fixture_identity()
                    engine = DeterministicEngine()
                else:
                    from kvsnap.connector.engine import VLLMEngine, create_async_llm
                    from kvsnap.connector.runtime import wait_runtime_identity
                    expected = KVIdentity.from_dict(json.loads(args.identity_file.read_text())) if args.identity_file else None
                    runtime = create_async_llm(args.root, model=expected.model_id if expected else args.model,
                                               revision=expected.model_revision if expected else args.revision,
                                               tokenizer_revision=expected.tokenizer_revision if expected else None,
                                               max_model_len=expected.position_encoding["max_model_len"] if expected else args.max_model_len,
                                               block_size=expected.block_size if expected else args.block_size,
                                               pinned_pool_bytes=args.pinned_pool_bytes,
                                               disk_capacity_bytes=args.disk_capacity_bytes)
                    try:
                        identity = await wait_runtime_identity(args.root, runtime.kvsnap_runtime_id, expected=expected)
                        engine = VLLMEngine(runtime, store)
                    except BaseException:
                        runtime.shutdown()
                        raise
                gateway = SessionGateway(engine, store)
                freeze_startup_heap()
                try:
                    await GatewayHTTP(gateway, identity).serve(args.host, args.port)
                finally:
                    if args.engine == "vllm":
                        runtime.shutdown()
            asyncio.run(serve())
        except KeyboardInterrupt:
            pass
        finally:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
