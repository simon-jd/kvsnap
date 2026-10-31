"""Async session coordinator with generation fencing and replayable output."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import gc
import secrets
from typing import Any, AsyncIterator
import uuid

from kvsnap.models import (ContractError, Durability, KVIdentity, OutputToken,
                        SamplingConfig, SessionState, SnapshotManifest, SnapshotState, integer, tokens)
from kvsnap.store import GenerationConflict as StoreGenerationConflict
from .engine import EngineAdapter, EngineRequest, EngineSnapshot


def freeze_startup_heap() -> None:
    """Leave everything allocated at startup out of later full collections.

    The gateway process also holds the vLLM client, tokenizer and model
    config. A full collection walked all of it with the GIL held, about
    150 ms during which no session received a token. vLLM freezes its
    engine process the same way.
    """
    gc.collect()
    gc.freeze()


class GenerationConflict(ContractError):
    pass


class StateConflict(ContractError):
    pass


@dataclass
class Session:
    session_id: str
    identity: KVIdentity
    sampling: SamplingConfig
    seed: int
    remaining_max_tokens: int
    durability: Durability
    generation: int = 0
    state: SessionState = SessionState.ACTIVE
    token_ids: list[int] = field(default_factory=list)
    num_kv_tokens: int = 0
    outputs: list[OutputToken] = field(default_factory=list)
    snapshot_id: str | None = None
    request: EngineRequest | None = None
    task: asyncio.Task | None = field(default=None, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    stop_reached: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "generation": self.generation,
                "state": self.state.value, "seed": self.seed,
                "remaining_max_tokens": self.remaining_max_tokens,
                "token_ids": list(self.token_ids), "num_kv_tokens": self.num_kv_tokens,
                "pending_tokens": self.token_ids[self.num_kv_tokens:],
                "output_seq_committed": len(self.outputs), "snapshot_id": self.snapshot_id,
                "durability": self.durability.value, "running": bool(self.task and not self.task.done()),
                "error": self.error}


class TurnHandle:
    def __init__(self, session: Session):
        self.session_id = session.session_id
        self.generation = session.generation
        self._queue: asyncio.Queue[OutputToken | Exception | None] = asyncio.Queue()

    def __aiter__(self) -> AsyncIterator[OutputToken]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[OutputToken]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item


class SessionGateway:
    def __init__(self, engine: EngineAdapter, store: Any):
        self.engine = engine
        self.store = store
        self.sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session {session_id}") from exc

    def _check(self, session: Session, expected_generation: int, *states: SessionState) -> None:
        integer(expected_generation, "expected_generation")
        if session.generation != expected_generation:
            raise GenerationConflict(f"expected generation {expected_generation}, current {session.generation}")
        if session.state not in states:
            raise StateConflict(f"operation is not allowed in {session.state.value}")

    async def _advance(self, session: Session) -> None:
        # Store calls run on a worker thread: a commit fsyncs and can wait for
        # another process's writer lock, while this event loop also delivers
        # every other session's tokens.
        new = session.generation + 1
        if hasattr(self.store, "claim_generation"):
            try:
                # The engine fixes a turn's durability when it starts, so a
                # host-memory turn's claim needs no fsync (see the store).
                accepted = await asyncio.to_thread(self.store.claim_generation, session.session_id,
                                                   session.generation, new,
                                                   durable=session.durability != Durability.VOLATILE)
            except StoreGenerationConflict as exc:
                raise GenerationConflict(str(exc)) from exc
            if accepted is False:
                raise GenerationConflict("another gateway claimed this generation")
        session.generation = new

    async def create(self, identity: KVIdentity, *, sampling: SamplingConfig | None = None,
                     seed: int | None = None, max_tokens: int = 256,
                     durability: Durability | str = Durability.VOLATILE) -> Session:
        integer(max_tokens, "max_tokens")
        seed = secrets.randbits(63) if seed is None else integer(seed, "seed")
        if seed >= 2**63:
            raise ContractError("seed must fit a signed 64-bit integer")
        session = Session(uuid.uuid4().hex, identity, sampling or SamplingConfig(), seed,
                          max_tokens, Durability(durability))
        if hasattr(self.store, "claim_generation"):
            await asyncio.to_thread(self.store.claim_generation, session.session_id, 0, 0)
        self.sessions[session.session_id] = session
        return session

    def _request(self, session: Session, count: int, external: int = 0) -> EngineRequest:
        return EngineRequest(request_id=uuid.uuid4().hex, session_id=session.session_id,
                             snapshot_id=uuid.uuid4().hex, generation=session.generation,
                             identity=session.identity, token_ids=tuple(session.token_ids),
                             seed=session.seed, sampling=session.sampling, max_tokens=count,
                             external_num_tokens=external, durability=session.durability,
                             output_seq_start=len(session.outputs),
                             remaining_max_tokens=session.remaining_max_tokens,
                             outputs=tuple(session.outputs), parent_id=session.snapshot_id)

    def _count(self, session: Session, max_tokens: int | None) -> int:
        if max_tokens is not None:
            integer(max_tokens, "max_tokens")
        context_left = session.identity.position_encoding["max_model_len"] - len(session.token_ids)
        return min(session.remaining_max_tokens, max_tokens if max_tokens is not None else session.remaining_max_tokens,
                   max(0, context_left))

    async def start_turn(self, session_id: str, token_ids: list[int] | tuple[int, ...], *,
                         expected_generation: int, max_tokens: int | None = None) -> TurnHandle:
        append = tokens(token_ids)
        session = self.get(session_id)
        async with session.lock:
            self._check(session, expected_generation, SessionState.ACTIVE)
            if session.task and not session.task.done():
                raise StateConflict("a turn is already running")
            if len(session.token_ids) + len(append) > session.identity.position_encoding["max_model_len"]:
                raise ContractError("turn exceeds model context")
            if not session.token_ids and not append:
                raise ContractError("first turn requires at least one token ID")
            if max_tokens is not None:
                integer(max_tokens, "max_tokens")
            available = min(session.remaining_max_tokens,
                            max_tokens if max_tokens is not None else session.remaining_max_tokens,
                            session.identity.position_encoding["max_model_len"] - len(session.token_ids) - len(append))
            if available <= 0 or (session.stop_reached and not append):
                raise ContractError("turn has no generation budget/context; use resume to replay a snapshot")
            await self._advance(session)
            session.token_ids.extend(append)
            if append:
                session.stop_reached = False
            handle = TurnHandle(session)
            count = 0 if session.stop_reached else self._count(session, max_tokens)
            if count:
                session.request = self._request(session, count)
                session.task = asyncio.create_task(self._consume(session, session.request, handle))
            else:
                handle._queue.put_nowait(None)
            return handle

    async def _consume(self, session: Session, request: EngineRequest, handle: TurnHandle) -> None:
        produced = 0
        try:
            async for event in self.engine.generate(request):
                async with session.lock:
                    # Delayed callbacks from an abandoned request never mutate
                    # the token log or consume another generation's budget.
                    if (event.generation != session.generation or session.request != request
                            or session.state not in (SessionState.ACTIVE, SessionState.PAUSING)):
                        continue
                    integer(event.token_id, "engine token_id")
                    integer(event.num_kv_tokens, "engine num_kv_tokens")
                    if session.stop_reached:
                        raise ContractError("engine emitted another token after its stop boundary")
                    if event.num_kv_tokens != len(session.token_ids):
                        raise ContractError("engine did not report a settled one-token boundary")
                    produced += 1
                    if produced > request.max_tokens:
                        raise ContractError("engine exceeded this request's token budget")
                    if session.remaining_max_tokens <= 0 or len(session.token_ids) >= session.identity.position_encoding["max_model_len"]:
                        raise ContractError("engine exceeded the declared token budget/context")
                    session.num_kv_tokens = event.num_kv_tokens
                    session.token_ids.append(event.token_id)
                    session.remaining_max_tokens -= 1
                    output = OutputToken(len(session.outputs) + 1, event.token_id, event.logprob)
                    session.outputs.append(output)
                    handle._queue.put_nowait(output)
                    if event.token_id in session.sampling.stop_token_ids or event.finish_reason == "stop":
                        session.stop_reached = True
        except Exception as exc:
            async with session.lock:
                if request.generation == session.generation and session.state != SessionState.CLOSED:
                    session.state = SessionState.FAILED
                    session.error = str(exc)
            handle._queue.put_nowait(exc)
        finally:
            handle._queue.put_nowait(None)

    async def pause(self, session_id: str, *, expected_generation: int,
                    durability: Durability | str | None = None) -> SnapshotManifest:
        session = self.get(session_id)
        async with session.lock:
            self._check(session, expected_generation, SessionState.ACTIVE)
            if durability is not None:
                requested = Durability(durability)
                if (requested != session.durability and session.request is not None
                        and getattr(self.engine, "snapshot_committed_at_request_end", False)):
                    raise ContractError("this engine commits durability at request creation; pause cannot change it")
                session.durability = requested
            session.state = SessionState.PAUSING
            request, task = session.request, session.task
        try:
            if task and not task.done():
                await self.engine.abort(request.request_id)
                await task
            async with session.lock:
                if session.state != SessionState.PAUSING:
                    raise StateConflict("engine failed while pausing")
            saved = (await self.engine.snapshot(request.request_id, session.num_kv_tokens)
                     if request else EngineSnapshot((), 0))
            if saved.num_kv_tokens != session.num_kv_tokens:
                raise ContractError("engine snapshot KV boundary differs from committed output boundary")
            manifest = SnapshotManifest(
                snapshot_id=request.snapshot_id if request else uuid.uuid4().hex,
                session_id=session.session_id, parent_id=session.snapshot_id,
                generation=session.generation, identity=session.identity,
                token_ids=tuple(session.token_ids), num_kv_tokens=session.num_kv_tokens,
                pending_tokens=tuple(session.token_ids[session.num_kv_tokens:]), seed=session.seed,
                sampling=session.sampling, remaining_max_tokens=session.remaining_max_tokens,
                output_seq_committed=len(session.outputs), outputs=tuple(session.outputs),
                durability=session.durability, stopped=session.stop_reached)
            if saved.committed_manifest is not None:
                committed = SnapshotManifest.from_dict(saved.committed_manifest)
                left, right = manifest.to_dict(), committed.to_dict()
                for key in ("segments", "state"):
                    left.pop(key); right.pop(key)
                if left != right:
                    raise ContractError("connector committed metadata differs from session boundary")
            else:
                committed = SnapshotManifest.from_dict(
                    await asyncio.to_thread(self.store.commit, manifest.to_dict(), saved.blocks))
            if request:
                await self.engine.release(request.request_id)
            async with session.lock:
                session.snapshot_id = committed.snapshot_id
                session.state = SessionState.PAUSED
                session.request = None
            return committed
        except Exception as exc:
            async with session.lock:
                session.state = SessionState.FAILED
                session.error = str(exc)
            raise

    def recover(self, snapshot_id: str) -> Session:
        manifest = SnapshotManifest.from_dict(self.store.get_manifest(snapshot_id))
        if hasattr(self.store, "is_session_closed") and self.store.is_session_closed(manifest.session_id):
            raise StateConflict("session is permanently closed; fork the snapshot into a new session instead")
        if manifest.state.value not in ("HostValid", "Durable"):
            raise ContractError("snapshot is not committed")
        if manifest.session_id in self.sessions:
            raise StateConflict("session is already registered in this gateway")
        generation = manifest.generation
        if hasattr(self.store, "get_generation"):
            generation = self.store.get_generation(manifest.session_id) or generation
        session = Session(manifest.session_id, manifest.identity, manifest.sampling,
                          manifest.seed, manifest.remaining_max_tokens, manifest.durability,
                          generation=generation, state=SessionState.PAUSED,
                          token_ids=list(manifest.token_ids), num_kv_tokens=manifest.num_kv_tokens,
                          outputs=list(manifest.outputs), snapshot_id=manifest.snapshot_id,
                          stop_reached=manifest.stopped or bool(manifest.outputs and manifest.outputs[-1].token_id in manifest.sampling.stop_token_ids))
        self.sessions[session.session_id] = session
        return session

    async def resume(self, session_id: str, *, expected_generation: int, ack_seq: int,
                     append_tokens: list[int] | tuple[int, ...] = (),
                     max_tokens: int | None = None, identity: KVIdentity | None = None) -> TurnHandle:
        append = tokens(append_tokens, "append_tokens")
        integer(ack_seq, "ack_seq")
        if max_tokens is not None:
            integer(max_tokens, "max_tokens")
        session = self.get(session_id)
        async with session.lock:
            self._check(session, expected_generation, SessionState.PAUSED)
            if ack_seq > len(session.outputs):
                raise ContractError("ack_seq exceeds the committed output sequence")
            # An evicted host-memory snapshot still describes the session; the
            # engine then misses and recomputes the prompt.
            manifest = SnapshotManifest.from_dict(
                await asyncio.to_thread(self.store.get_manifest, session.snapshot_id, include_invalid=True))
            target_identity = identity or session.identity
            prompt = tuple(manifest.token_ids) + append
            manifest.identity.require_match(target_identity)
            if len(prompt) > target_identity.position_encoding["max_model_len"]:
                raise ContractError("resume prompt exceeds model context")
            # A stopped/exhausted session still supports reconnect and replay;
            # it does not need a logits-producing token or an engine request.
            count = min(session.remaining_max_tokens, max_tokens if max_tokens is not None else session.remaining_max_tokens,
                        max(0, target_identity.position_encoding["max_model_len"] - len(prompt)))
            if session.stop_reached and not append:
                count = 0
            if count == 0 and append:
                raise ContractError("cannot append tokens to a replay-only resume; generation budget/context is required")
            if not count:
                external = manifest.num_kv_tokens
            elif manifest.state in (SnapshotState.HOST_VALID, SnapshotState.DURABLE):
                external = manifest.validate_resume(target_identity, prompt)
            else:
                external = 0
            await self._advance(session)
            session.state = SessionState.RESUMING
            handle = TurnHandle(session)
            request = EngineRequest(uuid.uuid4().hex, session.session_id, uuid.uuid4().hex,
                                    session.generation, target_identity, prompt, session.seed,
                                    session.sampling, count, external, session.durability,
                                    len(session.outputs), session.remaining_max_tokens,
                                    tuple(session.outputs), session.snapshot_id)
        try:
            blocks = None
            if not count or getattr(self.engine, "restore_needs_payload", True):
                # Replay-only resume (and engines that take the bytes from us)
                # validate every blob here. The vLLM worker reads and CRC-checks
                # the blobs itself while loading them, so it skips this second
                # read. The store blocks on disk and its own lock: keep it off
                # the event loop.
                blocks = await asyncio.to_thread(
                    self.store.load, manifest.snapshot_id, identity=target_identity.to_dict(),
                    token_prefix=list(manifest.token_ids))
            if count:
                await self.engine.restore(request, blocks)
            async with session.lock:
                session.token_ids = list(prompt)
                session.request = request if count else None
                session.state = SessionState.ACTIVE if count else SessionState.PAUSED
                if append:
                    session.stop_reached = False
                for output in session.outputs[ack_seq:]:
                    handle._queue.put_nowait(output)
                if count:
                    session.task = asyncio.create_task(self._consume(session, request, handle))
                else:
                    handle._queue.put_nowait(None)
            return handle
        except Exception as exc:
            async with session.lock:
                session.state = SessionState.FAILED
                session.error = str(exc)
            raise

    async def fork(self, snapshot_id: str, *, expected_generation: int) -> Session:
        integer(expected_generation, "expected_generation")
        manifest = SnapshotManifest.from_dict(await asyncio.to_thread(self.store.get_manifest, snapshot_id))
        def fork_manifest():
            try:
                return self.store.fork(snapshot_id, uuid.uuid4().hex, uuid.uuid4().hex, 0,
                                       expected_parent_generation=expected_generation)
            except StoreGenerationConflict as exc:
                raise GenerationConflict(str(exc)) from exc
        if manifest.session_id in self.sessions:
            parent = self.sessions[manifest.session_id]
            async with parent.lock:
                self._check(parent, expected_generation, SessionState.PAUSED, SessionState.ACTIVE, SessionState.CLOSED)
                child = await asyncio.to_thread(fork_manifest)
        else:
            child = await asyncio.to_thread(fork_manifest)
        return self.recover(child["snapshot_id"])

    async def close(self, session_id: str, *, expected_generation: int) -> None:
        session = self.get(session_id)
        async with session.lock:
            self._check(session, expected_generation, SessionState.ACTIVE, SessionState.PAUSED, SessionState.FAILED)
            if hasattr(self.store, "close_session"):
                try:
                    session.generation = await asyncio.to_thread(self.store.close_session, session.session_id,
                                                                 session.generation)
                except StoreGenerationConflict as exc:
                    raise GenerationConflict(str(exc)) from exc
            else:
                await self._advance(session)
            session.state = SessionState.CLOSED
            task, request = session.task, session.request
        if task and not task.done():
            await self.engine.abort(request.request_id)
            await task
        if request:
            await self.engine.release(request.request_id)

