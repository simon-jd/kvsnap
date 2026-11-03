import asyncio
import dataclasses
import tempfile
import unittest

from kvsnap.gateway import DeterministicEngine, EngineToken, SessionGateway, fixture_identity
from kvsnap.gateway.service import GenerationConflict, StateConflict
from kvsnap.models import ContractError, IdentityMismatch, SamplingConfig, SessionState
from kvsnap.store import SnapshotStore


async def collect(handle):
    return [token async for token in handle]


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(self.tmp.name, host_capacity_bytes=1024**2, disk_capacity_bytes=1024**2)
        self.engine = DeterministicEngine()
        self.gateway = SessionGateway(self.engine, self.store)
        self.identity = fixture_identity(block_size=4)

    async def asyncTearDown(self):
        for session in self.gateway.sessions.values():
            if session.task and not session.task.done():
                session.task.cancel()
                try:
                    await session.task
                except asyncio.CancelledError:
                    pass
        self.store.close()
        self.tmp.cleanup()

    async def create_paused(self, *, budget=12, first=4, durability="durable", prompt=(1, 2, 3)):
        session = await self.gateway.create(self.identity, seed=42, max_tokens=budget, durability=durability)
        emitted = await collect(await self.gateway.start_turn(session.session_id, list(prompt), expected_generation=0, max_tokens=first))
        manifest = await self.gateway.pause(session.session_id, expected_generation=1)
        return session, manifest, emitted

    async def test_resume_equals_uninterrupted_at_block_boundaries(self):
        for length in (3, 4, 5):
            session, manifest, first = await self.create_paused(prompt=tuple(range(length)))
            rest = await collect(await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=4))
            reference = await self.gateway.create(self.identity, seed=42, max_tokens=12)
            oracle = await collect(await self.gateway.start_turn(reference.session_id, list(range(length)), expected_generation=0))
            self.assertEqual(first + rest, oracle)
            self.assertEqual(manifest.num_kv_tokens, length + 3)
            self.assertEqual(len(manifest.pending_tokens), 1)
            self.assertEqual(session.remaining_max_tokens, 0)

    async def test_an_engine_that_loads_the_kv_itself_resumes_without_another_store_read(self):
        class SelfLoading(DeterministicEngine):
            restore_needs_payload = False

            async def restore(self, request, blocks):
                self.requests[request.request_id] = request
                self.histories[request.request_id] = list(request.token_ids)
                self.restored = request

        self.engine = SelfLoading()
        self.gateway = SessionGateway(self.engine, self.store)
        session, manifest, first = await self.create_paused()
        reads = []
        original = self.store.get_manifest
        self.store.get_manifest = lambda *args, **kwargs: reads.append(args) or original(*args, **kwargs)
        rest = await collect(await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=4))
        self.assertEqual(reads, [])
        self.assertEqual(self.engine.restored.token_ids, manifest.token_ids)
        self.assertEqual(self.engine.restored.external_num_tokens, manifest.num_kv_tokens)
        reference = await self.gateway.create(self.identity, seed=42, max_tokens=12)
        oracle = await collect(await self.gateway.start_turn(reference.session_id, [1, 2, 3], expected_generation=0))
        self.assertEqual(first + rest, oracle)

    async def test_an_engine_given_the_kv_bytes_resumes_from_a_fresh_store_read(self):
        session, _, _ = await self.create_paused()
        reads = []
        original = self.store.get_manifest
        self.store.get_manifest = lambda *args, **kwargs: reads.append(args) or original(*args, **kwargs)
        await collect(await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=4))
        self.assertEqual(reads, [(session.snapshot_id,)])

    async def test_ack_replays_committed_outputs_before_new_tokens(self):
        session, _, first = await self.create_paused()
        replay_and_new = await collect(await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=2, max_tokens=2))
        self.assertEqual(replay_and_new[:2], first[2:])
        self.assertEqual([o.seq for o in replay_and_new], [3, 4, 5, 6])
        self.assertEqual(session.remaining_max_tokens, 6)

    async def test_bad_ack_does_not_claim_generation(self):
        session, _, _ = await self.create_paused()
        for ack in (-1, 5, True):
            with self.assertRaises(ContractError):
                await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=ack)
            self.assertEqual(session.generation, 1)

    async def test_concurrent_resume_has_exactly_one_winner(self):
        session, _, _ = await self.create_paused()
        results = await asyncio.gather(*[self.gateway.resume(session.session_id, expected_generation=1, ack_seq=4) for _ in range(2)], return_exceptions=True)
        self.assertEqual(sum(not isinstance(r, Exception) for r in results), 1)
        self.assertTrue(any(isinstance(r, (GenerationConflict, StateConflict)) for r in results))
        await collect(next(r for r in results if not isinstance(r, Exception)))

    async def test_fork_shares_prefix_and_isolates_appended_tokens(self):
        parent, manifest, first = await self.create_paused()
        child = await self.gateway.fork(manifest.snapshot_id, expected_generation=1)
        child_manifest = self.store.get_manifest(child.snapshot_id)
        self.assertEqual(child_manifest["segments"], manifest.to_dict()["segments"])
        self.assertEqual(child_manifest["parent_id"], manifest.snapshot_id)
        parent_tokens = await collect(await self.gateway.resume(parent.session_id, expected_generation=1, ack_seq=4, append_tokens=[100]))
        child_tokens = await collect(await self.gateway.resume(child.session_id, expected_generation=0, ack_seq=4, append_tokens=[200]))
        self.assertNotEqual(parent_tokens, child_tokens)
        self.assertEqual(parent.token_ids[len(manifest.token_ids)], 100)
        self.assertEqual(child.token_ids[len(manifest.token_ids)], 200)
        # The immutable source manifest and shared blob bytes do not change.
        self.assertEqual(self.store.get_manifest(manifest.snapshot_id), manifest.to_dict())

    async def test_resume_rejects_identity_change(self):
        session, _, _ = await self.create_paused()
        with self.assertRaises(IdentityMismatch):
            await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=0,
                                      identity=dataclasses.replace(self.identity, tokenizer_revision="different"))
        self.assertEqual(session.generation, 1)

    async def test_restart_restores_seed_budget_log_and_generation(self):
        session, manifest, first = await self.create_paused()
        # A failed old attempt may have claimed an epoch without a new snapshot.
        self.store.claim_generation(session.session_id, 1, 2)
        second_gateway = SessionGateway(DeterministicEngine(), self.store)
        restored = second_gateway.recover(manifest.snapshot_id)
        self.assertEqual(restored.generation, 2)
        self.assertEqual(restored.seed, 42)
        output = await collect(await second_gateway.resume(restored.session_id, expected_generation=2, ack_seq=0, max_tokens=1))
        self.assertEqual(output[:4], first)
        self.assertEqual(restored.remaining_max_tokens, 7)
        self.assertEqual(restored.generation, 3)

    async def test_zero_budget_resume_only_replays(self):
        session, _, first = await self.create_paused(budget=4)
        requests_before = len(self.engine.requests)
        replay = await collect(await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=0))
        self.assertEqual(replay, first)
        self.assertEqual(len(self.engine.requests), requests_before)
        self.assertEqual(session.state, SessionState.PAUSED)

    async def test_pending_stop_token_is_replayed_without_generation(self):
        class StopEngine(DeterministicEngine):
            async def generate(self, request):
                self.requests[request.request_id] = request
                self.histories[request.request_id] = list(request.token_ids) + [99]
                yield EngineToken(99, len(request.token_ids), request.generation)
        engine = StopEngine()
        gateway = SessionGateway(engine, self.store)
        session = await gateway.create(self.identity, seed=1, max_tokens=10, sampling=SamplingConfig(stop_token_ids=(99,)))
        first = await collect(await gateway.start_turn(session.session_id, [1, 2], expected_generation=0))
        manifest = await gateway.pause(session.session_id, expected_generation=1)
        self.assertEqual(manifest.pending_tokens, (99,))
        replay = await collect(await gateway.resume(session.session_id, expected_generation=1, ack_seq=0))
        self.assertEqual(replay, first)
        self.assertEqual(session.remaining_max_tokens, 9)

    async def test_mid_turn_pause_settles_before_snapshot(self):
        self.engine.delay = 0.001
        session = await self.gateway.create(self.identity, seed=42, max_tokens=100)
        handle = await self.gateway.start_turn(session.session_id, [1, 2, 3], expected_generation=0)
        iterator = handle.__aiter__()
        first = await anext(iterator)
        manifest = await self.gateway.pause(session.session_id, expected_generation=1)
        remainder = [token async for token in iterator]
        self.assertEqual(manifest.output_seq_committed, 1 + len(remainder))
        self.assertEqual(session.state, SessionState.PAUSED)
        self.assertEqual(manifest.pending_tokens, (manifest.outputs[-1].token_id,))
        self.assertEqual(manifest.outputs[0], first)

    async def test_eos_stop_reason_survives_restart_and_engine_finishes(self):
        class EOSEngine(DeterministicEngine):
            settled = False
            async def generate(self, request):
                self.requests[request.request_id] = request
                self.histories[request.request_id] = list(request.token_ids) + [99]
                yield EngineToken(99, len(request.token_ids), request.generation, finish_reason="stop")
                self.settled = True
        engine = EOSEngine()
        gateway = SessionGateway(engine, self.store)
        session = await gateway.create(self.identity, seed=1, max_tokens=10, durability="durable")
        first = await collect(await gateway.start_turn(session.session_id, [1, 2], expected_generation=0))
        self.assertTrue(engine.settled)
        manifest = await gateway.pause(session.session_id, expected_generation=1)
        self.assertTrue(manifest.stopped)
        second_engine = DeterministicEngine()
        second = SessionGateway(second_engine, self.store)
        recovered = second.recover(manifest.snapshot_id)
        replay = await collect(await second.resume(recovered.session_id, expected_generation=1, ack_seq=0))
        self.assertEqual(replay, first)
        self.assertFalse(second_engine.requests)

    async def test_precommitted_engine_rejects_durability_change_before_state_mutation(self):
        self.engine.snapshot_committed_at_request_end = True
        session = await self.gateway.create(self.identity, seed=42, max_tokens=3, durability="durable")
        await collect(await self.gateway.start_turn(session.session_id, [1, 2], expected_generation=0, max_tokens=1))
        before = session.to_dict()
        with self.assertRaises(ContractError):
            await self.gateway.pause(session.session_id, expected_generation=1, durability="process")
        self.assertEqual(session.to_dict(), before)

    async def test_stale_engine_generation_is_ignored(self):
        class StaleEngine(DeterministicEngine):
            async def generate(self, request):
                yield EngineToken(999, len(request.token_ids), request.generation - 1)
                async for token in super().generate(request):
                    yield token
        gateway = SessionGateway(StaleEngine(), self.store)
        session = await gateway.create(self.identity, seed=42, max_tokens=2)
        output = await collect(await gateway.start_turn(session.session_id, [1, 2], expected_generation=0))
        self.assertEqual(len(output), 2)
        self.assertNotIn(999, session.token_ids)
        self.assertEqual(session.remaining_max_tokens, 0)

    async def test_close_fences_and_does_not_delete_snapshots(self):
        session, manifest, _ = await self.create_paused()
        await self.gateway.close(session.session_id, expected_generation=1)
        self.assertEqual(session.state, SessionState.CLOSED)
        self.assertEqual(session.generation, 2)
        self.assertEqual(self.store.get_manifest(manifest.snapshot_id)["state"], "Durable")
        with self.assertRaises(GenerationConflict):
            await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=0)
        second = SessionGateway(DeterministicEngine(), self.store)
        with self.assertRaises(StateConflict):
            second.recover(manifest.snapshot_id)
        child = await second.fork(manifest.snapshot_id, expected_generation=2)
        self.assertNotEqual(child.session_id, session.session_id)
        self.assertEqual(child.state, SessionState.PAUSED)

    async def test_seed_autogeneration_and_context_limit(self):
        session = await self.gateway.create(self.identity, max_tokens=2)
        self.assertGreaterEqual(session.seed, 0)
        with self.assertRaises(ContractError):
            await self.gateway.start_turn(session.session_id, [1] * 32769, expected_generation=0)
        self.assertEqual(session.generation, 0)

    async def test_second_pause_resume_cycle_preserves_dag_and_budget(self):
        session, first_manifest, _ = await self.create_paused()
        await collect(await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=4, max_tokens=2))
        second = await self.gateway.pause(session.session_id, expected_generation=2)
        self.assertEqual(second.parent_id, first_manifest.snapshot_id)
        self.assertEqual(second.remaining_max_tokens, 6)
        output = await collect(await self.gateway.resume(session.session_id, expected_generation=2, ack_seq=6))
        self.assertEqual(len(output), 6)

    async def test_zero_token_turn_does_not_reuse_old_request(self):
        session = await self.gateway.create(self.identity, seed=42, max_tokens=10)
        await collect(await self.gateway.start_turn(session.session_id, [1, 2], expected_generation=0, max_tokens=2))
        original = session.to_dict()
        with self.assertRaises(ContractError):
            await self.gateway.start_turn(session.session_id, [999], expected_generation=1, max_tokens=0)
        self.assertEqual(session.to_dict(), original)
        snapshot = await self.gateway.pause(session.session_id, expected_generation=1)
        self.assertEqual(snapshot.token_ids, tuple(original["token_ids"]))

    async def test_replay_only_append_is_rejected_without_mutation(self):
        session, manifest, _ = await self.create_paused()
        before = session.to_dict()
        with self.assertRaises(ContractError):
            await self.gateway.resume(session.session_id, expected_generation=1, ack_seq=4, append_tokens=[999], max_tokens=0)
        self.assertEqual(session.to_dict(), before)

    async def test_fork_checks_cross_gateway_fence(self):
        session, manifest, _ = await self.create_paused()
        self.store.claim_generation(session.session_id, 1, 2)
        second = SessionGateway(DeterministicEngine(), self.store)
        with self.assertRaises(GenerationConflict):
            await second.fork(manifest.snapshot_id, expected_generation=1)
        child = await second.fork(manifest.snapshot_id, expected_generation=2)
        self.assertEqual(child.state, SessionState.PAUSED)


if __name__ == "__main__":
    unittest.main()
