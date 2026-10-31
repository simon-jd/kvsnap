"""Gateway session rules: fencing, replay, fork and close under concurrent calls."""
import tempfile
import unittest

from kvsnap.gateway.engine import DeterministicEngine, fixture_identity
from kvsnap.gateway.service import GenerationConflict, SessionGateway, StateConflict
from kvsnap.models import ContractError, SessionState
from kvsnap.store import SnapshotStore


async def collect(handle):
    return [output async for output in handle]


class GatewayReviewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(self.temp.name)
        self.engine = DeterministicEngine()
        self.gateway = SessionGateway(self.engine, self.store)
        self.session = await self.gateway.create(fixture_identity(), seed=1,
                                                 max_tokens=10, durability="durable")
        await collect(await self.gateway.start_turn(self.session.session_id, [1, 2],
                                                    expected_generation=0, max_tokens=2))
        self.manifest = await self.gateway.pause(self.session.session_id, expected_generation=1)

    async def asyncTearDown(self):
        self.store.close()
        self.temp.cleanup()

    async def test_unregistered_parent_fork_checks_persisted_source_generation(self):
        self.store.claim_generation(self.session.session_id, 1, 2)
        restarted = SessionGateway(DeterministicEngine(), self.store)
        with self.assertRaises(GenerationConflict):
            await restarted.fork(self.manifest.snapshot_id, expected_generation=1)

    async def test_closed_session_cannot_resume_from_old_snapshot_after_restart(self):
        await self.gateway.close(self.session.session_id, expected_generation=1)
        self.store.close()
        self.store = SnapshotStore(self.temp.name)
        restarted = SessionGateway(DeterministicEngine(), self.store)
        try:
            restored = restarted.recover(self.manifest.snapshot_id)
        except StateConflict:
            return
        self.assertEqual(restored.state, SessionState.CLOSED,
                         "recovery must retain the Closed terminal state")
        with self.assertRaises(StateConflict):
            await restarted.resume(restored.session_id,
                                   expected_generation=restored.generation, ack_seq=2)

    async def test_replay_only_resume_does_not_silently_lose_appended_tokens(self):
        try:
            handle = await self.gateway.resume(self.session.session_id,
                                              expected_generation=1, ack_seq=2,
                                              append_tokens=[999], max_tokens=0)
        except ContractError:
            self.assertEqual(self.session.generation, 1)
            self.assertEqual(tuple(self.session.token_ids), self.manifest.token_ids)
            return
        await collect(handle)
        await collect(await self.gateway.resume(self.session.session_id,
                                               expected_generation=2, ack_seq=2,
                                               max_tokens=1))
        self.assertEqual(self.session.token_ids[len(self.manifest.token_ids)], 999,
                         "accepted append must survive replay-only resume")


if __name__ == "__main__":
    unittest.main()
