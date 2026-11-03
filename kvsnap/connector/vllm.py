"""KV connector that saves a finished turn's KV and restores it on the next turn.

Out-of-tree ``KVConnectorBase_V1`` for exactly vLLM 0.31.0, TP=1, one
full-attention KV group, synchronous scheduling. The scheduler side decides
which blocks to save or load and owns the snapshot store; the worker side moves
the bytes through the pinned-memory data plane.

* Save: ``request_finished`` turns the settled turn into a manifest and asks vLLM
  to keep the blocks. ``wait_for_save`` hands the save to a background job that
  gathers the blocks on the GPU, copies them to pinned memory, and reports the
  GPU blocks free as soon as that copy is fenced. Hashing and writing follow
  on the same job; the scheduler then commits the manifest.
* Restore: ``get_num_new_matched_tokens`` reports the snapshot's tokens as an
  external hit to be loaded asynchronously, so vLLM parks the request and keeps
  running the rest of the batch. A loader thread reads and checks the blocks,
  copies them to the GPU and scatters them into the request's blocks; vLLM
  schedules the request once ``get_finished`` reports it.

Failure policy. Anything that goes wrong before a DMA is issued is a miss:
a mismatched, evicted or corrupt snapshot is recomputed by vLLM, and a save
that cannot be prepared is recorded so the gateway fails that pause promptly.
Only a copy whose completion cannot be proven stops the engine, because the
pinned or GPU memory involved could still be written by the DMA engine.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
import ctypes
import os
from dataclasses import dataclass, field
import json
import threading
import time
import uuid
from pathlib import Path

import vllm
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole, KVConnectorWorkerMetadata,
)

from kvsnap.connector.dataplane import PinnedCapacityError, TorchBlockDataPlane
from kvsnap.connector.volatile import VolatileOwner, VolatileClient
from kvsnap.connector.telemetry import EventRecorder
from kvsnap.connector.protocol import finish_manifest, matched_tokens
from kvsnap.connector.identity import runtime_identity
from kvsnap.models import ContractError, KVIdentity
from kvsnap.store import SnapshotStore, GenerationConflict, CorruptSnapshotError
from kvsnap.store.crc32c import crc32c
from kvsnap.store.store import BlobWriter, StoreError


class DMAUnknown(RuntimeError):
    """A copy was issued and its completion cannot be proven; the engine must stop."""


@dataclass
class Transfer:
    request_id: str
    block_ids: list[int]
    manifest: dict
    # Leading blocks already held by the local prefix cache; never overwritten.
    skip_blocks: int = 0
    lease_id: str | None = None
    volatile_locations: dict | None = None
    generation: int | None = None
    operation_id: str | None = None
    # Set when the scheduler could not lease the snapshot after matching it.
    error: str | None = None


@dataclass
class KvsnapMetadata(KVConnectorMetadata):
    saves: list[Transfer] = field(default_factory=list)
    loads: list[Transfer] = field(default_factory=list)


@dataclass
class KvsnapWorkerMetadata(KVConnectorWorkerMetadata):
    prepared: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    loaded_leases: list[str] = field(default_factory=list)
    invalid_snapshots: list[str] = field(default_factory=list)
    failed_loads: list[str] = field(default_factory=list)

    def aggregate(self, other):
        if set(self.prepared) & set(other.prepared):
            raise RuntimeError("kvsnap supports TP=1 only")
        return KvsnapWorkerMetadata(self.prepared | other.prepared,
                                 self.errors | other.errors,
                                 self.loaded_leases + other.loaded_leases,
                                 self.invalid_snapshots + other.invalid_snapshots,
                                 self.failed_loads + other.failed_loads)


def _completed(fn, *args):
    """Run ``fn`` now and return its outcome as a done Future."""
    future = Future()
    try:
        future.set_result(fn(*args))
    except BaseException as exc:
        future.set_exception(exc)
    return future


class KvsnapConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        if vllm.__version__ != "0.31.0":
            raise RuntimeError("kvsnap requires vLLM 0.31.0")
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("kvsnap requires TP=1")
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("kvsnap requires one full-attention KV group")
        if getattr(vllm_config, "speculative_config", None):
            raise ValueError("speculative decoding is unsupported")
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError("kvsnap requires synchronous scheduling: request_finished must see settled tokens")
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config
        self.root = Path(extra["store_path"])
        self.runtime_id = extra.get("runtime_id")
        self.block_size = vllm_config.cache_config.block_size
        self.capacity = int(extra.get("pinned_pool_bytes", 24 * 1024**3))
        # Host-memory snapshots may fill this share of the pinned pool; the
        # rest is staging for disk restores and durable saves.
        self.host_tier_bytes = int(extra.get("host_tier_bytes", self.capacity * 3 // 4))
        # Synchronous loads run inside start_load_kv and stall the whole batch;
        # kept only to measure that stall.
        self.async_load = bool(extra.get("async_load", True))
        # CRC-check host-memory blocks on restore too; disk reads always are.
        self.verify_host_checksums = bool(extra.get("verify_host_checksums", False))
        disk_capacity = int(extra.get("disk_capacity_bytes", 200 * 1024**3))
        self.telemetry = EventRecorder(self.runtime_id, role=getattr(role, "name", str(role)))
        # Diagnostic only: the probe wakes every millisecond and competes for
        # the GIL itself, so it is never on by default.
        if os.environ.get("KVSNAP_GIL_PROBE"):
            self.telemetry.start_gil_probe()
        self.store = SnapshotStore(self.root, host_capacity_bytes=self.capacity,
                                   disk_capacity_bytes=disk_capacity,
                                   recover_on_open=False) if role == KVConnectorRole.SCHEDULER else None
        self.writer = BlobWriter(self.root, disk_capacity_bytes=disk_capacity) if role == KVConnectorRole.WORKER else None
        self._init_state()

    def _init_state(self):
        # Scheduler side.
        self.saves, self.loads = [], []
        self._last_step = None
        self.saving = {}       # request id -> Transfer awaiting its prepared result
        self.matched = {}      # request id -> manifest matched but not yet allocated
        self.parked = set()    # requests whose async load vLLM has not scheduled yet (telemetry)
        # A request whose load failed is recomputed; vLLM asks again when it
        # reschedules it from zero computed tokens, and must then miss.
        self.failed_loads = set()
        # Commits run in order on their own thread; a request stays pending
        # push work until its commit has finished.
        self.committer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvsnap-commit")
        self.committing = set()
        # Worker side.
        self.dataplane = self.volatile = None
        self.save_jobs = {}    # request id -> (Future of prepared record, D2H-fenced Event)
        self.load_jobs = {}    # request id -> (Future of copy count, Transfer)
        self.finished, self.known_finished, self.load_errors = set(), set(), set()
        self.worker_meta = KvsnapWorkerMetadata()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kvsnap-save")
        self.loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvsnap-load")

    def _telemetry_enabled(self):
        return self.telemetry.enabled and self.telemetry.error is None

    def _event(self, event, *, transfer=None, request_id=None, operation_id=None, **fields):
        if not self._telemetry_enabled():
            return
        if transfer is not None:
            manifest = transfer.manifest
            fields = dict(snapshot_id=manifest["snapshot_id"], num_kv_tokens=manifest["num_kv_tokens"],
                          durability=manifest["durability"], **fields)
            request_id = transfer.request_id
            operation_id = operation_id or transfer.operation_id
        self.telemetry.emit(event, request_id=request_id, operation_id=operation_id, **fields)

    @staticmethod
    def _instruction(request):
        return (request.kv_transfer_params or {}).get("kvsnap")

    def register_kv_caches(self, kv_caches):
        self.dataplane = TorchBlockDataPlane(kv_caches, self.block_size, self.capacity)
        self.dataplane.warm_up()       # compile the restore kernel now, not on the first restore
        self.volatile = VolatileOwner(self.root, self.capacity)
        if self.runtime_id:
            from kvsnap.connector.runtime import publish_runtime_identity
            layout = json.dumps(self.dataplane.layers, sort_keys=True, separators=(",", ":"))
            publish_runtime_identity(self.root, self.runtime_id,
                                     runtime_identity(self._vllm_config, layout))
        self._event("cache_registered", layout=self.dataplane.layers, block_bytes=self.dataplane.block_bytes,
                    pinned_capacity_bytes=self.capacity, async_load=self.async_load)

    # ---- scheduler side --------------------------------------------------

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        op = self._instruction(request)
        if not op or not op.get("resume_snapshot_id") or request.request_id in self.failed_loads:
            return 0, False
        try:
            self._check_engine_identity(op["identity"])
            snapshot = self.store.get_manifest(op["resume_snapshot_id"])
            count = matched_tokens(snapshot, op["identity"], list(request.prompt_token_ids), num_computed_tokens)
            if count and num_computed_tokens % self.block_size:
                raise ContractError("the local prefix-cache hit ends inside a block")
        except (ContractError, KeyError, StoreError) as exc:
            # A mismatched, evicted or invalidated snapshot is a miss: vLLM
            # computes the whole prompt, which is always correct.
            self._event("restore_miss", request_id=request.request_id,
                        snapshot_id=op["resume_snapshot_id"], reason=f"{type(exc).__name__}: {exc}")
            return 0, False
        self.matched[request.request_id] = snapshot
        self._event("external_tokens_matched", request_id=request.request_id,
                    snapshot_id=snapshot["snapshot_id"], matched_tokens=count,
                    num_computed_tokens=num_computed_tokens, num_kv_tokens=snapshot["num_kv_tokens"])
        return count, bool(count) and self.async_load

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        manifest = self.matched.pop(request.request_id, None)
        if not num_external_tokens:
            return
        if manifest is None:
            raise ContractError("vLLM allocated external tokens for a request that was never matched")
        op = self._instruction(request)
        all_ids = blocks.get_block_ids()[0]
        count = (manifest["num_kv_tokens"] + self.block_size - 1) // self.block_size
        local = manifest["num_kv_tokens"] - num_external_tokens
        if local < 0 or local % self.block_size:
            raise ContractError("unrepresentable local prefix")
        selected = list(all_ids[:count])
        if len(selected) != count or len(set(selected)) != count or any(b <= 0 for b in selected):
            raise ContractError("target allocation is incomplete, aliased or uses the null block")
        transfer = Transfer(request.request_id, selected, manifest, skip_blocks=local // self.block_size,
                            generation=op["generation"], operation_id=uuid.uuid4().hex)
        try:
            # The lease keeps GC and host-tier eviction away while the worker reads.
            transfer.lease_id = self.store.acquire_lease(manifest["snapshot_id"])
            if manifest["durability"] == "volatile":
                transfer.volatile_locations = self.store.volatile_locations(manifest)
        except (KeyError, StoreError) as exc:
            # Gone since the match: the worker reports these blocks as failed
            # loads and vLLM recomputes them.
            if transfer.lease_id:
                self.store.release_lease(transfer.lease_id)
                transfer.lease_id = None
            transfer.error = f"{type(exc).__name__}: {exc}"
        first = transfer.skip_blocks
        self._event("load_allocated", transfer=transfer, session_id=manifest["session_id"],
                    generation=op["generation"], source_generation=manifest["generation"],
                    external_num_tokens=num_external_tokens, local_num_tokens=local,
                    block_ids=selected[first:], lease_id=transfer.lease_id, error=transfer.error,
                    bytes=sum(s["length"] for s in manifest["segments"][first:count]))
        self.loads.append(transfer)
        if self.async_load:
            self.parked.add(request.request_id)

    def request_finished(self, request, block_ids):
        started = time.monotonic_ns()
        self.matched.pop(request.request_id, None)
        self.failed_loads.discard(request.request_id)
        self.parked.discard(request.request_id)
        op = self._instruction(request)
        if not op or op.get("op") != "save":
            return False, None
        try:
            self._check_engine_identity(op["identity"])
        except ContractError as exc:
            self._save_failed(op["snapshot_id"], request.request_id, f"session identity differs from the engine: {exc}")
            return False, None
        if request.num_output_placeholders:
            raise ContractError("unsettled async output placeholders")
        manifest = finish_manifest(op, request.all_token_ids, request.num_computed_tokens, request.output_token_ids)
        manifest["stopped"] = request.status.name == "FINISHED_STOPPED"
        count = (manifest["num_kv_tokens"] + self.block_size - 1) // self.block_size
        if len(block_ids) < count:
            raise ContractError("scheduler provided too few computed blocks")
        evicting = time.monotonic_ns()
        if manifest["durability"] == "volatile":
            block_bytes = sum(layer["bytes"] for layer in json.loads(manifest["identity"]["kv_layout"]))
            for snapshot_id in self.store.make_room_volatile(count * block_bytes, self.host_tier_bytes):
                self._event("host_tier_evicted", snapshot_id=snapshot_id, request_id=request.request_id)
        transfer = Transfer(request.request_id, list(block_ids[:count]), manifest)
        now = time.monotonic_ns()
        self._event("request_finished_boundary", transfer=transfer, duration_ns=now - started,
                    evict_ns=now - evicting,
                    session_id=manifest["session_id"], generation=manifest["generation"],
                    identity=manifest["identity"], total_tokens=len(manifest["token_ids"]),
                    pending_token_count=len(manifest["pending_tokens"]), block_ids=transfer.block_ids)
        self.saves.append(transfer)
        self.saving[request.request_id] = transfer
        return True, {"kvsnap_snapshot_id": manifest["snapshot_id"]}

    def build_connector_meta(self, scheduler_output):
        # Called once per engine step: a long gap between calls is a stalled
        # step, as opposed to a client that was slow to read its outputs.
        now = time.monotonic_ns()
        if self._last_step is not None and now - self._last_step > 20_000_000:
            self._event("step_gap", gap_ns=now - self._last_step, previous_ns=self._last_step,
                        scheduled_tokens=scheduler_output.total_num_scheduled_tokens)
        self._last_step = now
        if self.parked:
            scheduled = scheduler_output.num_scheduled_tokens
            for request_id in self.parked & scheduled.keys():
                # The load was reported and vLLM scheduled the request's suffix.
                self.parked.discard(request_id)
                self._event("load_scheduled", request_id=request_id, scheduled_tokens=scheduled[request_id])
        result = KvsnapMetadata(self.saves, self.loads)
        self.saves, self.loads = [], []
        return result

    def _check_engine_identity(self, claimed):
        # The scheduler holds no KV tensors, so it checks every identity field
        # but the layout; the worker checks the layout against the caches it
        # registered (_check_layout).
        actual = runtime_identity(self._vllm_config, claimed["kv_layout"])
        actual.require_match(KVIdentity.from_dict(claimed))

    def _save_failed(self, snapshot_id, request_id, reason):
        self.store.record_save_failure(snapshot_id, reason)
        self._event("save_failed", request_id=request_id, snapshot_id=snapshot_id, reason=reason)

    def update_connector_output(self, connector_output):
        for future in [f for f in self.committing if f.done()]:
            self.committing.discard(future)
            future.result()  # an unexpected commit failure stops the engine here
        meta = connector_output.kv_connector_worker_meta
        if meta is None:
            return
        for request_id, prepared in meta.prepared.items():
            transfer = self.saving.pop(request_id, None)
            if transfer is None:
                continue
            # Publishing takes milliseconds of SQLite and owner round trips;
            # on this thread it stalled the whole batch for that engine step.
            self.committing.add(self.committer.submit(self._commit, transfer, prepared))
        for request_id, error in meta.errors.items():
            transfer = self.saving.pop(request_id, None)
            if transfer is not None:
                self._save_failed(transfer.manifest["snapshot_id"], request_id, error)
        for snapshot_id in meta.invalid_snapshots:
            self.store.invalidate_snapshot(snapshot_id, "worker could not validate snapshot before H2D")
        self.failed_loads.update(meta.failed_loads)
        for lease in meta.loaded_leases:
            self.store.release_lease(lease)
            self._event("load_lease_released", lease_id=lease)

    def _commit(self, transfer, prepared):
        committing = time.monotonic_ns()
        try:
            self.store.commit_prepared(transfer.manifest, prepared,
                                       expected_generation=transfer.manifest["generation"])
        except GenerationConflict:
            return  # Late completion from a superseded turn; its intent expires.
        except (StoreError, ContractError, ValueError) as exc:
            self._save_failed(transfer.manifest["snapshot_id"], transfer.request_id, f"commit failed: {exc}")
            return
        self._event("manifest_committed", transfer=transfer, preparation_id=prepared["preparation_id"],
                    generation=transfer.manifest["generation"], duration_ns=time.monotonic_ns() - committing)

    def has_pending_push_work(self):
        return bool(self.saves or self.saving or self.committing)

    def has_pending_block_frees(self):
        return bool(self.saving)

    # ---- worker side: restore ---------------------------------------------

    def _check_layout(self, manifest):
        self._check_engine_identity(manifest["identity"])
        if self.dataplane is None:
            raise RuntimeError("KV caches were not registered")
        actual = json.dumps(self.dataplane.layers, sort_keys=True, separators=(",", ":"))
        if manifest["identity"]["kv_layout"] != actual:
            raise ContractError("actual registered KV layout differs from snapshot identity")

    def bind_connector_metadata(self, connector_metadata):
        super().bind_connector_metadata(connector_metadata)
        # vLLM binds the metadata before the forward but calls start_load_kv
        # after it when every load is asynchronous. An async load's blocks
        # belong to a request this forward does not run, so its copies need
        # only the work queued before the forward (whatever last used the
        # freed blocks), not the forward itself: record that point here.
        self._queued_before_forward = (self.dataplane.producer_event()
                                       if self.async_load and getattr(connector_metadata, "loads", None) else None)

    def start_load_kv(self, forward_context, **kwargs):
        loads = self._get_connector_metadata().loads
        if not loads:
            return
        # Newly allocated blocks may have been freed this step by requests
        # whose kernels are still queued; loads wait for that work first.
        ready = getattr(self, "_queued_before_forward", None) or self.dataplane.producer_event()
        self._queued_before_forward = None
        for transfer in loads:
            if self.async_load:
                self.load_jobs[transfer.request_id] = (self.loader.submit(self._load, transfer, ready), transfer)
            else:
                self._finish_load(transfer, _completed(self._load, transfer, ready))

    def _load(self, transfer, ready):
        """Read, check and copy one snapshot into its blocks; returns the H2D copy count."""
        if transfer.error:
            raise StoreError(transfer.error)
        self._check_layout(transfer.manifest)
        first, size = transfer.skip_blocks, self.dataplane.block_bytes
        targets = transfer.block_ids[first:]
        segments = transfer.manifest["segments"][first:len(transfer.block_ids)]
        with ExitStack() as stack:
            if transfer.manifest["durability"] == "volatile":
                sources = self._volatile_sources(transfer, segments, stack)
            else:
                # Each block lands in its slot of one pinned buffer and is
                # CRC-checked there before anything is copied to the GPU.
                staged = stack.enter_context(self.dataplane.staging(len(targets)))
                self.writer.read_prepared_into(segments, staged.pointer, size, first_block=first)
                sources = [(staged, i * size) for i in range(len(targets))]
            self._event("h2d_started", transfer=transfer, lease_id=transfer.lease_id, block_ids=targets,
                        skip_blocks=first, bytes=size * len(targets))
            ready.synchronize()
            ready_ns = time.monotonic_ns()
            try:
                copies = self.dataplane.load_scatter(targets, sources)
            except Exception as exc:
                raise DMAUnknown("kvsnap H2D completion could not be proven") from exc
        self._event("h2d_completed", transfer=transfer, lease_id=transfer.lease_id, block_ids=targets,
                    skip_blocks=first, bytes=size * len(targets), copies=copies, ready_ns=ready_ns)
        return copies

    def _volatile_sources(self, transfer, segments, stack):
        locations = transfer.volatile_locations
        if locations is None:
            raise ContractError("volatile transfer lacks worker handles")
        if all(location["owner"] == self.volatile.descriptor for location in locations.values()):
            pinned = stack.enter_context(self.volatile.borrow(segments, locations,
                                                              verify_checksums=self.verify_host_checksums))
            return [(allocation, 0) for allocation in pinned]
        # Another live worker owns the bytes: fetch them into local staging,
        # checking each block, before any copy reaches the GPU.
        size = self.dataplane.block_bytes
        staged = stack.enter_context(self.dataplane.staging(len(segments)))
        for index, segment in enumerate(segments):
            location = locations[segment["blob_id"]]
            data = VolatileClient(self.root, location["owner"]).read(segment["blob_id"], location)
            if segment["offset"] != 0 or len(data) != segment["length"] or crc32c(data) != segment["crc32c"]:
                raise CorruptSnapshotError("volatile segment validation failed")
            ctypes.memmove(staged.pointer + index * size, data, size)
        return [(staged, index * size) for index in range(len(segments))]

    def _finish_load(self, transfer, future):
        try:
            future.result()
        except DMAUnknown:
            raise
        except Exception as exc:
            # Nothing reached the GPU: vLLM recomputes these blocks.
            self.load_errors.update(transfer.block_ids[transfer.skip_blocks:])
            self.worker_meta.failed_loads.append(transfer.request_id)
            if isinstance(exc, CorruptSnapshotError):
                self.worker_meta.invalid_snapshots.append(transfer.manifest["snapshot_id"])
            self._event("load_failed", transfer=transfer, reason=f"{type(exc).__name__}: {exc}")
        if transfer.lease_id:
            self.worker_meta.loaded_leases.append(transfer.lease_id)

    def wait_for_layer_load(self, layer_name):
        # Loads complete before vLLM schedules the request (or, synchronously,
        # before start_load_kv returns), so no layer ever waits.
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        # Saves capture only the settled request_finished boundary.
        pass

    # ---- worker side: save ------------------------------------------------

    def wait_for_save(self):
        saves = self._get_connector_metadata().saves
        if not saves:
            return
        # This step's kernels wrote the last KV; the save job's copies wait for it.
        ready = self.dataplane.producer_event()
        for transfer in saves:
            fenced = threading.Event()
            self.save_jobs[transfer.request_id] = (self.executor.submit(self._save, transfer, ready, fenced), fenced)
            self._event("save_handed_off", transfer=transfer)

    def _reserve(self, size):
        try:
            return self.dataplane.pool.reserve(size)
        except PinnedCapacityError:
            # The owner frees evicted host snapshots on its next sweep; do it now.
            self.volatile.reconcile()
            return self.dataplane.pool.reserve(size)

    def _save(self, transfer, ready, fenced):
        """Copy one turn to pinned memory, then hash and write it. ``fenced`` frees the GPU blocks."""
        manifest, size = transfer.manifest, self.dataplane.block_bytes
        count = len(transfer.block_ids)
        owned = []
        try:
            self._check_layout(manifest)
            if manifest["durability"] == "volatile":
                for _ in range(count):
                    owned.append(self._reserve(size))
                targets = [(allocation, 0) for allocation in owned]
            else:
                owned.append(self._reserve(size * count))
                targets = [(owned[0], i * size) for i in range(count)]
            ready.synchronize()
        except BaseException:
            for allocation in owned:
                allocation.close()
            fenced.set()          # nothing was copied
            raise
        gathering = time.monotonic_ns()
        try:
            self.dataplane.save_gather(transfer.block_ids, targets, manifest["num_kv_tokens"])
        except Exception as exc:
            # Keep the pinned memory: the DMA engine may still write it.
            raise DMAUnknown("kvsnap D2H completion could not be proven") from exc
        fenced.set()
        preparing = time.monotonic_ns()
        self._event("d2h_completed", transfer=transfer, block_ids=transfer.block_ids, bytes=size * count,
                    duration_ns=preparing - gathering)
        if manifest["durability"] == "volatile":
            prepared = self.volatile.prepare_allocations(owned, session_id=manifest["session_id"],
                                                         generation=manifest["generation"])
        else:
            try:
                blocks = [owned[0].read_range(i * size, size) for i in range(count)]
            finally:
                owned[0].close()
            prepared = self.writer.prepare_blocks(blocks, manifest["durability"])
        self._event("preparation_completed", transfer=transfer, preparation_id=prepared["preparation_id"],
                    bytes=sum(segment["length"] for segment in prepared["segments"]),
                    duration_ns=time.monotonic_ns() - preparing)
        return prepared

    def get_finished(self, finished_req_ids):
        self.known_finished.update(r for r in finished_req_ids if r in self.save_jobs or r in self.finished)
        for request_id, (future, fenced) in list(self.save_jobs.items()):
            done = future.done()      # read first: a done job has always set fenced
            if fenced.is_set():
                self.finished.add(request_id)
            if not done:
                continue
            del self.save_jobs[request_id]
            try:
                self.worker_meta.prepared[request_id] = future.result()
            except DMAUnknown:
                raise
            except Exception as exc:
                self.worker_meta.errors[request_id] = f"{type(exc).__name__}: {exc}"
        sending = self.finished & self.known_finished
        self.finished -= sending
        self.known_finished -= sending
        for request_id in sending:
            self._event("save_free_authorized", request_id=request_id)
        receiving = set()
        for request_id, (future, transfer) in list(self.load_jobs.items()):
            if future.done():
                del self.load_jobs[request_id]
                self._finish_load(transfer, future)
                receiving.add(request_id)
                self._event("load_reported", transfer=transfer)
        return sending, receiving

    def build_connector_worker_meta(self):
        result, self.worker_meta = self.worker_meta, KvsnapWorkerMetadata()
        return result

    def get_block_ids_with_load_errors(self):
        result, self.load_errors = self.load_errors, set()
        return result

    def shutdown(self):
        self.loader.shutdown(wait=True)
        self.executor.shutdown(wait=True)
        self.committer.shutdown(wait=True)
        if self.volatile:
            self.volatile.close()
        if self.dataplane:
            self.dataplane.close()
        if self.store:
            self.store.close()
        self.telemetry.close()
