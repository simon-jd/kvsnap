"""Opt-in process-local connector event records, never GPU completion authority.

A logging failure never changes data-plane control flow, but it does end the
record: the recorder latches the error, reports it to stderr and stops writing.
A reader must treat missing events and KVSNAP_TELEMETRY_ERROR reports as an
incomplete record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import uuid


class EventRecorder:
    def __init__(self, runtime_id=None, *, role=None, directory=None):
        value = os.environ.get('KVSNAP_EVENT_DIR') if directory is None else directory
        self.enabled = bool(value)
        self.runtime_id, self.role = runtime_id, role
        self.error = None
        self.path = None
        self._closed = False
        self._directory = Path(value) if value else None
        self._pid = None
        self._fd = None
        self._lock = threading.Lock() if self.enabled else None

    def _fail(self, event, exc):
        # Do not include event fields or arbitrary exception text in diagnostics.
        # OSError text can disclose a path; errno and type are enough to act on.
        self.error = {'event': event if isinstance(event, str) else 'invalid_event', 'error_type': type(exc).__name__,
                      'errno': getattr(exc, 'errno', None), 'pid': os.getpid()}
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        message = 'KVSNAP_TELEMETRY_ERROR ' + json.dumps(self.error, separators=(',', ':')) + '\n'
        try:
            os.write(2, message.encode())
        except OSError:
            # The sticky error remains observable even if stderr also fails.
            pass

    def _open_for_process(self):
        pid = os.getpid()
        if pid != self._pid:
            # An inherited recorder never appends to its parent's event file.
            if self._fd is not None:
                os.close(self._fd)
            self._fd = None
            self._pid = pid
            self._directory.mkdir(parents=True, mode=0o700, exist_ok=True)
            self.path = self._directory / f'kvsnap-events-{pid}-{uuid.uuid4().hex}.jsonl'
            self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)

    def emit(self, event, *, request_id=None, operation_id=None, **fields):
        """Return whether a complete JSONL record was appended; never raise.

        Callers supply metadata only. Token IDs, prompts, outputs, auth tokens
        and pointer values are deliberately outside the connector event schema.
        Disabled recorders perform no clock, JSON, filesystem or lock work.
        """
        if not self.enabled or self._closed or self.error is not None:
            return False
        # Forking a live multithreaded worker is unsupported, but recreating the
        # lock here also prevents an inherited locked mutex from hiding failure.
        if self._pid is not None and os.getpid() != self._pid:
            self._lock = threading.Lock()
        with self._lock:
            if self.error is not None:
                return False
            try:
                self._open_for_process()
                if not isinstance(event, str) or not event:
                    raise ValueError('event name required')
                reserved = {'schema_version', 'evidence_tier', 'event', 'pid', 'runtime_id',
                            'role', 'request_id', 'operation_id', 'time_ns', 'timestamp_ns', 'monotonic_ns'}
                if reserved.intersection(fields):
                    raise ValueError('event fields overwrite schema keys')
                now = time.time_ns()
                record = dict(schema_version=1, evidence_tier='runtime', event=event,
                              pid=self._pid, runtime_id=self.runtime_id, role=self.role,
                              request_id=request_id, operation_id=operation_id,
                              time_ns=now, timestamp_ns=now, monotonic_ns=time.monotonic_ns(), **fields)
                data = (json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode()
                # A recorder has one file and one process. Thread serialization
                # keeps even large layout records from interleaving.
                view = memoryview(data)
                while view:
                    count = os.write(self._fd, view)
                    if count <= 0:
                        raise OSError('event write made no progress')
                    view = view[count:]
                return True
            except Exception as exc:
                self._fail(event, exc)
                return False

    def start_gil_probe(self, threshold_ns=10_000_000, period_s=0.001):
        """Report every time this process kept a 1 ms sleeper waiting past ``threshold_ns``.

        A sleeping thread must take the GIL back to wake, so a late wake-up
        bounds how long other threads held the GIL (or the process was
        descheduled). One probe per process; a disabled recorder starts none.
        """
        if not self.enabled or getattr(self, "_probe_pid", None) == os.getpid():
            return
        self._probe_pid = os.getpid()

        def probe():
            while not self._closed and self.error is None:
                start = time.monotonic_ns()
                time.sleep(period_s)
                late = time.monotonic_ns() - start - int(period_s * 1e9)
                if late > threshold_ns:
                    self.emit("gil_probe_late", late_ns=late, woke_monotonic_ns=start + late + int(period_s * 1e9))

        threading.Thread(target=probe, name="kvsnap-gil-probe", daemon=True).start()

    def close(self):
        if not self.enabled or self._closed:
            return
        if self._pid is not None and os.getpid() != self._pid:
            self._lock = threading.Lock()
        with self._lock:
            self._closed = True
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError as exc:
                    self._fail('recorder_close', exc)
                self._fd = None
