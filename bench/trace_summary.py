"""Summarize traced runs of the under-load benchmark.

    python -m bench.trace_summary --output results/turn-end-traces.json \\
        --trace LABEL=DIR --code LABEL=TEXT [--trace ... --code ...]

Each DIR comes from ``bench/restore_under_load.py --trace-dir DIR``. For each
engine (``sync`` or ``async`` load) it holds ``<arm>-<engine>.json`` per arm
(the resumed turns' timestamps, every decode gap over 20 ms, the client's
event-loop stalls and garbage-collection pauses) and ``events-<engine>/``, the
connector's event files. Connector events carry ``monotonic_ns`` and the
benchmark records ``perf_counter_ns``; on Linux both read CLOCK_MONOTONIC, so
they compare directly across processes.

For each engine the summary reports:

* save phases, over every save in the trace: the D2H copy
  (``d2h_completed``), the preparation that hashes, deduplicates and records
  the intent (``preparation_completed``), the manifest commit
  (``manifest_committed``), and ``request_finished`` on the scheduler thread,
  host-tier eviction included (``request_finished_boundary``);
* per arm, the engine steps longer than 20 ms (``step_gap``) that overlap a
  turn end, from 50 ms before a resumed turn ends until 200 ms after its pause
  returns, and how many fell at each turn end;
* per arm, every event-loop stall over 5 ms in the benchmark client (the probe
  sleeps 1 ms; a stall is the time until it woke), and the stalls that span
  the moment a resumed turn was submitted;
* per arm, garbage-collection pauses over 5 ms in the client.

``--code LABEL=TEXT`` records which code a trace ran.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

SAVE_PHASES = {"d2h": "d2h_completed", "prepare": "preparation_completed",
               "commit": "manifest_committed", "request_finished": "request_finished_boundary"}
TURN_END_BEFORE_NS, TURN_END_AFTER_NS = 50_000_000, 200_000_000


def stats(values):
    if not values:
        return None
    return {"n": len(values), "min": round(min(values), 1), "median": round(statistics.median(values), 1),
            "max": round(max(values), 1)}


def load_events(directory: Path) -> list[dict]:
    events = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open() as stream:
            events.extend(json.loads(line) for line in stream if line.strip())
    return sorted(events, key=lambda event: event["monotonic_ns"])


def overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def summarize_arm(data: dict, steps: list[dict]) -> dict:
    turns = data["turns"]
    per_turn, turn_end_steps = [], []
    for turn in turns:
        start, stop = turn["turn_ended"] - TURN_END_BEFORE_NS, turn["paused"] + TURN_END_AFTER_NS
        hits = [step["gap_ns"] / 1e6 for step in steps
                if overlaps(step["previous_ns"], step["previous_ns"] + step["gap_ns"], start, stop)]
        per_turn.append(len(hits))
        turn_end_steps.extend(hits)
    stalls = [(lag["t0"], lag["t0"] + (lag["late_ms"] + 1) * 1e6) for lag in data.get("loop_lag", [])]
    spanning = [(b - a) / 1e6 for a, b in stalls if any(a <= turn["submitted"] <= b for turn in turns)]
    return {
        "resumed_turns": len(turns),
        "turn_end_steps_over_20_ms": sorted(round(value, 1) for value in turn_end_steps),
        "turn_end_steps_per_turn": per_turn,
        "loop_stalls_over_5_ms": sorted(round((b - a) / 1e6, 1) for a, b in stalls),
        "loop_stalls_spanning_a_submit": sorted(round(value, 1) for value in spanning),
        # Absent (None) when the client did not record collections.
        "gc_pauses_over_5_ms": (sorted(round(pause["ms"], 1) for pause in data["gc_pauses"])
                                if "gc_pauses" in data else None),
    }


def summarize(directory: Path) -> dict:
    engines = {}
    for events_dir in sorted(directory.glob("events-*")):
        engine = events_dir.name.removeprefix("events-")
        events = load_events(events_dir)
        phases = {name: stats([e["duration_ns"] / 1e6 for e in events if e["event"] == kind and "duration_ns" in e])
                  for name, kind in SAVE_PHASES.items()}
        evict = [e["evict_ns"] / 1e6 for e in events if e["event"] == "request_finished_boundary" and "evict_ns" in e]
        phases["host_tier_eviction_in_request_finished"] = stats(evict)
        steps = [e for e in events if e["event"] == "step_gap"]
        arms = {}
        for path in sorted(directory.glob(f"*-{engine}.json")):
            arms[path.stem.removesuffix(f"-{engine}")] = summarize_arm(json.loads(path.read_text()), steps)
        engines[engine] = {"save_phases_ms": phases, "arms": arms}
    return engines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--code", action="append", default=[], metavar="LABEL=TEXT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = dict(item.split("=", 1) for item in args.code)
    traces = []
    for item in args.trace:
        label, directory = item.split("=", 1)
        entry = {"label": label, "code": code.get(label), "engines": summarize(Path(directory))}
        run = Path(directory) / "result.json"            # the benchmark's own output, when kept there
        if run.exists() and "environment" in (measured := json.loads(run.read_text())):
            entry["environment"] = measured["environment"]
        traces.append(entry)
    result = {
        "benchmark": "bench/restore_under_load.py --engines async --trace-dir DIR, other options at their "
                     "defaults: Qwen3-1.7B, 16 greedy decode streams, six 8k-token host-memory sessions "
                     "resuming one every 2 s, --max-num-batched-tokens 2048",
        "summary": "bench/trace_summary.py; times in ms",
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
