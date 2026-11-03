"""Where a resumed turn's time to first token goes, phase by phase.

    python -m bench.resume_phases --trace LABEL=DIR [--trace ...] --output results/resume-phases.json

Each DIR comes from ``bench/restore_under_load.py --trace-dir DIR`` (without
``--probes``, which slow the run they observe). For every measured resume of
the ``sync`` or ``async`` arm, the benchmark's submit and first-token times
are joined with the connector's events for that request (the adapter's
request id is a prefix of vLLM's) and cut into consecutive phases:

  gateway     resume() called -> request handed to vLLM (adapter_generate):
              manifest read and generation claim on the gateway
  admission   -> the scheduler matched the snapshot (external_tokens_matched):
              input processing, IPC to the engine core, the wait for the
              running step to end, and the scheduler's manifest lookup
  lease       -> blocks allocated and the snapshot leased (load_allocated)
  dispatch    -> the loader holds its sources (h2d_started)
  queued      -> the engine stream's queued work finished (h2d_completed's ready_ns)
  copy        -> every block copied and scattered (h2d_completed)
  report      -> get_finished returned the request (load_reported; async only)
  schedule    -> vLLM scheduled its suffix (load_scheduled; async only)
  prefill     -> its first output reached the adapter (adapter_first_output)
  delivery    -> the first token reached the caller of resume()

Where a mark is missing (a sync load has no report or schedule mark), the
phases on either side of it are reported as one, e.g. ``report+schedule+prefill``,
so each resume's phases always sum to its total. Connector events
carry ``monotonic_ns`` and the benchmark ``perf_counter_ns``; both read
CLOCK_MONOTONIC on Linux. Each engine also gets the copy's rate (bytes over
``queued`` to ``copy``) and how many loads were reported before their copy
completed, warm-up included, and the run's decode gaps from its result.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

MARKS = ("adapter_generate", "external_tokens_matched", "load_allocated", "h2d_started", "h2d_ready",
         "h2d_completed", "load_reported", "load_scheduled", "adapter_first_output")
FULL = ("submitted", *MARKS, "first_token")
PHASES = ("gateway", "admission", "lease", "dispatch", "queued", "copy", "report", "schedule", "prefill",
          "delivery")


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def load_events(directory: Path) -> list[dict]:
    events = []
    for path in sorted(directory.glob("*.jsonl")):
        events += [json.loads(line) for line in path.read_text().splitlines() if line]
    return events


def resume_phases(turns: list[dict], events: list[dict]) -> list[dict]:
    """Per resume: {phase: ms} for consecutive marks that are both present, plus total_ms."""
    handed = sorted((e["monotonic_ns"], e["request_id"]) for e in events if e["event"] == "adapter_generate")
    by_request = {}
    for event in events:
        if event.get("request_id"):
            by_request.setdefault(event["request_id"], []).append(event)
    rows = []
    for turn in turns:
        request = next((r for t, r in handed if turn["submitted"] <= t <= turn["first_token"]), None)
        times = {"submitted": turn["submitted"], "first_token": turn["first_token"]}
        for key, items in by_request.items():
            if request and key.startswith(request):
                for event in items:
                    times.setdefault(event["event"], event["monotonic_ns"])
                    if event["event"] == "h2d_completed" and "ready_ns" in event:
                        times.setdefault("h2d_ready", event["ready_ns"])
        present = [i for i, mark in enumerate(FULL) if mark in times]
        row = {"total_ms": (turn["first_token"] - turn["submitted"]) / 1e6}
        for start, end in zip(present, present[1:]):
            # Marks missing in between merge their phases: "report+schedule+prefill".
            row["+".join(PHASES[start:end])] = (times[FULL[end]] - times[FULL[start]]) / 1e6
        rows.append(row)
    return rows


def describe(values):
    return {"n": len(values), "median": statistics.median(values), "mean": statistics.mean(values),
            "p90": percentile(values, 0.9), "max": max(values)}


def summarize(directory: Path) -> dict:
    result = {}
    for engine in ("sync", "async"):
        arm = directory / f"{engine}-{engine}.json"
        if not arm.is_file():
            continue
        events = load_events(directory / f"events-{engine}")
        rows = resume_phases(json.loads(arm.read_text())["turns"], events)
        # The copy's rate: bytes over the time from the queued work's end to the copy's.
        rates = [e["bytes"] / (e["monotonic_ns"] - e["ready_ns"]) for e in events
                 if e["event"] == "h2d_completed" and "ready_ns" in e]
        result[engine] = {"resumes": len(rows), "total_ms": describe([r["total_ms"] for r in rows]),
                          "copy_gb_per_s": describe(rates) if rates else None,
                          "loads_reported_early": sum(1 for e in events
                                                      if e["event"] == "load_reported" and e.get("early")),
                          "loads_reported": sum(1 for e in events if e["event"] == "load_reported"),
                          "phases_ms": {p: describe([r[p] for r in rows if p in r])
                                        for p in dict.fromkeys(k for r in rows for k in r if k != "total_ms")}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for item in args.trace:
        label, directory = item.split("=", 1)
        result[label] = summarize(Path(directory))
        if (Path(directory) / "result.json").is_file():
            run = json.loads((Path(directory) / "result.json").read_text())
            result[label]["environment"] = run.get("environment")
            for engine in ("sync", "async"):
                if engine in result[label] and engine in run:
                    # The same run's decode gaps, so a phase change can be read against them.
                    result[label][engine]["decode_gap_ms"] = run[engine][engine]["decode_gap_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
