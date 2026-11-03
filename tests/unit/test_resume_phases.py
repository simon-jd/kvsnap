"""The resume phase breakdown joins the benchmark's turns with connector events."""
import unittest

from bench.resume_phases import resume_phases


def event(name, request_id, t, **fields):
    return {"event": name, "request_id": request_id, "monotonic_ns": t * 1_000_000, **fields}


class ResumePhaseTests(unittest.TestCase):
    def test_an_async_resume_is_cut_into_consecutive_phases_that_sum_to_its_total(self):
        turns = [{"submitted": 100_000_000, "first_token": 200_000_000}]
        events = [event("adapter_generate", "abc", 104),
                  event("external_tokens_matched", "abc-1f", 110),
                  event("load_allocated", "abc-1f", 112),
                  event("h2d_started", "abc-1f", 115),
                  event("h2d_completed", "abc-1f", 170, ready_ns=120_000_000),
                  event("load_reported", "abc-1f", 176),
                  event("load_scheduled", "abc-1f", 180),
                  event("adapter_first_output", "abc", 195),
                  event("adapter_generate", "other", 300)]
        [row] = resume_phases(turns, events)
        self.assertEqual(row, {"total_ms": 100.0, "gateway": 4.0, "admission": 6.0, "lease": 2.0,
                               "dispatch": 3.0, "queued": 5.0, "copy": 50.0, "report": 6.0,
                               "schedule": 4.0, "prefill": 15.0, "delivery": 5.0})
        self.assertAlmostEqual(sum(v for k, v in row.items() if k != "total_ms"), row["total_ms"])

    def test_missing_marks_merge_the_phases_around_them(self):
        turns = [{"submitted": 0, "first_token": 50_000_000}]
        events = [event("adapter_generate", "abc", 2), event("h2d_completed", "abc-x", 40, ready_ns=10_000_000),
                  event("adapter_first_output", "abc", 48)]
        [row] = resume_phases(turns, events)
        self.assertEqual(row, {"total_ms": 50.0, "gateway": 2.0, "admission+lease+dispatch+queued": 8.0,
                               "copy": 30.0, "report+schedule+prefill": 8.0, "delivery": 2.0})


if __name__ == "__main__":
    unittest.main()
