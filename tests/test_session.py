"""Unit tests: session serialization round-trip + registry protocol (D2)."""
import json
import tempfile
import unittest
from pathlib import Path

from visevi.core.registry import ToolRegistry
from visevi.core.session import EvidenceSession
from visevi.core.spec import ToolSpec
from visevi.core.units import EvidenceUnit, Demand, ToolRefusal


class _FakeTool:
    def describe(self):
        return ToolSpec(name="fake", category="perception", description="t",
                        input_schema={}, output_schema={})

    def run(self, session, **params):
        if params.get("boom"):
            return ToolRefusal(tool="fake", reason="test refusal")
        if params.get("crash"):
            raise RuntimeError("internal failure")
        u = EvidenceUnit(id=session.next_id("obs"), category="observation",
                         evidence_type="region", finding_or_anatomy="x", value={"a": 1})
        session.add_observation(u)
        return [u]


class TestSession(unittest.TestCase):
    def _session(self):
        s = EvidenceSession(image_ref="img.png", demand=Demand(proposition="p"))
        s.add_observation(EvidenceUnit(id="obs-0001", category="observation",
                                       evidence_type="region", finding_or_anatomy="Heart",
                                       value={"area_px": 10}, source_tool="fake"))
        s.graph.add_edge("obs-0001", "obs-0002", "supports", "R-TEST")
        s.trace_call(tool="fake", params={}, latency_ms=1.0, result="ok",
                     detail={"ids": ["obs-0001"]})
        return s

    def test_round_trip(self):
        s = self._session()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            s.save(p)
            loaded = EvidenceSession.load(p)
        self.assertEqual(loaded.image_ref, s.image_ref)
        self.assertEqual(loaded.demand.proposition, "p")
        self.assertIn("obs-0001", loaded.graph.nodes)
        self.assertEqual(loaded.graph.edges[0]["basis"], "R-TEST")
        self.assertEqual(len(loaded.trace), 1)
        # runtime caches stay out of serialization
        self.assertEqual(loaded.to_dict(), s.to_dict())

    def test_registry_traces_and_refusal(self):
        reg = ToolRegistry()
        reg.register(_FakeTool())
        s = EvidenceSession(image_ref="img.png")
        out = reg.run("fake", s)
        self.assertEqual(out[0].id, "obs-0001")
        self.assertEqual(s.trace[-1]["result"], "ok")
        ref = reg.run("fake", s, boom=True)
        self.assertIsInstance(ref, ToolRefusal)
        self.assertEqual(s.trace[-1]["result"], "refusal")

    def test_registry_turns_tool_exceptions_into_traced_refusals(self):
        reg = ToolRegistry()
        reg.register(_FakeTool())
        s = EvidenceSession(image_ref="img.png")
        out = reg.run("fake", s, crash=True)
        self.assertIsInstance(out, ToolRefusal)
        self.assertIn("RuntimeError", out.reason)
        self.assertEqual(s.trace[-1]["result"], "error")
        with self.assertRaises(KeyError):
            reg.run("missing", s)

    def test_describe_all_is_jsonable(self):
        reg = ToolRegistry()
        reg.register(_FakeTool())
        dumped = json.dumps(reg.describe_all())
        self.assertIn('"fake"', dumped)
        self.assertIn('"perception"', dumped)


if __name__ == "__main__":
    unittest.main()
