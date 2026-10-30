"""Minimal episode runner: one image + one demand -> serializable session.

Same registry for both layers; caller decides the tool sequence (v0: fixed
script per demand; policy plug-in is the Gate-3 extension).
"""
from __future__ import annotations

from pathlib import Path

from ..core.registry import ToolRegistry
from ..core.session import EvidenceSession
from ..core.units import Demand
from ..perception.anatomy import SegmentAnatomy, DetectLandmarks
from ..perception.metrics import build_metric_tools
from ..perception.anatomy.tools import build_anatomy_tools
from ..perception.quality import (CarryViewAssumption, AssessInspiration,
                                  AssessRotation, AssessExposure)
from ..construction.teachers import ComposeEvidenceCTR, ComposeEvidenceSymmetry


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (SegmentAnatomy(), DetectLandmarks(),
                 CarryViewAssumption(), AssessInspiration(), AssessRotation(),
                 AssessExposure(),
                 ComposeEvidenceCTR(), ComposeEvidenceSymmetry()):
        reg.register(tool)
    for tool in build_metric_tools():
        reg.register(tool)
    for tool in build_anatomy_tools():
        reg.register(tool)
    return reg


def build_model_tools() -> list:
    """Model-backed A tools that run in isolated workers (.venv-models)."""
    from ..perception.findings import LocalizeDevices, SegmentPneumothorax
    from ..perception.landmarks import LocateLungLandmarks
    from ..perception.structures import AssessImageQuality, MeasureCTR
    from ..perception.vlm import ClassifyRoiAttributes, ClassifyView, DetectAbnormalities, GroundPhrase

    return [LocateLungLandmarks(), DetectAbnormalities(), GroundPhrase(), SegmentPneumothorax(),
            MeasureCTR(), ClassifyRoiAttributes(), ClassifyView(), AssessImageQuality(), LocalizeDevices()]


def build_a_registry() -> ToolRegistry:
    """All A1-A26 interfaces; excludes legacy E/quality placeholders."""
    reg = ToolRegistry()
    for tool in [*build_anatomy_tools(), *build_metric_tools(), *build_model_tools()]:
        reg.register(tool)
    return reg


def a_capabilities(registry: ToolRegistry) -> dict[str, list[str]]:
    """Capability id -> registered interface names."""
    table: dict[str, list[str]] = {}
    for spec in registry.describe_all():
        table.setdefault(spec['description'].split(':', 1)[0], []).append(spec['name'])
    return table


def run_ctr_episode(image_ref: str, masks_dir: str, landmarks_json: str,
                    out_dir: str) -> EvidenceSession:
    reg = build_registry()
    session = EvidenceSession(
        image_ref=str(Path(image_ref).resolve()),
        demand=Demand(proposition="是否存在心脏增大表现之证据",
                      requirements=[{"evidence_type": "region", "target": "Heart"},
                                    {"evidence_type": "region", "target": "lungs"},
                                    {"evidence_type": "landmark", "target": "heart|lungs"}]))
    reg.run("segment_anatomy", session, masks_dir=masks_dir, image_ref=image_ref)
    reg.run("detect_landmarks", session, landmarks_json=landmarks_json,
            image_ref=image_ref)
    reg.run("carry_view_assumption", session)
    reg.run("assess_inspiration", session)
    reg.run("assess_rotation", session)
    reg.run("assess_exposure", session)
    reg.run("compose_evidence_ctr", session)
    reg.run("compose_evidence_symmetry", session)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session.save(out / "session.json")
    (out / "program_traces.json").write_text(
        __import__("json").dumps(session.program_traces, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return session
