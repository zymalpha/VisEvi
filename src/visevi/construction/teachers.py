"""Teacher programs (Layer-2, rule mode): typed composition programs whose
execution is recorded as program-trace GT (L2 supervision samples, D4/D9).

Each teacher:
  1. pulls Elements from the session graph (never re-perceives: D6),
  2. runs deterministic geometry (perception/geometry.py),
  3. calibrates + discounts each source into a DS mass (construction/ds.py),
  4. combines independent sources (Dempster/Yager with first-class conflict K),
  5. writes an EvidenceClaim with [Bel, Pl] + provenance edges + program trace,
  6. runs the sufficiency check (E10-lite) -> sets session.verdict.

组合≠诊断: propositions stay at finding-evidence level.
"""
from __future__ import annotations

import numpy as np

from ..core.spec import ToolSpec, parameter_schema, evidence_result_schema
from ..core.units import EvidenceUnit, ToolRefusal, make_claim, AnswerVerdict, fov_refusal
from ..perception import geometry as geo
from . import ds

PROGRAM = "P-CTR-01"   # rule ids (teacher program library, D4)


class ComposeEvidenceCTR:
    """Teacher T-CTR (E1 family): cardiac-enlargement-pattern evidence from
    A2 heart mask + A12 CTR (mask caliber) + A5 landmark caliber (independent
    source) + view condition (explicit PA assumption while A24 unavailable)."""

    ALPHA_VIEW = ds.ALPHA["view_assumed_PA"]

    def describe(self) -> ToolSpec:
        return ToolSpec(
            name="compose_evidence_ctr", category="construction",
            description="Cardiac enlargement pattern evidence: CTR from two "
                        "independent calibers (PSPNet masks, HybridGNet landmarks), "
                        "DS combination, view carried as explicit assumption",
            input_schema=parameter_schema(), output_schema=evidence_result_schema())

    def run(self, session, **_) -> list[EvidenceUnit] | ToolRefusal:
        masks = session.masks
        missing = [k for k in ("Heart", "Right_Lung", "Left_Lung") if k not in masks]
        if missing or not session.landmarks:
            return ToolRefusal(tool="compose_evidence_ctr",
                               reason="required Layer-1 elements missing (D6: no "
                                      "re-perception; caller must run perception first)",
                               missing=[f"mask:{m}" for m in missing] or ["landmarks"])
        truncated = session.fov_truncated(("Heart", "Right_Lung", "Left_Lung"))
        if truncated:
            return fov_refusal("compose_evidence_ctr", truncated)

        steps = []
        # -- step 1: measurement, mask caliber (A12) ----------------------
        ctr_m = geo.ctr_from_masks(masks["Heart"], masks["Right_Lung"] | masks["Left_Lung"])
        m_unit = session.add_observation(EvidenceUnit(
            id=session.next_id("obs"), category="observation", evidence_type="measurement",
            finding_or_anatomy="CTR_mask_caliber",
            value={k: ctr_m[k] for k in ("CTR", "heart_max_width_px", "thorax_max_width_px",
                                         "heart_width_row_y", "thorax_width_row_y")},
            confidence=None, source_tool="compose_evidence_ctr/measure",
            provenance=session.mask_provenance(("Heart", "Right_Lung", "Left_Lung")),
            validity_conditions=[ctr_m["method"], "pixel units (no mm spacing)"]))
        steps.append({"op": "measure_ctr", "input": "masks:Heart|lung_union",
                      "output": m_unit.id})
        for u in session.graph.by_category("observation"):
            if u.evidence_type == "region":
                session.graph.add_edge(u.id, m_unit.id, "attribute_of", PROGRAM)

        # -- step 2: measurement, landmark caliber (independent source) ----
        lm = session.landmarks["lm_px"]
        (a_r, b_r), (a_l, b_l), (a_h, b_h) = (session.landmarks["groups"][g]
                                               for g in ("right_lung", "left_lung", "heart"))
        heart_x = lm[a_h:b_h, 0]
        thorax_x = np.concatenate([lm[a_r:b_r, 0], lm[a_l:b_l, 0]])
        ctr_lm = float(heart_x.max() - heart_x.min()) / float(thorax_x.max() - thorax_x.min())
        l_unit = session.add_observation(EvidenceUnit(
            id=session.next_id("obs"), category="observation", evidence_type="measurement",
            finding_or_anatomy="CTR_landmark_caliber",
            value={"CTR": round(ctr_lm, 4),
                   "method": "heart-keypoint x-extent over lung-keypoint x-extent "
                             "(independent caliber, different upstream tool)"},
            confidence=None, source_tool="compose_evidence_ctr/measure",
            provenance=[u.id for u in session.graph.by_category("observation")
                        if u.evidence_type == "landmark"],
            validity_conditions=["x-extent proxy of contour keypoints (not rib-inner "
                                 "margin); params to be calibrated"]))
        steps.append({"op": "measure_ctr_landmark", "input": "landmarks:heart|lungs",
                      "output": l_unit.id})
        for u in session.graph.by_category("observation"):
            if u.evidence_type == "landmark":
                session.graph.add_edge(u.id, l_unit.id, "attribute_of", PROGRAM)

        # -- step 3: calibrate + discount (mass construction) --------------
        alpha = self.ALPHA_VIEW
        m1 = ds.discount(ds.g_ctr(ctr_m["CTR"]), alpha * ds.ALPHA["pspnet_mask"])
        m2 = ds.discount(ds.g_ctr(ctr_lm, strength=0.7),
                         alpha * ds.ALPHA["hybridgnet_landmarks"])
        steps.append({"op": "calibrate+discount",
                      "inputs": [m_unit.id, l_unit.id],
                      "masses": {"mask_caliber": m1, "landmark_caliber": m2}})

        # -- step 4: combine (independence by provenance: PSPNet vs HybridGNet)
        m, k, rule_used = ds.combine(m1, m2)
        iv = ds.bel_pl(m)
        supported = "NOT_F" if m["NOT_F"] > m["F"] else "F"
        claim = make_claim(
            session.next_id("claim"),
            proposition="心胸比增大表现（cardiac enlargement pattern）",
            bel=iv["bel_not_f"] if supported == "NOT_F" else iv["bel_F"],
            pl=iv["pl_not_f"] if supported == "NOT_F" else iv["pl_F"],
            conflict_k=k, support_ids=[m_unit.id, l_unit.id], counter_ids=[],
            rule_ids=[PROGRAM],
            validity_conditions=[
                "view assumed PA (A24 unavailable) — explicit assumption, alpha=0.95",
                "CTR proxy definition (max row-width ratio), not rib-inner-margin CTR",
                "calibration params v0 literature placeholders, to be calibrated",
                "pixel units"],
            source_tool="compose_evidence_ctr",
            extras={"supported_direction": "absent" if supported == "NOT_F" else "present",
                    "bel_interval_full": iv, "combination_rule": rule_used,
                    "ctr_mask": round(ctr_m["CTR"], 4), "ctr_landmark": round(ctr_lm, 4)})
        session.add_claim(claim)
        for sid in (m_unit.id, l_unit.id):
            session.graph.add_edge(sid, claim.id, "supports", PROGRAM)
        steps.append({"op": "combine_verify", "inputs": [m_unit.id, l_unit.id],
                      "output": claim.id, "combination": rule_used, "conflict_k": k})
        session.trace_program({"program_id": PROGRAM, "rule": "T-CTR",
                               "steps": steps, "produced": [claim.id]})

        # -- step 5: sufficiency (E10-lite) --------------------------------
        diff = abs(iv["bel_F"] - iv["bel_not_f"])
        if diff > 0.2:
            direction = claim.value["supported_direction"]
            finding = "检出心胸比增大表现" if direction == "present" else "未检出心胸比增大表现"
            session.verdict = AnswerVerdict(
                answerable=True,
                answer=(f"证据级发现：在 PA 投照假设下，{finding}"
                        f"（支持方向 {direction}，Bel={claim.value['bel']:.2f}, "
                        f"区间 [{claim.value['bel']:.2f}, {claim.value['pl']:.2f}]）"),
                evidence_refs=[claim.id], confidence=round(diff, 4))
        else:
            session.verdict = AnswerVerdict(
                answerable=False, abstain_reason="证据区间过宽：双口径 CTR 落在校准"
                "过渡带且视角为假设，无法区分增大/不增大（missing: A24 投照识别）",
                evidence_refs=[claim.id], confidence=round(diff, 4))
        return [claim]


class ComposeEvidenceSymmetry:
    """Teacher T-SYM (E5 family): bilateral high-symmetry evidence from mirror
    Dice (A19) — a NORMAL-pattern evidence, demo of the same pipeline."""

    def describe(self) -> ToolSpec:
        return ToolSpec(
            name="compose_evidence_symmetry", category="construction",
            description="Bilateral lung symmetry evidence from mirror-Dice (A19)",
            input_schema=parameter_schema(), output_schema=evidence_result_schema())

    def run(self, session, **_) -> list[EvidenceUnit] | ToolRefusal:
        masks = session.masks
        missing = [k for k in ("Right_Lung", "Left_Lung") if k not in masks]
        if missing:
            return ToolRefusal(tool="compose_evidence_symmetry",
                               reason="required Layer-1 elements missing",
                               missing=[f"mask:{m}" for m in missing])
        truncated = session.fov_truncated(("Right_Lung", "Left_Lung"))
        if truncated:
            return fov_refusal("compose_evidence_symmetry", truncated)
        steps = []
        union = masks["Right_Lung"] | masks["Left_Lung"]
        bb = geo.bbox(union)
        mid = (bb["x0"] + bb["x1"]) / 2.0
        sym = geo.mirror_dice(masks["Right_Lung"], masks["Left_Lung"], mid)
        s_unit = session.add_observation(EvidenceUnit(
            id=session.next_id("obs"), category="observation", evidence_type="measurement",
            finding_or_anatomy="mirror_dice",
            value={"mirror_dice": round(sym["mirror_dice"], 4), "midline_x": sym["midline_x"],
                   "method": sym["method"]},
            confidence=None, source_tool="compose_evidence_symmetry/measure",
            provenance=session.mask_provenance(("Right_Lung", "Left_Lung")),
            validity_conditions=["midline = lung-union bbox center (documented caliber)"]))
        steps.append({"op": "measure_mirror_dice", "input": "masks:Right_Lung|Left_Lung",
                      "output": s_unit.id})

        m = ds.discount(ds.g_symmetry(sym["mirror_dice"]), ds.ALPHA["pspnet_mask"])
        iv = ds.bel_pl(m)
        claim = make_claim(
            session.next_id("claim"),
            proposition="双肺高度对称（normal-pattern evidence）",
            bel=iv["bel_F"], pl=iv["pl_F"], conflict_k=0.0,
            support_ids=[s_unit.id], counter_ids=[], rule_ids=["P-SYM-01"],
            validity_conditions=["single source (no independent caliber yet — "
                                 "landmark symmetry is a Gate-1 increment)",
                                 "symmetry thresholds v0 placeholders"],
            source_tool="compose_evidence_symmetry",
            extras={"supported_direction": "present" if m["F"] > m["NOT_F"] else "absent",
                    "combination_rule": "single_source", "dice": round(sym["mirror_dice"], 4)})
        session.add_claim(claim)
        session.graph.add_edge(s_unit.id, claim.id, "supports", "P-SYM-01")
        steps.append({"op": "calibrate+combine_verify", "inputs": [s_unit.id],
                      "output": claim.id})
        session.trace_program({"program_id": "P-SYM-01", "rule": "T-SYM",
                               "steps": steps, "produced": [claim.id]})
        return [claim]
